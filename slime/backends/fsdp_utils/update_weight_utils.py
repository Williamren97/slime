import ray
import torch
import torch.distributed as dist
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType


class UpdateWeightFromTensor:
    def __init__(self, args, model):
        self.args = args
        self.model = model

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(
                ranks=group_ranks,
                backend="gloo",
            )
            if dist.get_rank() in group_ranks:
                self._ipc_gather_src = start_rank
                self._ipc_gather_group = new_group
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self):
        """Optimized weight update using SHARDED_STATE_DICT to avoid memory explosion."""
        monkey_patch_torch_reductions()
        
        # Use SHARDED_STATE_DICT to avoid loading full model on each GPU
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            local_state_dict = self.model.state_dict()
        
        # Gather all shards to reconstruct full parameters
        all_shards = [None] * dist.get_world_size(self._ipc_gather_group)
        dist.all_gather_object(all_shards, local_state_dict, group=self._ipc_gather_group)
        
        if dist.get_rank() == self._ipc_gather_src:
            # Reconstruct full parameters from shards
            full_params = {}
            for shard in all_shards:
                if shard is not None:
                    full_params.update(shard)
            
            # Initialize distributed communication group for weight updates
            group_name = f"fsdp_update_group_{id(self)}"
            for engine in self.rollout_engines:
                # Setup distributed group
                ref = engine.init_weights_update_group.remote(
                    master_address="localhost",
                    master_port=0,
                    rank_offset=0,
                    world_size=dist.get_world_size(self._ipc_gather_group),
                    group_name=group_name,
                    backend="nccl" if torch.cuda.is_available() else "gloo"
                )
                ray.get(ref)
            
            # Send parameter metadata to SGLang
            for engine in self.rollout_engines:
                ref = engine.update_weights_from_distributed.remote(
                    names=list(full_params.keys()),
                    dtypes=[param.dtype for param in full_params.values()],
                    shapes=[param.shape for param in full_params.values()],
                    group_name=group_name,
                    flush_cache=True
                )
                ray.get(ref)
            
            # Broadcast actual parameter data
            for param in full_params.values():
                dist.broadcast(param.data, src=self._ipc_gather_src, group=self._ipc_gather_group)
