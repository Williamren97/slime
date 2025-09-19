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
        """
        Optimized weight update using SHARDED_STATE_DICT to avoid memory explosion.
        
        Memory optimization:
        - Before: All GPUs load full model (e.g., 8 × 70B = 560B memory)
        - After: Each GPU only handles local shards (e.g., 8 × 8.75B = 70B memory)
        """
        monkey_patch_torch_reductions()
        
        # Use SHARDED_STATE_DICT to avoid loading full model on each GPU
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            local_state_dict = self.model.state_dict()
        
        # Extract parameter metadata for distributed weight update
        param_names = list(local_state_dict.keys())
        param_dtypes = [param.dtype for param in local_state_dict.values()]
        param_shapes = [param.shape for param in local_state_dict.values()]
        
        # Only the gather source rank communicates with SGLang engines
        if dist.get_rank() == self._ipc_gather_src:
            # Use SGLang's distributed weight update for memory-efficient parameter sharing
            for engine in self.rollout_engines:
                ref = engine.update_weights_from_distributed.remote(
                    names=param_names,
                    dtypes=param_dtypes,
                    shapes=param_shapes,
                    group_name=f"fsdp_update_group_{id(self)}",
                    flush_cache=True  # Ensure cache is cleared after update
                )
                ray.get(ref)
