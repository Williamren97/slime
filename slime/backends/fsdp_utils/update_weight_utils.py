import ray
import torch
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType


class UpdateWeightFromTensor:
    def __init__(self, args, model):
        self.args = args
        self.model = model

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines
        # Each GPU will send its shard directly to its corresponding engine
        self._ipc_engine = rollout_engines[0]  # Simplified: use first engine for now

    @torch.no_grad()
    def update_weights(self):
        """Optimized weight update using SHARDED_STATE_DICT with SGLang's native shard support."""
        monkey_patch_torch_reductions()
        
        # Each GPU gets its own FSDP shard - no memory explosion
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            local_state_dict = self.model.state_dict()
        
        # Serialize local shard
        serialized_state_dict = MultiprocessingSerializer.serialize(
            [(name, tensor) for name, tensor in local_state_dict.items()], 
            output_str=True
        )
        
        # Send shard directly to SGLang - it handles aggregation internally
        ref = self._ipc_engine.update_weights_from_tensor.remote(
            serialized_named_tensors=[serialized_state_dict],
            load_format="sharded_state_dict",
            flush_cache=True
        )
        ray.get(ref)
