"""
FSDP Version Detection and Compatibility Utilities

This module provides utilities for detecting FSDP versions and handling
compatibility differences between FSDP v1 and FSDP v2.
"""

import logging
from contextlib import nullcontext
from packaging import version
import torch
from torch.distributed.tensor import DTensor

# FSDP v2 imports (like veRL)
from torch.distributed.fsdp.api import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

# Import FSDP v2 components based on PyTorch version (like veRL)
if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
else:
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy = None, None, None, None

# Keep ShardedTensor import for backward compatibility
try:
    from torch.distributed._shard.sharded_tensor import ShardedTensor
except ImportError:
    ShardedTensor = None

logger = logging.getLogger(__name__)


def fsdp_version(model):
    """
    Detect FSDP version like veRL does.
    
    Args:
        model: The FSDP model to check
        
    Returns:
        int: 1 for FSDP v1, 2 for FSDP v2, 0 for non-FSDP models
    """
    if isinstance(model, FSDP):
        return 1
    elif FSDPModule is not None and isinstance(model, FSDPModule):
        return 2
    else:
        return 0


def is_fsdp_v2_available():
    """
    Check if FSDP v2 (fully_shard) is available in the current PyTorch version.
    
    Returns:
        bool: True if FSDP v2 is available, False otherwise
    """
    return fully_shard is not None and version.parse(torch.__version__) >= version.parse("2.4")


def create_fsdp_model(model, args, auto_wrap_policy=None):
    """
    Create an FSDP model using the appropriate version (v1 or v2).
    
    Args:
        model: The base model to wrap
        args: Arguments containing FSDP configuration
        auto_wrap_policy: Auto wrap policy for FSDP
        
    Returns:
        The FSDP-wrapped model
    """
    if is_fsdp_v2_available():
        logger.info(f"Using FSDP v2 (fully_shard) with PyTorch {torch.__version__}")
        # FSDP v2 using fully_shard (produces DTensor objects)
        return fully_shard(model)
    else:
        logger.info(f"Using FSDP v1 with PyTorch {torch.__version__}")
        # FSDP v1 (produces ShardedTensor objects in sharded mode)
        return FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            sharding_strategy=getattr(args, 'fsdp_sharding_strategy', 'FULL_SHARD'),
            cpu_offload=getattr(args, 'fsdp_cpu_offload', None),
            forward_prefetch=getattr(args, 'fsdp_forward_prefetch', True),
            backward_prefetch=getattr(args, 'fsdp_backward_prefetch', 'BACKWARD_PRE'),
            limit_all_gathers=getattr(args, 'fsdp_limit_all_gathers', True),
        )


def get_fsdp_state_dict_context(model, full_state_dict=True):
    """
    Get the appropriate context manager for FSDP state dict operations.
    
    Args:
        model: The FSDP model
        full_state_dict: Whether to get full state dict
        
    Returns:
        Context manager for state dict operations
    """
    model_fsdp_version = fsdp_version(model)
    
    if model_fsdp_version == 1 and full_state_dict:
        # FSDP v1 - use legacy context manager for full state dict
        return FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT)
    else:
        # FSDP v2 or sharded mode - no context manager needed
        return nullcontext()


def preprocess_tensor_for_update_weights(tensor):
    """
    Preprocess tensor for weight updates, handling different tensor types.
    
    Args:
        tensor: The tensor to preprocess (DTensor, ShardedTensor, or regular Tensor)
        
    Returns:
        torch.Tensor: Regular tensor ready for weight updates
    """
    if isinstance(tensor, DTensor):
        # FSDP v2 case (like veRL)
        return tensor.full_tensor()
    elif ShardedTensor is not None and isinstance(tensor, ShardedTensor):
        # FSDP v1 case - need to convert ShardedTensor to regular tensor
        logger.info(f"Converting ShardedTensor to full tensor for tensor with shape {tensor.size()}")
        
        # Get metadata about the sharding
        full_shape = tensor.size()
        
        # Create a full tensor with the correct shape
        full_tensor = torch.zeros(full_shape, dtype=tensor.dtype, device='cuda')
        
        # Get local shards and place them in the correct positions
        local_shards = tensor.local_shards()
        for shard in local_shards:
            # Get the shard's offset and size from metadata
            shard_metadata = shard.metadata
            offsets = shard_metadata.shard_offsets
            sizes = shard_metadata.shard_sizes
            
            # Create slices for each dimension
            slices = []
            for i, (offset, size) in enumerate(zip(offsets, sizes)):
                slices.append(slice(offset, offset + size))
            
            # Place the local shard data in the correct position of the full tensor
            full_tensor[tuple(slices)] = shard.tensor
        
        return full_tensor
    
    # Regular tensor - return as is
    return tensor


