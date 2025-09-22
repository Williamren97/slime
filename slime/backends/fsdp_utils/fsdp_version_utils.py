"""
FSDP v2 Utilities

This module provides utilities for FSDP v2 (fully_shard) support only.
FSDP v1 is no longer supported and will raise errors if detected.
"""

import torch
from torch.distributed.tensor import DTensor

# Note: We only support FSDP v2, FSDP v1 support removed


def create_fsdp_v2_model(model):
    """Create an FSDP v2 model using fully_shard."""
    import logging
    from packaging import version
    
    # Import FSDP v2 components based on PyTorch version
    if version.parse(torch.__version__) >= version.parse("2.6"):
        from torch.distributed.fsdp import fully_shard
    elif version.parse(torch.__version__) >= version.parse("2.4"):
        from torch.distributed._composable.fsdp import fully_shard
    else:
        fully_shard = None
    
    # Check if FSDP v2 is available
    if not (fully_shard is not None and version.parse(torch.__version__) >= version.parse("2.4")):
        raise RuntimeError(
            f"FSDP v2 is not available in PyTorch {torch.__version__}. "
            "Please upgrade to PyTorch >= 2.4 to use FSDP v2."
        )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Using FSDP v2 (fully_shard) with PyTorch {torch.__version__}")
    # FSDP v2 using fully_shard (produces DTensor objects)
    return fully_shard(model)


def preprocess_tensor_for_update_weights(tensor):
    """
    Preprocess tensor for weight updates - FSDP v2 only (DTensor support).
    
    Args:
        tensor: The tensor to preprocess (DTensor or regular Tensor)
        
    Returns:
        torch.Tensor: Regular tensor ready for weight updates
    """
    if isinstance(tensor, DTensor):
        # FSDP v2 case - convert DTensor to full tensor
        return tensor.full_tensor()
    
    # Regular tensor - return as is
    return tensor
