# Memory Debugging with FSDP

This guide explains how to use the memory snapshot and visualization features for debugging GPU memory leaks during FSDP training.

## Overview

The memory debugging feature provides:
- Automatic memory history recording
- Periodic memory snapshots during training
- Memory snapshots at key training points
- Support for `torch.memory_viz` visualization

## Configuration

### Command Line Arguments

```bash
python train.py \
    --enable-memory-visualize \
    --memory-snapshot-out-dir ./debug_snapshots \
    --memory-snapshot-interval 50 \
    ... # other args
```

### YAML Configuration

```yaml
enable_memory_visualize: true
memory_snapshot_out_dir: "./debug_snapshots"
memory_snapshot_interval: 50  # Dump every 50 training steps (0 to disable)
```

### Configuration Parameters

- `enable_memory_visualize` (bool): Enable memory history recording and snapshot dumping
- `memory_snapshot_out_dir` (str): Directory to save memory snapshots (default: "./mem_snapshots")
- `memory_snapshot_interval` (int): Dump snapshot every N training steps (default: 100, 0 to disable periodic dumps)

## Memory Snapshot Points

Memory snapshots are automatically captured at these key points:

1. **Model Initialization**: After FSDP model creation (`model_init`)
2. **Training Start**: Beginning of each training step (`train_start_rollout{rollout_id}`)
3. **Training End**: End of each training step (`train_end_rollout{rollout_id}`)
4. **Weight Update**: Before and after weight updates (`before_weight_update`, `after_weight_update`)
5. **Periodic**: Every N training steps (configurable via `memory_snapshot_interval`)

## Output Structure

```
debug_snapshots/
├── key_points/
│   ├── memory_snapshot_rank0_model_init_1234567890.pickle
│   ├── memory_snapshot_rank0_before_weight_update_1234567890.pickle
│   └── memory_snapshot_rank0_after_weight_update_1234567890.pickle
├── step_50/
│   └── memory_snapshot_rank0_train_end_rollout50_step50_1234567890.pickle
└── step_100/
    └── memory_snapshot_rank0_train_end_rollout100_step100_1234567890.pickle
```

## Analyzing Memory Snapshots

### Using torch.memory_viz

Install the memory visualization tool:
```bash
pip install torch-memory-viz
```

Visualize memory usage:
```python
import pickle
import torch.memory_viz as memory_viz

# Load snapshot
with open('memory_snapshot_rank0_model_init_1234567890.pickle', 'rb') as f:
    snapshot = pickle.load(f)

# Generate visualization
memory_viz.plot_snapshot(snapshot, show=True)
```

### Comparing Snapshots

Compare memory usage between different points:
```python
# Load two snapshots
with open('before_weight_update.pickle', 'rb') as f:
    snapshot_before = pickle.load(f)
    
with open('after_weight_update.pickle', 'rb') as f:
    snapshot_after = pickle.load(f)

# Compare memory growth
memory_viz.compare_snapshots(snapshot_before, snapshot_after)
```

## Common Memory Leak Patterns

### 1. Gradual Memory Growth
Check periodic snapshots to identify gradual memory increases:
```bash
ls -la debug_snapshots/step_*/
```

### 2. Weight Update Leaks
Compare before/after weight update snapshots:
```bash
ls -la debug_snapshots/key_points/*weight_update*
```

### 3. Training Loop Leaks
Compare training start/end snapshots for the same rollout:
```bash
ls -la debug_snapshots/key_points/*train_start* debug_snapshots/key_points/*train_end*
```

## Best Practices

1. **Enable Only When Needed**: Memory recording has overhead, enable only for debugging
2. **Use Periodic Sampling**: Set appropriate interval (50-100 steps) to balance detail vs. storage
3. **Monitor Disk Space**: Snapshots can be large, especially for big models
4. **Compare Key Points**: Focus on before/after comparisons at critical operations
5. **Use Rank 0**: Memory patterns are usually consistent across ranks, focus on rank 0

## Example Usage

### Basic Memory Debugging
```bash
python train.py \
    --enable-memory-visualize \
    --memory-snapshot-interval 100 \
    ... # other training args
```

### Detailed Memory Debugging
```bash
python train.py \
    --enable-memory-visualize \
    --memory-snapshot-out-dir ./detailed_debug \
    --memory-snapshot-interval 10 \
    ... # other training args
```

### Analysis Script
```python
import os
import pickle
import torch.memory_viz as memory_viz
from pathlib import Path

def analyze_memory_growth(snapshot_dir):
    """Analyze memory growth from periodic snapshots."""
    step_dirs = sorted([d for d in Path(snapshot_dir).iterdir() if d.is_dir() and d.name.startswith('step_')])
    
    memory_usage = []
    for step_dir in step_dirs:
        snapshot_files = list(step_dir.glob('*.pickle'))
        if snapshot_files:
            with open(snapshot_files[0], 'rb') as f:
                snapshot = pickle.load(f)
            
            total_memory = sum(trace.get('size', 0) for trace in snapshot)
            step_num = int(step_dir.name.split('_')[1])
            memory_usage.append((step_num, total_memory))
    
    # Plot memory growth
    steps, memory = zip(*memory_usage)
    print(f"Memory growth from step {steps[0]} to {steps[-1]}: {memory[-1] - memory[0]} bytes")
    
    return memory_usage

# Usage
memory_growth = analyze_memory_growth('./debug_snapshots')
```

## Troubleshooting

### Memory Recording Not Available
If you see "Warning: torch.cuda.memory.memory_snapshot not available":
- Ensure PyTorch version >= 1.10
- Check CUDA availability
- Verify GPU memory recording support

### Large Snapshot Files
- Reduce `memory_snapshot_interval` to capture fewer snapshots
- Focus on specific problematic areas
- Use `memory_snapshot_out_dir` on fast storage

### Performance Impact
- Memory recording adds ~5-10% overhead
- Disable periodic sampling during production runs
- Use only for targeted debugging sessions