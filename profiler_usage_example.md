# Memory Profiler Usage - Simple Setup

Configuration-based memory profiler similar to veRL PR #3099.

## Setup

1. Create `profiler_config.yaml`:
```yaml
global_profiler:
  tool: torch_memory
  save_path: ./mem_snapshots
  trace_alloc_max_entries: 100000
  stack_depth: 32
  context: all
  stacks: all
```

2. Run training:
```bash
./test_fsdp_colocated_2GPU.sh
```

## Output

Memory snapshots saved to `./mem_snapshots/`:
- `before_weight_update_rank0_<timestamp>.pickle`
- `after_preprocessing_rank0_<timestamp>.pickle`
- `after_weight_update_rank0_<timestamp>.pickle`
- `periodic_rank0_<timestamp>.pickle`

## Analysis

Load snapshot with Python:
```python
import pickle
with open('mem_snapshots/snapshot.pickle', 'rb') as f:
    data = pickle.load(f)
```

Use with `torch.memory_viz` for visualization.