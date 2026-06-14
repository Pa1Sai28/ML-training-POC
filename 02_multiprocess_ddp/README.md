# 02 — Multiprocess DDP

## Question this module answers
**"What changes when training is split across multiple processes?"**

This module extends `01_single_gpu_baseline/train.py` with PyTorch's `DistributedDataParallel` (DDP). Instead of one process training on the
full dataset, N processes each train on a different slice of the data, and gradients are synchronized (averaged) after every backward pass so
all copies of the model stay identical.

On an M2 MacBook Air there's no multi-GPU setup, so this simulates DDP using multiple **CPU processes** with the `gloo` backend. The mechanics
— process groups, gradient AllReduce, distributed samplers — are identical to what happens across multiple GPUs or AWS Trainium chips.Only the communication backend changes.

## What's new vs Module 01

| Concept | Module 01 | Module 02 |
|---|---|---|
| Processes | 1 | N (configurable via `--world_size`) |
| Data split | Full dataset, shuffled | `DistributedSampler` — each process gets a non-overlapping slice |
| Model wrapping | Plain `nn.Module` | Wrapped in `DistributedDataParallel` |
| Gradient sync | N/A | Automatic AllReduce after `loss.backward()` |
| Logging | Every step | Rank 0 only (otherwise N duplicate logs) |
| Launch | `python train.py` | `mp.spawn()` launches N processes |

## Run it

```bash
python 02_multiprocess_ddp/train_ddp.py --world_size 2 --steps 200
python 02_multiprocess_ddp/train_ddp.py --world_size 1 --steps 200   # baseline comparison
```

## Results (M2 MacBook Air, CPU, seed=42)

| World size | Final loss (200 steps) | tok/s/proc | Total time |
|---|---|---|---|
| 1 | 2.5820 | ~38,800 | 21.9s |
| 2 | 2.6154 | ~23,400 | 35.1s |

**Total throughput**: world_size=1 → ~38,800 tok/s. world_size=2 → ~46,800 tok/s
(2 × 23,400). That's roughly a **1.2x speedup**, not 2x.

## Why isn't it 2x?

Two reasons, both real-world distributed training lessons:

1. **Communication overhead** — every step now includes a gradient AllReduce across processes, which didn't exist in the single-process case.
2. **Core contention** — the M2 Air has a limited number of CPU cores.Two processes competing for the same cores means each one individually
   runs slower than it would alone.

This is exactly why, in real distributed training, the interconnect speed between devices (NVLink for GPUs, or chip-to-chip interconnect for AWS Trainium/Neuron) matters so much — it determines how much of that "ideal Nx speedup" you actually get to keep.

## Key files

- `train_ddp.py` — training script with DDP setup, `DistributedSampler`,
  and `mp.spawn()` launch
- `results/metrics.json` — saved metrics from the most recent run (rank 0 only)

## Key takeaway

DDP's core promise is **identical models, different data**. Every process starts with the same weights (same seed before model construction), sees a different slice of data, computes its own gradients, and then those gradients are averaged across all processes before the optimizer step. The result: every process ends the step with identical updated weights — no explicit weight-syncing required after the initial broadcast.