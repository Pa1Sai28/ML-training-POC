# 03 — Mixed Precision Benchmarking

## Question this module answers
**"Does training in BF16 change speed, memory, or convergence compared to FP32 — and what actually has to stay in FP32?"**

This module extends `01_single_gpu_baseline/train.py` with PyTorch's `torch.autocast` context manager, comparing FP32 (full precision) against
BF16 (Brain Float 16) on the same model, same data, same seed.

## What's new vs Module 01

| Concept | Module 01 | Module 03 |
|---|---|---|
| Precision | FP32 throughout | FP32 weights, BF16 forward pass (when `--precision bf16`) |
| Forward pass | Plain | Wrapped in `torch.autocast(device_type="cpu", dtype=torch.bfloat16)` |
| Backward / optimizer | FP32 | FP32 (unchanged — autocast doesn't touch these) |

## Run it

```bash
python 03_mixed_precision/train_bf16.py --precision fp32 --steps 200
python 03_mixed_precision/train_bf16.py --precision bf16 --steps 200
```

## Results (M2 MacBook Air, CPU, seed=42, 200 steps)

| Precision | Final loss | tok/s | Total time |
|---|---|---|---|
| FP32 | 2.6126 | ~38,000 | 21.8s |
| BF16 | 2.6126 | ~2,450 | 334.4s |

**Loss curves were nearly identical** — confirming that autocast correctly keeps the master weights and loss computation in FP32, so convergence is unaffected by the lower-precision forward pass.

**BF16 was ~16x SLOWER**, not faster.

## Why BF16 was slower, not faster

This was the real finding of this module. Mixed precision speedups come from **dedicated low-precision compute hardware** — NVIDIA Tensor Cores, TPUs, AWS Trainium/Inferentia chips. These accelerators can execute BF16 matrix multiplies at higher throughput than FP32.

Apple M-series CPUs have **no native BF16 compute path**. When `autocast` casts operations to BF16, PyTorch emulates BF16 arithmetic on top of FP32 hardware — which costs more than just running FP32 directly.

**The takeaway**: mixed precision isn't a free win from "smaller numbers move faster." It's a hardware-dependent optimization that only pays off when the accelerator has a fast path for the lower-precision dtype. On AWS Trainium — which has dedicated BF16/FP16 compute units — I'd expect to see the speedup that didn't materialize here on CPU.

## What stayed in FP32 (and why)

`autocast` automatically keeps certain operations in FP32 even inside the
`bf16` context:
- **LayerNorm** — sensitive to precision in normalization statistics
- **Softmax** — exponentials amplify small errors
- **Cross-entropy loss** — the final loss value needs to be precise for
  stable gradients

Only the large matrix multiplies (attention projections, feedforward layers) are actually computed in BF16. Model weights and gradients remain
FP32 throughout — confirmed by the matching loss values.

## Key files

- `train_bf16.py` — training script with `--precision fp32|bf16` flag
- `results/metrics_fp32.json`, `results/metrics_bf16.json` — saved metrics
  from each run

## Key takeaway

Precision and hardware are inseparable. The same dtype conversion that gives a 2-4x speedup on a GPU/TPU/Trainium chip can be a 16x *slowdown* on a CPU with no native low-precision compute units. Before optimizing for a dtype, the actual hardware's compute paths matter more than the dtype's theoretical properties.