# understanding-llm-training

**"I wanted to understand how LLMs actually train at scale — so I built it from scratch, one concept at a time, on an M2 MacBook Air."**

This is a learning-in-public repo. Each module adds exactly one layer of complexity, with the goal of understanding every line before moving forward. No copy-pasting frameworks without knowing what they do.

Running entirely on Apple Silicon (M2) — no cloud GPU required for any module, including the distributed and profiling ones.

---

## Modules

| # | Question | Concepts | Status |
|---|----------|----------|--------|
| [01](01_single_gpu_baseline/) | What does one training step actually do? | Training loop, attention, residuals, AdamW, grad clipping, LR warmup | ✅ Done |
| [02](02_multiprocess_ddp/) | What changes when you add a second process? | DDP, gradient AllReduce, DistributedSampler, rank/world_size | ✅ Done |
| [03](03_mixed_precision/) | Does precision actually matter? | autocast, FP32 vs BF16 on CPU | ✅ Done |
| [04](04_profiling/) | Where does the time actually go? | torch.profiler, record_function, Chrome traces | ✅ Done |

---

## What I actually found (the short version)

Every module started with an intuition, and every intuition was at least partially wrong until I measured it:

- **DDP with 2 processes**: expected ~2x speedup, got **~1.2x** — communication overhead and CPU core contention eat into the ideal scaling.
- **BF16 mixed precision**: expected 2-3x speedup like MPS gave in Module 01, got **16x SLOWER** — Apple CPUs have no native BF16 compute path, so PyTorch emulates it on top of FP32 hardware.
- **Profiling a training step**: expected matmuls to dominate, found that dropout's random number generation (26% of CPU time) nearly matched
  matmul (22%)** — removing dropout cut total step time by 30%.

The common thread: hardware and workload specifics determine the answer not general rules of thumb. That's the habit this repo was built to practice.

---

## Why M2 Mac?

Most distributed training tutorials assume CUDA (NVIDIA GPU). The M2 has unified memory architecture with Apple's Metal Performance Shaders backend (`mps`) — genuinely different from CUDA in interesting ways, and forces you to understand what's actually happening rather than relying on pre-configured cloud scripts.

Real constraints hit along the way (documented in [BUGS.md](BUGS.md) and
[notes/WHAT_I_LEARNED.md](notes/WHAT_I_LEARNED.md)):

- MPS doesn't support `num_workers > 0` in DataLoader (process boundary issue)
- A training divergence bug traced back to a missing random seed, not
  hardware — reproducibility has to come first
- `torch.distributed` on M2 uses the `gloo` backend (CPU-based), not `nccl`
- BF16 "mixed precision" can be *slower* than FP32 without dedicated
  low-precision hardware
- Profiling revealed dropout's RNG cost rivals matrix multiplication at
  small model scale

---

## Setup

```bash
git clone git@github.com:Pa1Sai28/ML-training-POC.git
cd ML-training-POC
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Each module is self-contained and runnable from the repo root:

```bash
python 01_single_gpu_baseline/train.py --steps 500 --device mps
python 02_multiprocess_ddp/train_ddp.py --world_size 2 --steps 200
python 03_mixed_precision/train_bf16.py --precision bf16 --steps 200
python 04_profiling/train_profiled.py --dropout 0.1
```

That's it. No CUDA drivers, no cloud account, no expensive setup.

---

## Notes

- [BUGS.md](BUGS.md) — real issues encountered and how they were debugged
- [notes/WHAT_I_LEARNED.md](notes/WHAT_I_LEARNED.md) — running notes,
  surprises, and a question-driven log for each module

---

*Built while preparing for ML systems engineering roles. Learning in public.*