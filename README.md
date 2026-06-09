# understanding-llm-training

**"I wanted to understand how LLMs actually train at scale — so I built it
from scratch, one concept at a time, on an M2 MacBook Air."**

This is a learning-in-public repo. Each module adds exactly one layer of
complexity, with the goal of understanding every line before moving forward.
No copy-pasting frameworks without knowing what they do.

Running entirely on Apple Silicon (M2) — no cloud GPU required for the first
three modules.

---

## Modules

| # | Question | Concepts | Status |
|---|----------|----------|--------|
| [01](01_single_gpu_baseline/) | What does one training step actually do? | Training loop, attention, residuals, AdamW, grad clipping, LR warmup | ✅ Done |
| [02](02_multiprocess_ddp/) | What changes when you add a second process? | DDP, all-reduce, DistributedSampler, rank/world_size | 🔨 In progress |
| [03](03_mixed_precision/) | Does precision actually matter? FP32 vs BF16 benchmarked | autocast, GradScaler, BF16 vs FP16 tradeoffs | 🔜 Next |
| [04](04_profiling/) | Where is the time actually going? | torch.profiler, Chrome trace, bottleneck identification | 🔜 Next |

---

## Why M2 Mac?

Most distributed training tutorials assume CUDA (NVIDIA GPU). The M2 has a
unified memory architecture with Apple's Metal Performance Shaders backend
(`mps`) — genuinely different from CUDA in interesting ways.

Constraints I hit (and documented):
- MPS doesn't support `num_workers > 0` in DataLoader
- `torch.distributed` on M2 uses the `gloo` backend (not `nccl`)
- BF16 support on MPS is newer and has some op coverage gaps

These are real engineering constraints. Working around them teaches you more
than running pre-configured scripts on cloud GPUs.

---

## Setup

```bash
git clone https://github.com/Pa1Sai28/understanding-llm-training
cd understanding-llm-training
pip install torch matplotlib
```

That's it. No CUDA drivers, no cloud account, no expensive setup.

---

## Notes

- [WHAT_I_LEARNED.md](notes/WHAT_I_LEARNED.md) — running notes, surprises, things that broke
- [INTERVIEW_PREP.md](notes/INTERVIEW_PREP.md) — questions I'm using to test my own understanding

---

*Built while preparing for ML systems engineering roles. Learning in public.*
