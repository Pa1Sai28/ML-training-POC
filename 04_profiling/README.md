# 04 — Profiling

## Question this module answers
**"Where does the time in a training step actually go?"**

This module extends `01_single_gpu_baseline/train.py` with `torch.profiler`, wrapping a representative sample of training steps to measure per-operation CPU time and memory, after running unprofiled warmup steps to avoid one-time startup costs skewing the results.

## What's new vs Module 01

| Concept | Module 01 | Module 04 |
|---|---|---|
| Instrumentation | None | `torch.profiler.profile()` wrapping steps |
| Labeling | N/A | `record_function()` around each training phase |
| Output | Loss/throughput logs | Profiler summary table + Chrome trace JSON |
| Warmup | N/A | Separate unprofiled warmup steps before profiling |

## Run it

```bash
python 04_profiling/train_profiled.py --dropout 0.1   # default model
python 04_profiling/train_profiled.py --dropout 0.0   # dropout removed, for comparison
```

Outputs land in `results/`:
- `profile_summary_dropout{X}.txt` — top-15 operations by CPU time
- `trace_dropout{X}.json` — Chrome trace, viewable at `chrome://tracing` or
  [ui.perfetto.dev](https://ui.perfetto.dev)

## Results (M2 MacBook Air, CPU, seed=42, 20 profiled steps after 10 warmup steps)

| | dropout=0.1 | dropout=0.0 |
|---|---|---|
| Total CPU time | 2.366s | 1.642s |
| forward | 52.6% | 35.1% |
| backward | 43.8% | 60.2% |
| `aten::dropout` | 28.3% | — |
| `aten::bernoulli_` (dropout RNG) | 26.1% | — |
| `aten::mm` (matmul) | 22.4% | 31.9% |
| `aten::bmm` (batch matmul, attention) | 12.4% | 18.1% |

## The big finding: dropout's RNG cost almost as much as matmul

With the default `dropout=0.1`, `aten::bernoulli_` — the random number generation used to build dropout masks — consumed **26.1%** of total CPU
time, nearly matching `aten::mm` (matmul, 22.4%) and exceeding `aten::bmm` (attention matmul, 12.4%) individually.

Removing dropout entirely (`dropout=0.0`) cut total step time by **30.6%** (2.366s → 1.642s). The matmul operations themselves took roughly the same absolute time either way (~524ms for `aten::mm`) — what changed is that dropout's RNG overhead disappeared completely, shrinking the total and making matmul a *larger share* of a *smaller* total.

## Why this matters

Intuition says "the model does matmuls, so matmuls are the bottleneck." The profiler shows a different story at this scale: a seemingly minor
operation (generating random dropout masks) costs nearly as much as the core linear algebra. This is almost certainly a **small-model artifact** — at production LLM scale, matmuls would dominate by orders of magnitude and dropout would be negligible. But it demonstrates the core value of profiling: verify where time goes, don't assume it from the architecture diagram.

## forward vs backward shift

With dropout: forward (52.6%) > backward (43.8%) — dropout's RNG cost is counted inside the forward pass.

Without dropout: backward (60.2%) > forward (35.1%) — once dropout's forward-pass overhead is removed, backward pass (which has its own matmul
gradients, `GeluBackward0`, `BmmBackward0`, etc.) becomes the larger share.

## Key files

- `train_profiled.py` — training script with `record_function` labels and
  `torch.profiler` wrapping
- `results/profile_summary_dropout*.txt` — text summaries
- `results/trace_dropout*.json` — Chrome trace files for visual timeline inspection

## Key takeaway

Profiling reveals costs that architectural reasoning alone misses. On this M2 CPU, dropout's random mask generation rivaled matrix multiplication in cost — something no amount of "thinking about the forward pass" would predict. On Trainium, the equivalent exercise would be identifying which operations don't map efficiently onto the chip's compute engines — the profiler is the tool that turns "I think X is slow" into "X is actually 26% of total time, here's the trace."