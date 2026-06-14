# Findings — Experiments, Hypotheses, and Results

A consolidated, scannable log of every experiment run across this repo:
what was tested, what was expected, and what actually happened.
For narrative detail and reasoning, see [WHAT_I_LEARNED.md](WHAT_I_LEARNED.md).

---

## Module 01 — Single GPU Baseline

| Experiment | Hypothesis | Result |
|---|---|---|
| CPU vs MPS training speed | MPS should be faster | MPS ~2.5x faster (99K vs 40K tok/s), confirmed |
| MPS loss increasing over training | Suspected MPS hardware bug | Root cause: missing random seed, not hardware (see BUGS.md Bug 01) |
| Loss at initialization | Should be ≈ ln(65) ≈ 4.17 for 65-char vocab | Confirmed: 4.1935 at step 0 (seed=42) |

**Configuration**: n_embd=128, n_head=4, n_layer=4, block_size=128, batch_size=32, seed=42

---

## Module 02 — Multiprocess DDP

| Experiment | Hypothesis | Result |
|---|---|---|
| world_size=1 vs world_size=2 total throughput | ~2x speedup expected | ~1.2x actual (38,800 → 46,800 tok/s total) |
| Loss convergence, 1 vs 2 processes | Should be similar | Similar: final loss 2.582 (n=1) vs 2.6154 (n=2) over 200 steps |
| Per-process throughput, world_size=2 | Half of world_size=1's rate | ~23,400 tok/s/proc vs ~38,800 tok/s (single) — more than half lost to overhead |

**Configuration**: same model config as Module 01, gloo backend, CPU only, seed=42

**Why the gap from ideal 2x**: gradient AllReduce communication overhead +
CPU core contention (M2 Air has limited cores shared across processes)

---

## Module 03 — Mixed Precision (FP32 vs BF16)

| Experiment | Hypothesis | Result |
|---|---|---|
| BF16 vs FP32 speed on CPU | Expected speedup (2-3x, like MPS in Module 01) | **16x SLOWER** (2,450 vs 38,000 tok/s) |
| BF16 vs FP32 loss convergence | Expected some divergence due to reduced precision | **Identical to 4 decimals** (2.6126 both) |
| autocast op coverage | Expected some ops to fall back to FP32 | Confirmed: LayerNorm, softmax, cross-entropy stayed FP32 automatically |

**Configuration**: same model config, 200 steps, seed=42, `torch.autocast(device_type="cpu", dtype=torch.bfloat16)`

**Why BF16 was slower**: Apple M-series CPUs have no native BF16 compute
units — PyTorch emulates BF16 arithmetic on FP32 hardware, adding overhead
with no throughput benefit. Speedup requires dedicated low-precision compute
(Tensor Cores, TPUs, Trainium/Inferentia).

---

## Module 04 — Profiling

| Experiment | Hypothesis | Result |
|---|---|---|
| Biggest time consumer in a training step | Expected matmul-dominated | forward (52.6%) + backward (43.8%) ≈ 96%, as expected at the phase level |
| Within forward pass, what dominates? | Expected matmul/attention | `aten::bernoulli_` (dropout RNG) = 26.1%, nearly matching `aten::mm` (22.4%) |
| Effect of removing dropout (`dropout=0.0`) | Should reduce total time | **30.6% reduction** (2.366s → 1.642s for 20 steps) |
| Matmul absolute time, with vs without dropout | Should stay roughly constant | Confirmed: ~524ms either way — dropout removal didn't speed up matmul, it removed a separate cost |
| forward vs backward share, with vs without dropout | — | With dropout: forward > backward (52.6% vs 43.8%). Without: backward > forward (60.2% vs 35.1%) — dropout's cost was counted in forward |

**Configuration**: same model config, 10 warmup steps (unprofiled) + 20 profiled steps, seed=42

---

## Cross-module observations

| Observation | Modules involved |
|---|---|
| Reproducibility (fixed seeds) is a prerequisite for trusting any other comparison | 01, 02, 03, 04 |
| "Ideal" scaling/speedup numbers (2x for DDP, faster for BF16) require specific hardware support that general-purpose CPUs often lack | 02, 03 |
| Small model scale can make "auxiliary" operations (RNG, framework overhead) comparable in cost to "core" operations (matmul) — this likely inverts at production scale | 04 |
| Every quantitative claim in this repo is backed by a saved `results/*.json` or `results/*.txt` file from an actual run | 01-04 |