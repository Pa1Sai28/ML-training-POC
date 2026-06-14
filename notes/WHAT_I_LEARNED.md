# What I Learned — Running Notes

*Updated as I work through each module. Surprises, things that broke,things I thought I understood but didn't.*

---

## Module 01 — Single GPU Baseline

**Things I thought were obvious but weren't:**

`zero_grad()` placement confused me at first. I had it before the forward pass in my mental model but it doesn't matter as long as it's before `backward()``set_to_none=True` is a small optimization — it frees the gradient memory entirely rather than filling it with zeros. Worth knowing the difference.

**The loss starting value:**
Loss starts at ~4.17 for 65 characters. This is `ln(65)` — the cross-entropy of a uniform distribution over 65 classes. The model starts by assigning equal probability (~1/65) to every next character. As it learns, it gets better than random. This is a useful sanity check: if your loss starts *lower* than `ln(vocab_size)`, something is wrong with your data (leakage). If it starts *higher*, your initialization is off.

**Weight tying:**
I understood this conceptually but implementing it made it concrete. It's literally one line: `self.tok_emb.weight = self.head.weight`. After that, both `nn.Module` references point to the same underlying tensor. Gradients from both the embedding lookup and the final projection update the same weights.

**MPS on M2:**
Using `--device mps` is ~3-4× faster than CPU on this model. The limitation I hit: `num_workers > 0` in DataLoader throws a RuntimeError with MPS. MPS tensors live in a process-local memory space that worker processes can't access. Fix: `num_workers=0`. This is documented in PyTorch's MPS release notes but easy to miss.

**What "training" actually means — the moment it clicked:**
The loss going from 4.17 → 2.1 over 500 steps *is* the model learning. Every step, the weights shift slightly to make the correct next character slightly more probable. The generated text going from noise to semi-coherent Shakespeare is the direct visible result. This sounds obvious written out but seeing it happen in your own code is different.

---

## Module 02 — Coming next

Questions I'm going into it with:
- What does `dist.init_process_group()` actually set up?
- How does `DistributedSampler` know which indices to give to which rank?
- What happens at the hardware level during `all_reduce`?
- If both processes start with the same random seed, are the models always in sync?

---

---
## Module 02 — Multiprocess DDP

**Going in, I asked myself these questions — here's what I found:**

**"What does `dist.init_process_group()` actually set up?"**
It's a rendezvous step. Every process dials into the same MASTER_ADDR/PORT, and once all `world_size` processes have checked in, the library builds direct communication channels between every pair of processes. After this call returns, processes can send/receive tensors to each other (via collective operations like AllReduce) without any of this setup code running again.

**"How does `DistributedSampler` know which indices to give to which rank?"**
It's pure arithmetic, no communication needed. Given `num_replicas` (world_size) and `rank`, it deterministically computes "every Nth index starting at rank" (after shuffling with a shared seed). Rank 0 and rank 1 never talk to each other to divide up the data — they each independently compute their own non-overlapping slice from the same formula and the same seed.

**"What happens at the hardware level during `all_reduce`?"**
On CPU with the `gloo` backend: each process's gradient tensors get sent over local sockets to the other process(es), summed (or averaged), and the result sent back — so every process ends up holding the same averaged gradient tensor. On GPU/Trainium with NCCL or Neuron's collectives, this happens over much faster physical interconnects (NVLink, chip-to-chip), which is why interconnect speed directly determines how close you get to "ideal" Nx scaling.

**"If both processes start with the same random seed, are the models always in sync?"**
Yes — but it's not automatic from the seed alone. The seed has to be set *before* model construction on every process, so every process builds
identical initial weights. DDP then also broadcasts rank 0's weights to everyone at wrap time as a safety net. From there, identical weights +
identical (averaged) gradients on every step = the optimizer step produces identical weights everywhere, forever. No further syncing needed.

**The throughput surprise:**
I expected world_size=2 to roughly double throughput vs world_size=1. Instead: 1 process → ~38,800 tok/s total. 2 processes → ~46,800 tok/s total (~1.2x). The AllReduce communication overhead and CPU core contention on the M2 Air eat most of the theoretical 2x gain. This is a real lesson in why distributed training efficiency depends heavily on interconnect speed — the same reason AWS Trainium chips have dedicated chip-to-chip interconnect rather than relying on general networking.

**Logging discipline:**
Every `print()` and metrics-save needed an `if rank == 0:` guard. Forgetting this means N processes all print the same thing — easy to miss in a 2-process test, immediately obvious (and annoying) at higher world_size.

---
## Module 03 — Coming next

Questions I'm going into it with:
- What's the actual numerical difference between FP32 and BF16 — what gets
  truncated?
- Does BF16 affect the *loss values themselves*, or just memory/speed?
- Are there operations that MUST stay in FP32 even during "mixed precision"
  training, and why?
- Will I see the same 2-2.5x speedup pattern I saw with MPS in Module 01?

---
## Module 03 — Mixed Precision Benchmarking

**Going in, I asked myself these questions — here's what I found:**

**"What's the actual numerical difference between FP32 and BF16 — what gets
truncated?"**
BF16 keeps the same 8-bit exponent as FP32 (same dynamic range, won't overflow/underflow differently than FP32) but cuts the mantissa from 23 bits to 7 bits. So BF16 numbers can represent the same *range* of magnitudes as FP32, just with far less precision within that range — like rounding 3.14159265 to 3.14.

**"Does BF16 affect the loss values themselves, or just memory/speed?"**
In this run: final loss was identical to 4 decimal places (2.6126 both ways), and the per-step loss curve matched almost exactly. autocast keeps the loss computation itself in FP32 (cross-entropy is precision-sensitive), and the master weights stay FP32 throughout — so convergence was unaffected here.

**"Are there operations that MUST stay in FP32 even during mixed precision,
and why?"**
Yes — confirmed by autocast's own behavior. Things like LayerNorm, softmax, and the loss computation are kept in FP32 automatically by autocast, because they involve operations (small differences, exponentials, sums of many values) where BF16's reduced precision causes real numerical instability. Matrix multiplies (the bulk of the compute) are the main thing that gets cast to BF16.

**"Will I see the same 2-2.5x speedup pattern I saw with MPS in Module 01?"**
No — the opposite. BF16 was ~16x SLOWER than FP32 on CPU (2,450 tok/s vs 38,000 tok/s), while loss curves were nearly identical.

**The real finding — why mixed precision didn't help here:**
Mixed precision speedups come from *dedicated low-precision hardware units* (NVIDIA Tensor Cores, TPUs, AWS Trainium/Inferentia). Apple M-series CPUs have no native BF16 compute path — PyTorch has to emulate BF16 arithmetic on top of FP32 hardware, which is slower than just running FP32 directly. The dtype conversion happened correctly and numerically the result was sound — there's just no hardware speed benefit to claim on this device.

**Why this matters for AWS Neuron specifically:**
This is the inverse of what I'd expect on Trainium, which has dedicated BF16/FP16 compute units designed for exactly this kind of workload. The
takeaway: mixed precision isn't a free speedup you get from "smaller numbers" — it's a hardware-dependent optimization that only pays off when
the accelerator has a fast path for the lower-precision dtype. On Trainium, I'd expect to see the speedup that I *didn't* see here.

---
## Module 04 — Coming next

Questions I'm going into it with:
- What does a PyTorch profiler trace actually show — time per operation,
  or something else?
- Where does the time actually go in a training step — is it the matmuls,
  the data loading, or something else entirely?
- Will profiling reveal anything surprising given what we already found
  about BF16 on CPU in Module 03?
- How do profiler outputs translate to identifying bottlenecks on real
  accelerators like Trainium?