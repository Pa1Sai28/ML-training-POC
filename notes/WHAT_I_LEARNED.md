# What I Learned — Running Notes

*Updated as I work through each module. Surprises, things that broke,
things I thought I understood but didn't.*

---

## Module 01 — Single GPU Baseline

**Things I thought were obvious but weren't:**

`zero_grad()` placement confused me at first. I had it before the forward pass
in my mental model but it doesn't matter as long as it's before `backward()`.
`set_to_none=True` is a small optimization — it frees the gradient memory
entirely rather than filling it with zeros. Worth knowing the difference.

**The loss starting value:**
Loss starts at ~4.17 for 65 characters. This is `ln(65)` — the cross-entropy
of a uniform distribution over 65 classes. The model starts by assigning
equal probability (~1/65) to every next character. As it learns, it gets better
than random. This is a useful sanity check: if your loss starts *lower* than
`ln(vocab_size)`, something is wrong with your data (leakage). If it starts
*higher*, your initialization is off.

**Weight tying:**
I understood this conceptually but implementing it made it concrete. It's
literally one line: `self.tok_emb.weight = self.head.weight`. After that, both
`nn.Module` references point to the same underlying tensor. Gradients from
both the embedding lookup and the final projection update the same weights.

**MPS on M2:**
Using `--device mps` is ~3-4× faster than CPU on this model. The limitation
I hit: `num_workers > 0` in DataLoader throws a RuntimeError with MPS.
MPS tensors live in a process-local memory space that worker processes
can't access. Fix: `num_workers=0`. This is documented in PyTorch's MPS
release notes but easy to miss.

**What "training" actually means — the moment it clicked:**
The loss going from 4.17 → 2.1 over 500 steps *is* the model learning.
Every step, the weights shift slightly to make the correct next character
slightly more probable. The generated text going from noise to semi-coherent
Shakespeare is the direct visible result. This sounds obvious written out but
seeing it happen in your own code is different.

---

## Module 02 — Coming next

Questions I'm going into it with:
- What does `dist.init_process_group()` actually set up?
- How does `DistributedSampler` know which indices to give to which rank?
- What happens at the hardware level during `all_reduce`?
- If both processes start with the same random seed, are the models always in sync?

---
