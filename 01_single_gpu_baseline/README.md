# 01 — Single GPU Baseline
### "What does one training step actually do?"

This is the foundation. Before any distributed complexity, you need to own
the single-process training loop — every line, every decision.

Everything in large-scale LLM training (DDP, FSDP, mixed precision, Trainium)
is this same loop, split across machines or optimized for hardware.
If this loop is fuzzy, everything built on top of it will be too.

---

## What We're Training

A character-level GPT on Tiny Shakespeare (~1MB of text).

**Why character-level?**
Real LLMs use subword tokens (GPT-2 has 50,257). Characters keep vocab_size=65
so the model trains in ~60 seconds on a MacBook. The architecture — embedding,
attention, residuals, LayerNorm — is identical to production models.
Only the numbers are smaller.

**Why Shakespeare?**
It's a real human language with structure. After 500 steps you can see whether
the model is learning (output starts looking like noisy Shakespeare) vs failing
(random gibberish). That visual feedback loop is important when you're learning.

---

## The Architecture

```
Input: sequence of character IDs  [B, T]
  ↓
Token Embedding + Position Embedding  →  [B, T, n_embd]
  ↓
× 4 Transformer Blocks:
    LayerNorm → CausalSelfAttention → residual add
    LayerNorm → FeedForward         → residual add
  ↓
Final LayerNorm
  ↓
Linear projection → logits  [B, T, vocab_size]
  ↓
Cross-entropy loss vs targets (input shifted by 1)
```

Default config: ~400K parameters. Tiny — but real.

---

## One Training Step (the loop you need to understand cold)

```python
x, y = next(data_iter)          # 1. get a batch of (input, target) pairs
logits, loss = model(x, y)      # 2. forward pass → compute predictions + loss
optimizer.zero_grad()           # 3. clear gradients from previous step
loss.backward()                 # 4. backprop: compute gradient for every parameter
clip_grad_norm_(model, 1.0)     # 5. clip gradient norm to prevent explosions
optimizer.step()                # 6. update every parameter using its gradient
```

Six lines. Every LLM training job on every piece of hardware is this,
repeated millions of times.

---

## What Each Piece Does (and Why It's There)

**Token + Position Embeddings (added, not concatenated)**
The model gets two signals per token: what it IS (token embedding) and
WHERE it is (position embedding). They're summed into a single vector.
The model learns to separate and use both signals through attention.

**Causal Mask**
A lower-triangular matrix of 1s. Position 5 can see positions 0–5, blocked
from 6, 7, 8... This is what makes it *auto-regressive* — it can only look
backwards. Without this mask you'd be "cheating" — the model could see the
answer (the next token) while predicting it.

**Residual Connections (`x = x + attention(x)`)**
Every block adds its output to its input rather than replacing it.
This creates a "gradient highway" straight back to early layers —
gradients don't have to pass through every weight matrix to reach
the first layer. This is why transformers can be 100+ layers deep.

**Weight Tying**
The token embedding matrix and the output projection share the same weights.
The embedding maps token→vector; the output maps vector→token probability.
Making them the same tensor forces consistency: the "meaning" of a token
in input must match its "meaning" in output. Also saves parameters.

**AdamW (not vanilla Adam)**
AdamW applies weight decay directly to the weights, not via the gradient.
This matters because Adam adapts the effective learning rate per parameter —
applying decay through the gradient interacts with that adaptation in a broken
way. "Decoupled weight decay" fixes it. Every modern LLM uses AdamW.

**Gradient Clipping (max_norm=1.0)**
Measures the total magnitude of all gradients combined. If it exceeds 1.0,
scales all gradients down proportionally. Does NOT zero them — just shrinks
a step that would be dangerously large. Critical under reduced precision (BF16)
where numerical noise can amplify gradient spikes.

---

## M2 Mac: CPU vs MPS

This script runs on both. The difference:

| Backend | What uses it | Typical speedup |
|---------|-------------|-----------------|
| `cpu`   | All CPU cores | baseline |
| `mps`   | M2 GPU (Metal) | ~3–5× for matmuls |

```bash
# Let it auto-detect (uses MPS if available)
python 01_single_gpu_baseline/train.py

# Force CPU (useful for comparison)
python 01_single_gpu_baseline/train.py --device cpu

# Force MPS explicitly
python 01_single_gpu_baseline/train.py --device mps
```

**M2 quirk worth knowing:**
`num_workers > 0` in DataLoader breaks with MPS — MPS tensors can't cross
process boundaries. This script sets `num_workers=0`. This becomes relevant
in Module 04 where we profile the data loading bottleneck.

---

## Running It

```bash
# Install (only torch needed for this module)
pip install torch

# Run with defaults (~60-90s on M2 CPU, ~20-30s on MPS)
python 01_single_gpu_baseline/train.py

# Run longer for better convergence
python 01_single_gpu_baseline/train.py --steps 2000

# Bigger model (still runs on M2)
python 01_single_gpu_baseline/train.py --n_embd 256 --n_layer 6 --steps 1000
```

---

## What "Good" Looks Like

At random initialization, loss ≈ `ln(vocab_size)` = `ln(65)` ≈ **4.17**
(the model assigns equal probability to all 65 characters — pure noise).

After 500 steps you should see loss drop to roughly **2.0–2.5**.
The generated text should look like garbled Shakespeare — real words starting
to appear, wrong but structured.

After 2000+ steps: loss ~1.5–1.8, generated text has recognizable phrases.

If loss stays at 4.17 or increases: something is broken (check device, check
that data loaded correctly, check the loss isn't returning NaN).

---

## What's in results/

After training:
- `results/metrics.json` — loss, LR, step time, tokens/sec at every log step

These numbers feed directly into Module 03 where we compare against BF16.

---

## Questions This Module Answers

Before moving to Module 02, you should be able to answer:

1. Why is the target `y` just `x` shifted by one position?
2. What does `loss.backward()` actually compute?
3. Why does `zero_grad()` come AFTER the forward pass, not before?
4. What breaks if you remove gradient clipping and get a large gradient?
5. Why do we add token and position embeddings instead of concatenating them?
6. What does a loss of 4.17 mean at the start, and why that specific number?

If any are fuzzy — the answers are all in the comments in `train.py`.

---

## Next: Module 02

[02 — Multiprocess DDP →](../02_multiprocess_ddp/README.md)

"What changes when you add a second process?"
We take this exact loop and split it across 2 processes on the same M2.
No new concepts in the model — everything new is in how the processes coordinate.
