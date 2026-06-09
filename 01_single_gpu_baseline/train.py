"""
01_single_gpu_baseline/train.py
================================
Question this file answers:
  "What does one training step actually do — at every line?"

This trains a tiny GPT-style language model on character-level Shakespeare.
No GPU required. Runs on M2 Mac using the CPU (or MPS if available).
Target runtime: ~60-90 seconds for 500 steps.

Before adding ANY distributed complexity (DDP, FSDP, mixed precision),
you need to own this loop. Everything in LLM training is this loop,
scaled up and split across machines.

Run:
    python 01_single_gpu_baseline/train.py
    python 01_single_gpu_baseline/train.py --steps 1000 --device mps
"""

import os
import math
import time
import json
import argparse
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# BLOCK 1: DATA
# =============================================================================
# We use the "Tiny Shakespeare" dataset — 1MB of Shakespeare plays as plain text.
# It's the standard toy dataset for language model tutorials because:
#   - Small enough to load in memory instantly
#   - Real human language (not random tokens)
#   - Easy to see if the model is learning (output starts looking like Shakespeare)
#
# Character-level means each token is a single character: 'H', 'e', 'l', 'l', 'o'
# Real LLMs use subword tokens (GPT-2 has 50,257 of them) but characters keep
# vocab_size tiny (65 unique chars) so the model trains in seconds, not hours.
# The concepts — embedding, attention, loss — are identical.

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "01_single_gpu_baseline/data/shakespeare.txt"


def download_shakespeare():
    """Download Tiny Shakespeare if not already present."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading Tiny Shakespeare (~1MB)...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
        print(f"  Saved to {DATA_PATH}")
    return open(DATA_PATH, "r").read()


class CharDataset(Dataset):
    """
    Turns raw text into (input_sequence, target_sequence) pairs.

    The key insight: for a language model, the target is just the input
    shifted by one position. Given "Hello", the model learns:
      - After 'H'         → predict 'e'
      - After 'He'        → predict 'l'
      - After 'Hel'       → predict 'l'
      - After 'Hell'      → predict 'o'

    This is called "next token prediction" — the training objective for
    every LLM including GPT-4, LLaMA, and Claude.
    """

    def __init__(self, text: str, block_size: int):
        # Build vocabulary: every unique character in the text
        chars = sorted(set(text))
        self.vocab_size = len(chars)

        # Two lookup tables: char→int and int→char
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}

        # Encode entire text as a sequence of integers
        self.data = torch.tensor(
            [self.char_to_idx[c] for c in text], dtype=torch.long
        )
        self.block_size = block_size
        print(f"  Dataset: {len(text):,} chars | vocab: {self.vocab_size} | "
              f"sequences: {len(self):,}")

    def __len__(self):
        # Every starting position except the last block_size chars
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x: block_size tokens starting at idx
        # y: the same window shifted right by 1 — these are the targets
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# =============================================================================
# BLOCK 2: MODEL
# =============================================================================
# A real GPT architecture at tiny scale.
# The design decisions here are the same ones used in GPT-2, LLaMA, and Mistral.
# Only the numbers are different.

class CausalSelfAttention(nn.Module):
    """
    Self-attention with causal (left-to-right) masking.

    "Self-attention" means every token looks at every other token to decide
    how much to borrow from each one. The "causal" part means token 5 can
    only look at tokens 0-5, not 6, 7, 8... This is what makes it a language
    MODEL (predicts what comes next) rather than an encoder (reads the whole
    sequence bidirectionally like BERT).

    Q, K, V — Query, Key, Value:
      Think of it like a search engine built into each layer.
      Q = "what am I looking for?"
      K = "what do I have to offer?"
      V = "what I'll actually contribute if selected"
      Attention score = how well Q matches K → weights the V contribution.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        # One linear layer produces Q, K, V all at once (3× the embedding size)
        # More efficient than three separate projections
        self.qkv   = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj  = nn.Linear(n_embd, n_embd, bias=False)
        self.drop  = nn.Dropout(dropout)

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Causal mask: lower-triangular matrix of 1s
        # Position i can attend to positions 0..i, blocked from i+1..T
        # register_buffer = not a trainable parameter, but moves to device with model
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
                  .view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape   # Batch, Time (sequence length), Channels (n_embd)

        # Split the combined QKV projection into three separate tensors
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)

        # Reshape for multi-head: [B, T, C] → [B, n_head, T, head_dim]
        def reshape(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        # Scale by sqrt(head_dim) to keep variance stable regardless of head size
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale  # [B, n_head, T, T]

        # Apply causal mask: future positions get -inf → softmax makes them ~0
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.drop(att)

        # Weighted sum of values
        out = att @ v                                    # [B, n_head, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # → [B, T, C]
        return self.proj(out)


class FeedForward(nn.Module):
    """
    Two linear layers with GELU activation between them.
    Expands to 4× the embedding dimension then contracts back.
    This is where most of the "memory" of a transformer lives —
    attention routes information, FFN transforms it.
    """
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer layer = LayerNorm + Attention + LayerNorm + FFN.
    The residual connections (x = x + ...) are critical:
    they let gradients flow directly back to early layers,
    which is why transformers can be trained 100+ layers deep.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.ff   = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual: attention reads normalized x
        x = x + self.ff(self.ln2(x))    # residual: FFN transforms normalized x
        return x


class TinyGPT(nn.Module):
    """
    A complete GPT language model.
    This architecture is identical in structure to GPT-2 —
    only n_embd, n_head, n_layer are smaller.
    """

    def __init__(self, vocab_size: int, n_embd: int, n_head: int,
                 n_layer: int, block_size: int, dropout: float = 0.1):
        super().__init__()

        # Token embedding: integer token ID → dense vector
        # e.g. character 'H' (id=27) → 128-dim vector of learned floats
        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        # Position embedding: position 0,1,2... → dense vector
        # The model needs to know WHERE in the sequence each token is
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.drop   = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f   = nn.LayerNorm(n_embd)      # final layer norm before output
        self.head   = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: the output projection reuses the token embedding weights.
        # Intuition: the embedding that maps token→vector and the projection that
        # maps vector→token-probability should be consistent with each other.
        # Bonus: saves vocab_size × n_embd parameters (65 × 128 = 8,320 here).
        # GPT-2, LLaMA, and most modern LLMs use this.
        self.tok_emb.weight = self.head.weight

        # Initialize weights: small normal distribution
        # Too-large initialization causes gradient explosion from step 1
        self.apply(self._init_weights)

        print(f"  Model: {self.num_params():,} parameters")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Both embeddings are [B, T, n_embd] — add them elementwise
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Project to vocabulary size → logits (raw scores for each possible next char)
        logits = self.head(x)   # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Cross-entropy loss: flatten to [B*T, vocab_size] vs [B*T]
            # This is "how wrong was the model about every next character?"
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8):
        """
        Autoregressively generate new tokens after a seed sequence.
        temperature > 1 = more random, temperature < 1 = more conservative.
        Used to visualize what the model has learned — does output look like Shakespeare?
        """
        for _ in range(max_new_tokens):
            # Crop to block_size if sequence gets too long
            idx_cond = idx[:, -self.pos_emb.num_embeddings:]
            logits, _ = self(idx_cond)
            # Take the last time step's logits only
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# =============================================================================
# BLOCK 3: LEARNING RATE SCHEDULE
# =============================================================================
# Why not just use a fixed learning rate?
#
# Problem 1 — cold start: At step 0, weights are random noise, gradient estimates
# are wildly inaccurate, optimizer momentum is uninitialized. A full-size learning
# rate on a randomly-initialized model causes massive early instability.
# Warmup ramps from 0 → peak_lr over the first N steps.
#
# Problem 2 — end of training: Near convergence, you're close to a good solution.
# Large steps overshoot it. Cosine decay smoothly reduces LR to near-zero by the
# final step, letting the model settle precisely into the minimum it has found.

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    # Phase 1: linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Phase 2: cosine decay from max_lr → ~0
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# BLOCK 4: TRAINING LOOP
# =============================================================================

def train(args):
    # ── Device selection ──────────────────────────────────────────────────────
    # M2 Mac supports three backends:
    #   "cpu"  — always works, slowest
    #   "mps"  — Apple Metal Performance Shaders, uses M2 GPU cores
    #            ~3-5× faster than CPU for matrix operations
    #            Note: some ops not yet supported in MPS, fallback to CPU if needed
    #   "cuda" — NVIDIA only, not available on M2
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  01 — Single GPU Baseline")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    text = download_shakespeare()
    dataset = CharDataset(text, block_size=args.block_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers > 0 is faster on CPU but causes issues with MPS
        # This is a real M2 quirk — MPS tensors can't cross process boundaries
        num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = TinyGPT(
        vocab_size  = dataset.vocab_size,
        n_embd      = args.n_embd,
        n_head      = args.n_head,
        n_layer     = args.n_layer,
        block_size  = args.block_size,
        dropout     = 0.1,
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # AdamW = Adam with decoupled weight decay
    # Weight decay = L2 regularization on weights (but NOT on biases or norms)
    # "decoupled" means the decay is applied directly to weights, not via the gradient
    # This is what every modern LLM uses: GPT-3, LLaMA, Mistral, all use AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),   # standard for LLM training (slightly higher beta2 than default)
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    metrics = {
        "device": str(device),
        "model_params": model.num_params(),
        "steps": [], "loss": [], "lr": [],
        "step_time_ms": [], "tokens_per_sec": [],
    }

    model.train()
    data_iter  = iter(loader)
    t_start    = time.time()

    print(f"\nTraining for {args.steps} steps...\n")

    for step in range(args.steps):

        # ── Get a batch ───────────────────────────────────────────────────────
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        # Move tensors to device (CPU/MPS)
        # non_blocking=True allows the transfer to overlap with CPU work
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ── Update learning rate ──────────────────────────────────────────────
        # We set LR manually on every step rather than using a scheduler.
        # Why? Full visibility — you can print the exact LR at any step.
        # Schedulers abstract this away, which is fine for production but
        # makes debugging harder when you're learning.
        lr = get_lr(step, args.warmup, args.steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Forward pass ──────────────────────────────────────────────────────
        # The model takes input tokens x and the targets y,
        # returns logits (predictions) and loss (how wrong it was)
        t0 = time.perf_counter()
        logits, loss = model(x, y)

        # ── Backward pass ─────────────────────────────────────────────────────
        # zero_grad clears gradients from the previous step
        # set_to_none=True is faster than .zero_() — sets grad to None instead
        # of filling with zeros, saving a memory write
        optimizer.zero_grad(set_to_none=True)

        # loss.backward() runs the chain rule through every layer
        # After this call, every parameter has a .grad tensor
        loss.backward()

        # ── Gradient clipping ─────────────────────────────────────────────────
        # Measures the global norm of ALL gradients combined.
        # If total norm > max_norm (1.0), scales every gradient down proportionally.
        # Does NOT zero gradients — just shrinks them if the step would be too large.
        # This is critical for training stability: one bad batch can't destroy training.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ── Optimizer step ────────────────────────────────────────────────────
        # Updates every parameter using its .grad tensor
        # AdamW also applies weight decay here
        optimizer.step()

        step_ms = (time.perf_counter() - t0) * 1000
        tokens_per_sec = (args.batch_size * args.block_size) / (step_ms / 1000)

        # ── Logging ───────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            loss_val = loss.item()  # .item() pulls scalar from GPU/MPS tensor to Python float
            print(f"  step {step:4d} | loss {loss_val:.4f} | "
                  f"lr {lr:.2e} | {step_ms:.1f}ms | {tokens_per_sec:,.0f} tok/s")
            metrics["steps"].append(step)
            metrics["loss"].append(round(loss_val, 4))
            metrics["lr"].append(round(lr, 6))
            metrics["step_time_ms"].append(round(step_ms, 1))
            metrics["tokens_per_sec"].append(round(tokens_per_sec, 1))

    # ── Final stats ───────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    metrics["total_time_s"] = round(total_time, 1)
    metrics["final_loss"] = round(loss.item(), 4)

    os.makedirs("01_single_gpu_baseline/results", exist_ok=True)
    with open("01_single_gpu_baseline/results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done in {total_time:.1f}s")
    print(f"  Final loss: {metrics['final_loss']}")
    print(f"  Metrics saved → 01_single_gpu_baseline/results/metrics.json")
    print(f"{'='*60}")

    # ── Generate a sample ─────────────────────────────────────────────────────
    # This is the "did it learn anything?" check.
    # Seed with a newline character, generate 300 chars, see if it looks like text.
    print("\n--- Sample generation (300 chars) ---\n")
    model.eval()
    seed_char  = "\n"
    seed_idx   = dataset.char_to_idx[seed_char]
    context    = torch.tensor([[seed_idx]], dtype=torch.long, device=device)
    generated  = model.generate(context, max_new_tokens=300)
    output     = "".join([dataset.idx_to_char[i.item()] for i in generated[0]])
    print(output)
    print("\n" + "-"*40)

    return metrics


# =============================================================================
# BLOCK 5: CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="01 — Single GPU Baseline Training")

    # Model size
    p.add_argument("--n_embd",     type=int,   default=128,  help="Embedding dimension")
    p.add_argument("--n_head",     type=int,   default=4,    help="Number of attention heads")
    p.add_argument("--n_layer",    type=int,   default=4,    help="Number of transformer layers")
    p.add_argument("--block_size", type=int,   default=128,  help="Context window (tokens)")

    # Training
    p.add_argument("--steps",      type=int,   default=500,  help="Training steps")
    p.add_argument("--batch_size", type=int,   default=32,   help="Batch size")
    p.add_argument("--lr",         type=float, default=3e-4, help="Peak learning rate")
    p.add_argument("--warmup",     type=int,   default=50,   help="LR warmup steps")
    p.add_argument("--log_every",  type=int,   default=50,   help="Log every N steps")

    # Device
    p.add_argument("--device",     type=str,   default="auto",
                   choices=["auto", "cpu", "mps"],
                   help="auto=use MPS if available, cpu=force CPU")

    # Reproducibility — added after Bug 01 (see BUGS.md)
    p.add_argument("--seed",       type=int,   default=42,   help="Random seed for reproducibility")

    args = p.parse_args()

    # Set seed before anything else
    torch.manual_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()

