"""
03_mixed_precision/train_bf16.py
=================================
Question this file answers:
  "Does training in BF16 change speed, memory, or convergence
  compared to FP32 — and what actually has to stay in FP32?"

This extends 01_single_gpu_baseline/train.py with PyTorch's
autocast mixed-precision context manager. The model architecture,
data, and optimizer are unchanged — only the forward pass dtype
changes during the `bf16` run.

Run:
    python 03_mixed_precision/train_bf16.py --precision fp32 --steps 200
    python 03_mixed_precision/train_bf16.py --precision bf16 --steps 200
"""

import os
import math
import time
import json
import argparse
import urllib.request
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# BLOCK 1: DATA
# =============================================================================
# Identical to Modules 01/02 — same dataset, same tokenization.
# Mixed precision doesn't change how data is loaded, only how the
# forward pass computes once the data hits the model.

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
    """Identical to Modules 01/02."""

    def __init__(self, text: str, block_size: int):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.data = torch.tensor(
            [self.char_to_idx[c] for c in text], dtype=torch.long
        )
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y
    
# =============================================================================
# BLOCK 2: MODEL
# =============================================================================
# Identical to Modules 01/02. Mixed precision is applied via autocast
# around the forward pass — the model definition itself never changes.
# This is an important property of autocast: it's non-invasive.

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.qkv   = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj  = nn.Linear(n_embd, n_embd, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
                  .view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)

        def reshape(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.drop(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
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
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.ff   = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_head: int,
                 n_layer: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop   = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.head   = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss
    
# =============================================================================
# BLOCK 3: LEARNING RATE SCHEDULE
# =============================================================================
# Identical to Modules 01/02 — warmup + cosine decay.
# Precision doesn't affect this — it's pure Python math, no tensors involved.

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

# =============================================================================
# BLOCK 4: TRAINING FUNCTION
# =============================================================================
# This is nearly identical to Module 01's train() function. The ONLY
# difference is the forward pass — wrapped (or not) in torch.autocast.
#
# autocast is a context manager: operations INSIDE the `with` block that
# support BF16 run in BF16 automatically. Operations that don't support
# BF16 (or that PyTorch knows need full precision, like some reductions)
# silently stay in FP32. You don't have to manually cast each tensor —
# autocast handles the dtype decisions per-operation.
#
# Importantly: model WEIGHTS stay in FP32 the whole time. Only the
# intermediate activations/computations inside the `with` block are
# computed in BF16. This is "mixed" precision — not "everything BF16".

def train(args):
    device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"  03 — Mixed Precision Benchmark")
    print(f"  Precision: {args.precision}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    text = download_shakespeare()
    dataset = CharDataset(text, block_size=args.block_size)
    print(f"  Dataset: {len(text):,} chars | vocab: {dataset.vocab_size} | "
          f"sequences: {len(dataset):,}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ── Model ────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = TinyGPT(
        vocab_size  = dataset.vocab_size,
        n_embd      = args.n_embd,
        n_head      = args.n_head,
        n_layer     = args.n_layer,
        block_size  = args.block_size,
        dropout     = 0.1,
    ).to(device)
    print(f"  Model: {model.num_params():,} parameters")

    # ── Optimizer ────────────────────────────────────────────────────────
    # The optimizer ALWAYS operates on FP32 weights, regardless of
    # --precision. This is the "master copy" discussed earlier.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # ── Precision setup ──────────────────────────────────────────────────
    # autocast_enabled controls whether we wrap the forward pass.
    # For fp32, we use a "null context" (nullcontext) — i.e. no-op,
    # so the code path is identical either way except for this one thing.
    use_bf16 = (args.precision == "bf16")
    
# ── Training loop ────────────────────────────────────────────────────
    metrics = {
        "precision": args.precision,
        "model_params": model.num_params(),
        "steps": [], "loss": [], "lr": [],
        "step_time_ms": [], "tokens_per_sec": [],
    }

    model.train()
    data_iter = iter(loader)
    t_start = time.time()
    print(f"\nTraining for {args.steps} steps in {args.precision}...\n")

    for step in range(args.steps):
        # ── Get a batch ──────────────────────────────────────────────────
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ── Update learning rate ────────────────────────────────────────
        lr = get_lr(step, args.warmup, args.steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Forward pass ─────────────────────────────────────────────────
        # THIS is the one line that differs between fp32 and bf16 runs.
        #
        # bf16:  torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        #        — matmuls, linear layers, conv inside this block run in
        #          bf16. LayerNorm, softmax, and the final cross_entropy
        #          loss computation stay in fp32 automatically — autocast
        #          knows these need full precision for numerical stability.
        #
        # fp32:  nullcontext() — does nothing, forward pass runs exactly
        #        as it did in Module 01.
        t0 = time.perf_counter()

        ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16) if use_bf16 else nullcontext()
        with ctx:
            logits, loss = model(x, y)

        # ── Backward pass ────────────────────────────────────────────────
        # Gradients are computed and stored in FP32 regardless of the
        # forward pass precision — autocast only affects the forward
        # computation graph's intermediate dtypes, not .grad tensors.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # ── Gradient clipping ────────────────────────────────────────────
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ── Optimizer step ────────────────────────────────────────────────
        # Operates on FP32 master weights — identical either way.
        optimizer.step()

        step_ms = (time.perf_counter() - t0) * 1000
        tokens_per_sec = (args.batch_size * args.block_size) / (step_ms / 1000)

        # ── Logging ───────────────────────────────────────────────────────
        if step % args.log_every == 0:
            loss_val = loss.item()
            print(f"  step {step:4d} | loss {loss_val:.4f} | "
                  f"lr {lr:.2e} | {step_ms:.1f}ms | {tokens_per_sec:,.0f} tok/s")
            metrics["steps"].append(step)
            metrics["loss"].append(round(loss_val, 4))
            metrics["lr"].append(round(lr, 6))
            metrics["step_time_ms"].append(round(step_ms, 1))
            metrics["tokens_per_sec"].append(round(tokens_per_sec, 1))
            
# ── Final stats ──────────────────────────────────────────────────────
    total_time = time.time() - t_start
    metrics["total_time_s"] = round(total_time, 1)
    metrics["final_loss"] = round(loss.item(), 4)

    os.makedirs("03_mixed_precision/results", exist_ok=True)
    out_path = f"03_mixed_precision/results/metrics_{args.precision}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done in {total_time:.1f}s")
    print(f"  Final loss: {metrics['final_loss']}")
    print(f"  Metrics saved → {out_path}")
    print(f"{'='*60}")

    return metrics

# =============================================================================
# BLOCK 5: CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="03 — Mixed Precision Benchmark")

    # Model size (same defaults as Modules 01/02)
    p.add_argument("--n_embd",     type=int,   default=128)
    p.add_argument("--n_head",     type=int,   default=4)
    p.add_argument("--n_layer",    type=int,   default=4)
    p.add_argument("--block_size", type=int,   default=128)

    # Training
    p.add_argument("--steps",      type=int,   default=200)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--warmup",     type=int,   default=50)
    p.add_argument("--log_every",  type=int,   default=50)

    # Precision
    p.add_argument("--precision",  type=str,   default="fp32",
                   choices=["fp32", "bf16"],
                   help="fp32=baseline, bf16=autocast mixed precision")

    # Reproducibility
    p.add_argument("--seed",       type=int,   default=42)

    args = p.parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()