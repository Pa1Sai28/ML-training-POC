"""
04_profiling/train_profiled.py
===============================
Question this file answers:
  "Where does the time in a training step actually go?"

This extends 01_single_gpu_baseline/train.py with torch.profiler,
wrapping a handful of training steps to measure per-operation time
(matmuls, attention, layernorm, optimizer step, data loading) and
exporting both a summary table and a Chrome trace for visualization.

Run:
    python 04_profiling/train_profiled.py --steps 20
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
from torch.profiler import profile, record_function, ProfilerActivity

# =============================================================================
# BLOCK 1: DATA
# =============================================================================
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "01_single_gpu_baseline/data/shakespeare.txt"


def download_shakespeare():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading Tiny Shakespeare (~1MB)...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
        print(f"  Saved to {DATA_PATH}")
    return open(DATA_PATH, "r").read()


class CharDataset(Dataset):
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
def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

# =============================================================================
# BLOCK 4: TRAINING FUNCTION WITH PROFILING
# =============================================================================
# record_function(name) is a label — like a sticky note on a region of code.
# The profiler groups all operations inside that block under this name in
# its output, making it easy to answer "how much time went to forward pass
# vs backward pass vs data loading" without reading raw op-by-op traces.
#
# We only profile a SMALL number of steps (--profile_steps, default 10)
# after a few warmup steps. Profiling itself adds overhead, so profiling
# the entire training run would give misleading numbers — we want a
# representative sample of "steady state" steps.

def train(args):
    device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"  04 — Profiling")
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
        dropout     = args.dropout,
    ).to(device)
    print(f"  Model: {model.num_params():,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    model.train()
    data_iter = iter(loader)

    # ── Warmup steps (NOT profiled) ─────────────────────────────────────
    # The first few steps include one-time costs (JIT warmup, memory
    # allocation, page faults) that would skew profiling results.
    # We run these normally first, so the profiled steps reflect
    # steady-state behavior.
    print(f"\nRunning {args.warmup_steps} warmup steps (not profiled)...")
    for step in range(args.warmup_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        lr = get_lr(step, args.warmup, args.steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"Warmup done. Final warmup loss: {loss.item():.4f}")
    
# ── Profiled steps ───────────────────────────────────────────────────
    # profile() context manager records everything inside it.
    #
    # activities=[CPU]      — we're only profiling CPU ops (no CUDA on M2)
    # record_shapes=True    — captures tensor shapes per op (helps identify
    #                          which layer/operation a given cost belongs to)
    # profile_memory=True   — tracks memory allocation per op
    # with_stack=False      — Python stack traces add overhead; skip for now
    print(f"\nProfiling {args.profile_steps} steps...\n")

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:

        for step in range(args.profile_steps):

            # ── Data loading ─────────────────────────────────────────────
            with record_function("data_loading"):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    x, y = next(data_iter)

                x = x.to(device)
                y = y.to(device)

            # ── LR update ────────────────────────────────────────────────
            with record_function("lr_update"):
                lr = get_lr(args.warmup_steps + step, args.warmup, args.steps, args.lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            # ── Forward pass ─────────────────────────────────────────────
            with record_function("forward"):
                logits, loss = model(x, y)

            # ── Backward pass ────────────────────────────────────────────
            with record_function("zero_grad"):
                optimizer.zero_grad(set_to_none=True)

            with record_function("backward"):
                loss.backward()

            # ── Gradient clipping ────────────────────────────────────────
            with record_function("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ── Optimizer step ───────────────────────────────────────────
            with record_function("optimizer_step"):
                optimizer.step()

            # Mark step boundary for the profiler's step-based features
            prof.step()

    print(f"Profiling done. Final loss: {loss.item():.4f}")
    
    
# ── Print summary table ─────────────────────────────────────────────
    # key_averages() aggregates all recorded events across the profiled
    # steps, grouped by name (our record_function labels + internal
    # PyTorch op names). sort_by="cpu_time_total" ranks by total CPU
    # time consumed — the biggest contributors to step time appear first.
    print(f"\n{'='*60}")
    print(f"  Profiler Summary — Top 15 by CPU time")
    print(f"{'='*60}\n")

    summary_table = prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=15
    )
    print(summary_table)

    # ── Save outputs ─────────────────────────────────────────────────────
    os.makedirs("04_profiling/results", exist_ok=True)

    # Chrome trace — open in chrome://tracing or https://ui.perfetto.dev
    # for a visual timeline of every operation.
    suffix = f"_dropout{args.dropout}"
    trace_path = f"04_profiling/results/trace{suffix}.json"
    prof.export_chrome_trace(trace_path)

    summary_path = f"04_profiling/results/profile_summary{suffix}.txt"
    with open(summary_path, "w") as f:
        f.write(summary_table)

    print(f"\n{'='*60}")
    print(f"  Chrome trace saved → {trace_path}")
    print(f"  Summary saved      → {summary_path}")
    print(f"{'='*60}")

    return prof

# =============================================================================
# BLOCK 5: CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="04 — Profiling")

    # Model size (same defaults as previous modules)
    p.add_argument("--n_embd",     type=int,   default=128)
    p.add_argument("--n_head",     type=int,   default=4)
    p.add_argument("--n_layer",    type=int,   default=4)
    p.add_argument("--block_size", type=int,   default=128)
    p.add_argument("--dropout",    type=float, default=0.1)

    # Training
    p.add_argument("--steps",         type=int,   default=200,
                   help="Total steps, used for LR schedule continuity")
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--warmup",        type=int,   default=50)

    # Profiling
    p.add_argument("--warmup_steps",  type=int,   default=10,
                   help="Steps to run before profiling starts (not profiled)")
    p.add_argument("--profile_steps", type=int,   default=20,
                   help="Number of steps to profile")

    # Reproducibility
    p.add_argument("--seed",          type=int,   default=42)

    args = p.parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()