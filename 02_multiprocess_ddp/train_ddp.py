"""
02_multiprocess_ddp/train_ddp.py
=================================
Question this file answers:
  "What changes when training is split across multiple processes?"

This extends 01_single_gpu_baseline/train.py with PyTorch's
DistributedDataParallel (DDP). Instead of one process training on the full dataset, N processes each train on a slice of the
dataset, and gradients are synchronized (averaged) after every backward pass so all copies of the model stay identical.

On an M2 Mac we don't have multiple GPUs, so we simulate this with multiple CPU processes using the "gloo" backend. The DDP mechanics
are identical to what happens across multiple Trainium chips or GPUs — only the communication backend changes.

Run:
    python 02_multiprocess_ddp/train_ddp.py --world_size 2
    python 02_multiprocess_ddp/train_ddp.py --world_size 4 --steps 500
"""

import os
import math
import time
import json
import argparse
import urllib.request

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# =============================================================================
# BLOCK 1: DATA
# =============================================================================
# Identical to Module 01 — same dataset, same character-level tokenization.
# We point DATA_PATH at Module 01's data folder so we don't download the same 1MB file twice.

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
    Identical to Module 01 — DDP changes HOW the dataset is sampled
    (via DistributedSampler), not what the dataset itself looks like.
    """

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
# Identical to Module 01. The model architecture has no awareness of distributed training — DDP wraps it from the outside.

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
# Identical to Module 01 — warmup + cosine decay.
# Every process computes the same LR independently using the same formula and same step number, so no synchronization is needed here.

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

# =============================================================================
# BLOCK 4: DISTRIBUTED SETUP / CLEANUP
# =============================================================================
# These are NEW in Module 02 — they don't exist in single-process training.
#
# Every process that participates in DDP must:
#   1. Know its own "rank" (a unique ID: 0, 1, 2, ... world_size-1)
#   2. Know "world_size" (total number of processes)
#   3. Join a "process group" — a communication channel all processes share
#
# Think of it like a conference call: MASTER_ADDR/MASTER_PORT is the dial-in number, rank is "which participant am I", world_size is
# "how many participants total". Once everyone has dialed in via init_process_group, they can all send/receive tensors to each other.
#
# Backend choice:
#   "nccl" — NVIDIA GPU-to-GPU communication (fastest, needs CUDA)
#   "gloo" — CPU-based communication (works everywhere, what we use on M2)

def setup(rank: int, world_size: int):
    """Join the distributed process group. Called once per process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Leave the process group cleanly. Called once per process, at the end."""
    dist.destroy_process_group()
    
# =============================================================================
# BLOCK 5: TRAINING FUNCTION (runs once per process)
# =============================================================================
# This function is what each process runs independently. mp.spawn() will call this function once per process, automatically passing in `rank`
# as the first argument.

def train(rank: int, world_size: int, args):
    # ── Join the process group ──────────────────────────────────────────
    setup(rank, world_size)

    # ── Device selection ────────────────────────────────────────────────
    # On M2 Mac, MPS does not support multiple processes sharing the GPU
    # the way CUDA does, so for DDP we use CPU. Each process gets its own
    # CPU cores. This mirrors how DDP works across N separate GPUs/Trainium
    # chips — each process owns one device, no sharing.
    device = torch.device("cpu")

    # Only rank 0 prints setup info — otherwise every process prints the
    # same lines and the output becomes unreadable.
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  02 — Multiprocess DDP")
        print(f"  World size: {world_size} processes")
        print(f"  Device: {device} (per process)")
        print(f"{'='*60}\n")
        
# ── Data ─────────────────────────────────────────────────────────────
    text = download_shakespeare()
    dataset = CharDataset(text, block_size=args.block_size)

    if rank == 0:
        print(f"  Dataset: {len(text):,} chars | vocab: {dataset.vocab_size} | "
              f"sequences: {len(dataset):,}")

    # DistributedSampler splits the dataset into world_size non-overlapping
    # chunks. Each process only ever sees its own chunk.
    #
    # Example with world_size=2 and 1000 sequences:
    #   rank 0 sees sequences [0, 2, 4, 6, ...]   (even indices)
    #   rank 1 sees sequences [1, 3, 5, 7, ...]   (odd indices)
    #
    # shuffle=True still happens, but each process shuffles only WITHIN
    # its own chunk, and all processes use the same `seed` so the chunking
    # is deterministic and reproducible across runs.
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,   # sampler replaces shuffle=True at the DataLoader level
        num_workers=0,
    )
    
# ── Model ────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)  # same seed on every process →
                                   # every process starts with IDENTICAL
                                   # random weights before wrapping in DDP
    model = TinyGPT(
        vocab_size  = dataset.vocab_size,
        n_embd      = args.n_embd,
        n_head      = args.n_head,
        n_layer     = args.n_layer,
        block_size  = args.block_size,
        dropout     = 0.1,
    ).to(device)

    if rank == 0:
        print(f"  Model: {model.num_params():,} parameters")
        print(f"  Wrapping model in DistributedDataParallel...\n")

    # ── DDP wrapping ─────────────────────────────────────────────────────
    # This is THE key line of Module 02. DDP wraps the model and:
    #   1. Confirms all processes start with identical weights (it actually
    #      broadcasts rank 0's weights to everyone, as a safety net)
    #   2. Registers a "hook" on every parameter — after loss.backward()
    #      computes local gradients, this hook automatically triggers an
    #      AllReduce: gradients from ALL processes are averaged together,
    #      and every process ends up with the SAME averaged gradient.
    #   3. Because every process then sees the same gradients AND started
    #      with the same weights, optimizer.step() produces the SAME
    #      updated weights on every process — they stay in sync forever,
    #      with no explicit weight-syncing needed after step 0.
    ddp_model = DDP(model, device_ids=None)  # device_ids=None for CPU

# ── Optimizer ────────────────────────────────────────────────────────
    # IMPORTANT: optimizer is built on ddp_model.parameters(), not
    # model.parameters(). DDP doesn't create new parameters — it wraps
    # the same underlying tensors — but always construct the optimizer
    # from the wrapped model's parameters as a convention, since in more
    # advanced setups (FSDP, etc.) the wrapper CAN change parameter objects.
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # ── Training loop ────────────────────────────────────────────────────
    metrics = {
        "world_size": world_size,
        "model_params": model.num_params(),
        "steps": [], "loss": [], "lr": [],
        "step_time_ms": [], "tokens_per_sec": [],
    }

    ddp_model.train()
    data_iter = iter(loader)
    t_start = time.time()

    if rank == 0:
        print(f"Training for {args.steps} steps across {world_size} processes...\n")

    for step in range(args.steps):
        # ── Get a batch ──────────────────────────────────────────────────
        try:
            x, y = next(data_iter)
        except StopIteration:
            # IMPORTANT: set_epoch tells the sampler to reshuffle.
            # Without this, every "epoch" would repeat the EXACT same
            # per-process ordering — fine for a short demo, but a real
            # training run needs this for correct shuffling across epochs.
            sampler.set_epoch(step)
            data_iter = iter(loader)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ── Update learning rate ────────────────────────────────────────
        # Every process computes the same LR independently (Block 3 note).
        lr = get_lr(step, args.warmup, args.steps, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Forward pass ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        logits, loss = ddp_model(x, y)

        # ── Backward pass ────────────────────────────────────────────────
        # This line LOOKS identical to Module 01, but DDP has registered
        # hooks on every parameter. As soon as each parameter's gradient
        # is computed, DDP triggers AllReduce in the background to average
        # that gradient across all processes — overlapping communication
        # with the rest of the backward computation for efficiency.
        # By the time .backward() returns, every process has the SAME
        # averaged gradients.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # ── Gradient clipping ────────────────────────────────────────────
        # Operates on the now-synchronized (averaged) gradients.
        # Since all processes have identical gradients at this point,
        # clipping produces identical results everywhere.
        nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)

        # ── Optimizer step ────────────────────────────────────────────────
        # Same averaged gradients + same starting weights = same updated
        # weights on every process. No explicit weight sync needed.
        optimizer.step()

        step_ms = (time.perf_counter() - t0) * 1000
        # tokens_per_sec here reflects ONE process's throughput.
        # Total system throughput = this value × world_size, since all
        # processes are training simultaneously on different data.
        tokens_per_sec = (args.batch_size * args.block_size) / (step_ms / 1000)

        # ── Logging (rank 0 only) ────────────────────────────────────────
        if rank == 0 and step % args.log_every == 0:
            loss_val = loss.item()
            print(f"  step {step:4d} | loss {loss_val:.4f} | "
                  f"lr {lr:.2e} | {step_ms:.1f}ms | {tokens_per_sec:,.0f} tok/s/proc")
            metrics["steps"].append(step)
            metrics["loss"].append(round(loss_val, 4))
            metrics["lr"].append(round(lr, 6))
            metrics["step_time_ms"].append(round(step_ms, 1))
            metrics["tokens_per_sec"].append(round(tokens_per_sec, 1))

# ── Final stats (rank 0 only) ──────────────────────────────────────────
    total_time = time.time() - t_start
    if rank == 0:
        metrics["total_time_s"] = round(total_time, 1)
        metrics["final_loss"] = round(loss.item(), 4)

        os.makedirs("02_multiprocess_ddp/results", exist_ok=True)
        with open("02_multiprocess_ddp/results/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Done in {total_time:.1f}s")
        print(f"  Final loss: {metrics['final_loss']}")
        print(f"  Metrics saved → 02_multiprocess_ddp/results/metrics.json")
        print(f"{'='*60}")

    # ── Leave the process group ─────────────────────────────────────────────
    cleanup()

# =============================================================================
# BLOCK 6: CLI / LAUNCH
# =============================================================================
# This is the other major difference from Module 01. Instead of running
# train() directly, we use mp.spawn() to launch `world_size` separate
# Python processes, each running train(rank, world_size, args) with a
# different `rank` (0, 1, 2, ... automatically assigned by mp.spawn).

def main():
    p = argparse.ArgumentParser(description="02 — Multiprocess DDP Training")

    # Model size (same defaults as Module 01)
    p.add_argument("--n_embd",     type=int,   default=128)
    p.add_argument("--n_head",     type=int,   default=4)
    p.add_argument("--n_layer",    type=int,   default=4)
    p.add_argument("--block_size", type=int,   default=128)

    # Training
    p.add_argument("--steps",      type=int,   default=500)
    p.add_argument("--batch_size", type=int,   default=32,
                   help="Batch size PER PROCESS (total = batch_size * world_size)")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--warmup",     type=int,   default=50)
    p.add_argument("--log_every",  type=int,   default=50)

    # Distributed
    p.add_argument("--world_size", type=int,   default=2,
                   help="Number of processes to launch")

    # Reproducibility
    p.add_argument("--seed",       type=int,   default=42)

    args = p.parse_args()

    print(f"Launching {args.world_size} processes (gloo backend, CPU)...")
    mp.spawn(
        train,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()