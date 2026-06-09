"""
debug_mps_training.py
=====================
Comparing CPU vs MPS loss trajectory over 20 steps
using identical weights, identical data order (fixed seed).
Goal: see exactly where MPS diverges from CPU.
"""
import sys, torch, torch.nn as nn
sys.path.insert(0, "01_single_gpu_baseline")
from train import TinyGPT, CharDataset, download_shakespeare, get_lr
from torch.utils.data import DataLoader

text = download_shakespeare()
dataset = CharDataset(text, block_size=128)

def run(device_name, steps=20):
    torch.manual_seed(0)
    device = torch.device(device_name)
    model = TinyGPT(dataset.vocab_size, 128, 4, 4, 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    torch.manual_seed(0)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    data_iter = iter(loader)
    losses = []
    for step in range(steps):
        x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(round(loss.item(), 4))
    return losses

print("Running CPU...")
cpu_losses = run("cpu")
print("Running MPS...")
mps_losses = run("mps")

print(f"\n{'Step':<6} {'CPU':>8} {'MPS':>8} {'Delta':>10}")
print("-" * 35)
for i, (c, m) in enumerate(zip(cpu_losses, mps_losses)):
    delta = m - c
    flag = " ⚠️" if abs(delta) > 0.1 else ""
    print(f"{i:<6} {c:>8.4f} {m:>8.4f} {delta:>+10.4f}{flag}")
