"""
debug_mps_mask.py
=================
Checking if the causal mask moves correctly to MPS device.
If mask stays on CPU while model runs on MPS, attention
has no causal masking — model sees future tokens and learns
a broken shortcut.
"""
import sys, torch
sys.path.insert(0, "01_single_gpu_baseline")
from train import TinyGPT, CharDataset, download_shakespeare

device = torch.device("mps")
text = download_shakespeare()
dataset = CharDataset(text, block_size=128)
model = TinyGPT(dataset.vocab_size, 128, 4, 4, 128).to(device)

# Check where the mask actually lives after .to(device)
print("Checking buffer devices after .to(mps):\n")
for name, buf in model.named_buffers():
    print(f"  {name:<45} device={buf.device}  shape={list(buf.shape)}")

# Check the mask values — should be lower triangular
mask = model.blocks[0].attn.mask
print(f"\nCausal mask (first 6x6):\n{mask[0,0,:6,:6].long()}")
print("\nExpected lower triangular:")
print("[[1,0,0,0,0,0]")
print(" [1,1,0,0,0,0]")
print(" [1,1,1,0,0,0] ...]")

# Now run CPU vs MPS loss on the SAME batch with SAME weights
print("\n--- Same batch, same weights: CPU vs MPS ---")
torch.manual_seed(42)
model_cpu = TinyGPT(dataset.vocab_size, 128, 4, 4, 128)
# Copy exact same weights to MPS model
model_mps = TinyGPT(dataset.vocab_size, 128, 4, 4, 128).to(device)
model_mps.load_state_dict(model_cpu.state_dict())

x, y = dataset[0]
x, y = x.unsqueeze(0), y.unsqueeze(0)

with torch.no_grad():
    _, loss_cpu = model_cpu(x, y)
    _, loss_mps = model_mps(x.to(device), y.to(device))

print(f"  CPU loss : {loss_cpu.item():.6f}")
print(f"  MPS loss : {loss_mps.item():.6f}")
print(f"  Difference: {abs(loss_cpu.item() - loss_mps.item()):.6f}")
print(f"  {'✅ Match' if abs(loss_cpu.item() - loss_mps.item()) < 0.01 else '❌ Mismatch — device issue confirmed'}")
