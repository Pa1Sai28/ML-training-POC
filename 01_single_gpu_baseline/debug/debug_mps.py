import sys, torch
sys.path.insert(0, "01_single_gpu_baseline")
from train import TinyGPT, CharDataset, download_shakespeare
from torch.utils.data import DataLoader

device = torch.device("mps")
text = download_shakespeare()
dataset = CharDataset(text, block_size=128)
model = TinyGPT(dataset.vocab_size, 128, 4, 4, 128).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
data_iter = iter(loader)

print("Checking for NaN at each step:\n")
print(f"{'Step':<6} {'Loss':<10} {'NaN logits':<12} {'NaN loss':<10} {'NaN grad'}")
print("-" * 50)
for step in range(5):
    x, y = next(data_iter)
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    loss_val = loss.item()
    nan_logits = torch.isnan(logits).any().item()
    nan_loss = torch.isnan(loss).any().item()
    loss.backward()
    grad = model.tok_emb.weight.grad
    nan_grad = torch.isnan(grad).any().item() if grad is not None else False
    print(f"{step:<6} {loss_val:<10.4f} {str(nan_logits):<12} {str(nan_loss):<10} {nan_grad}")
    optimizer.zero_grad(set_to_none=True)
