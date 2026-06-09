# Bugs & Debugging Notes

Real issues encountered while building this repo — how they were
investigated and what was actually found.

---

## Bug 01 — MPS Loss Increasing During Training
**Module:** 01_single_gpu_baseline
**Status:** ✅ Fixed

### Observed
`--device mps` produced increasing loss (4.24 → 5.13) over 500 steps.
`--device cpu` trained correctly (4.17 → 2.46) on the same code.

### Investigated
1. Checked for NaN in logits/loss/gradients → none found
2. Checked causal mask device after `.to(mps)` → correctly on `mps:0`
3. Fixed seed, same data order, 20 steps CPU vs MPS → both identical

### Root cause
No fixed random seed between runs. The two runs loaded different batches
in different order. MPS was fine — it just saw harder batches by chance.

### Fix
Added `--seed` argument to `train.py`. `torch.manual_seed(args.seed)`
set before training ensures reproducible runs across devices.

### Lesson
> Control for randomness before assuming a hardware bug.

---
<!-- Future bugs will be added here as modules progress -->
