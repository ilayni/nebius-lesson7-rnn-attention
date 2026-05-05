# Part 2 — Task 4: Training

## Goal

Train the CharRNN model to predict the next character in a dinosaur name sequence.

## What was fixed in the provided train function

The provided function had two bugs that prevented it from running correctly:

| Bug | Fix |
|-----|-----|
| `inputs.to(device)` — `.to()` is not in-place, tensors stayed on CPU | `inputs = inputs.to(device)` |
| `targets.view(batch_size*n_steps)` — fails when last batch is smaller | `targets.view(-1)` |

Also added `drop_last=True` to both DataLoaders so every batch is exactly 64 samples, matching the fixed hidden state size initialized at the start of each epoch.

## Training configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch size | 64 |
| Sequence length | 10 |
| Hidden size | 256 |
| LSTM layers | 2 |
| Dropout | 0.5 |
| Optimizer | Adam, lr=0.001 |
| Gradient clipping | 5 |
| Device | MPS |

## Training loss

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | 2.88 → 1.94 | 2.86 → 1.91 |
| 3 | ~1.72 | ~1.74 |
| 5 | ~1.58 | ~1.67 |
| 7 | ~1.54 | ~1.65 |
| 10 | ~1.37 | ~1.64 |

## Conclusions

- Loss dropped from **2.88 → ~1.45** (train) and **2.86 → ~1.64** (val) — the model learned the character distribution of dinosaur names
- The gap between train and val loss is small, indicating no severe overfitting
- Val loss plateaus around epoch 5–6, suggesting the model has reached its capacity with this architecture and dataset size. More epochs or a larger hidden size could push it further
- Starting loss ~2.88 ≈ ln(18), meaning the model initially predicted ~18 characters with equal probability; by epoch 10 it has narrowed down to roughly e^1.64 ≈ 5 equally likely candidates per step

## Saved to disk

| File | Contents |
|------|----------|
| `saved_models/charrnn.pt` | Model weights, vocab_size, n_hidden, n_layers |
