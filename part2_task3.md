# Part 2 — Task 3: CharRNN Model + Forward Pass

## Goal

Understand the CharRNN model architecture and trace data through each layer for a single batch.

## Architecture

A 2-layer LSTM with dropout. No embedding layer — one-hot vectors go directly into the LSTM.

```
Input IDs (64, 10)
    ↓ one-hot encode
(64, 10, 28)
    ↓ LSTM (2 layers, hidden=256, dropout=0.5)
(64, 10, 256)
    ↓ Dropout(0.5)
(64, 10, 256)
    ↓ Reshape (flatten batch × seq)
(640, 256)
    ↓ Linear(256 → 28)
(640, 28)   ← logits
    ↓ Softmax
(640, 28)   ← probability distribution over vocab
```

## Step-by-step output (untrained model)

| Step | Layer | Shape |
|------|-------|-------|
| 1 | Input IDs | `(64, 10)` |
| 2 | After one-hot | `(64, 10, 28)` |
| 3 | After LSTM | `(64, 10, 256)` |
| 4 | After dropout | `(64, 10, 256)` |
| 5 | After reshape | `(640, 256)` |
| 6 | After FC (logits) | `(640, 28)` |
| 7 | After softmax | `(640, 28)` — sums to 1 ✓ |

## Conclusion

Before training, all 28 characters get equal probability ≈ **0.038 (= 1/28)** — pure random chance. This is the expected baseline that training in Task 4 must improve on. The reshape step (step 5) is the key design choice that allows a single `Linear` layer to score every time step in the batch simultaneously.
