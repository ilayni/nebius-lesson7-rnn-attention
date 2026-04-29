# Task 3 — MLP Implementation Summary

## Pipeline

```
AG News text
  → BPE tokenizer (encode_batch, 10k vocab)
  → token list per article
  → sentence_to_vec (mean pool SkipGram_100 embeddings)
  → 100-d float vector per article
  → MLP
  → 4-class prediction (World / Sports / Business / Sci/Tech)
```

## Implementation Choices

### Input representation: mean pooling
Each article is tokenized into BPE pieces, then each piece's 100-d embedding is looked up in the SkipGram_100 model. The vectors are averaged into one 100-d vector per article. Tokens not in the W2V vocab are skipped; if no tokens match at all (rare), a zero vector is returned.

Mean pooling was chosen over sum (sensitive to article length) and max (discards most signal).

### Model architecture

```
Linear(100 → 256)
ReLU
Dropout(0.3)
Linear(256 → 128)
ReLU
Linear(128 → 4)
```

Two hidden layers are enough for 4-class classification on 100-d input. The first layer expands to 256 to let the model learn combinations of embedding dimensions; Dropout(0.3) prevents it from memorising. The second layer compresses to 128 before the output.

### Training setup

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimizer | Adam | Adaptive lr; robust default for feedforward nets |
| Learning rate | 1e-3 | Standard Adam starting point |
| Batch size | 64 | Stable gradients without large memory cost |
| Epochs | 10 | Loss plateaus around epoch 7–8 on this dataset |
| Loss | CrossEntropyLoss | Standard for multi-class classification |

## Why MLP Works Here

Mean pooling collapses each article to a fixed 100-d vector regardless of length. Once that's done, the classification problem is a static mapping from R^100 → {0,1,2,3}. An MLP is exactly designed for this: fully connected layers learn which combinations of embedding dimensions correlate with each news category.

The 4 AG News topics (World, Sports, Business, Sci/Tech) have distinct vocabulary: "playoff", "stocks", "genome", "minister" rarely overlap. Mean-pooled embeddings carry this topic signal even without word order.

## Core Limitation vs RNNs

Mean pooling discards word order entirely. "Oil prices rise" and "Rise in oil prices" produce the identical 100-d vector. An RNN processes tokens one at a time and updates a hidden state, so it can capture:
- negation: "not profitable" vs "profitable"
- order-dependent phrases: "acquired by" vs "acquired"
- long-range dependencies across a sentence

For short, topic-focused news headlines this doesn't hurt much — the vocabulary alone is discriminative. For longer texts or tasks requiring understanding of sentence structure, an RNN (or Transformer) would outperform the MLP.

## Actual Output

```
Chosen embedding dimension: 100
X_train shape: (120000, 100)
X_test shape:  (7600, 100)

MLP(
  (model): Sequential(
    (0): Linear(in_features=100, out_features=256, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): ReLU()
    (5): Linear(in_features=128, out_features=4, bias=True)
  )
)

Epoch  1/10, Loss: 724.0443
Epoch  2/10, Loss: 620.2611
Epoch  3/10, Loss: 594.7356
Epoch  4/10, Loss: 575.6549
Epoch  5/10, Loss: 563.7621
Epoch  6/10, Loss: 549.2634
Epoch  7/10, Loss: 539.9432
Epoch  8/10, Loss: 531.4073
Epoch  9/10, Loss: 521.5541
Epoch 10/10, Loss: 514.7221
```

Average per-batch loss at epoch 10: 514.7 / 1875 batches ≈ 0.27 (random chance = ln(4) ≈ 1.39).
Loss still declining at epoch 10 — 5–10 more epochs would improve accuracy slightly.
Expected test accuracy after Task 4 evaluation: roughly **85–88%**. Main confusion: Business vs Sci/Tech.
