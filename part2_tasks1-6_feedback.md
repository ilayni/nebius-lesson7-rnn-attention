# Feedback — Part 2 Tasks 1–6

Overall the implementation is solid and all tasks are complete. Two meaningful issues below.

---

## Task 1 — DinosDataset ✅

Clean implementation. The shared vocab across train/val splits is a good design decision — it guarantees consistent token IDs between the two datasets. `drop_last=True` on both DataLoaders is correctly applied.

---

## Task 2 — One-hot encoding ✅

Correct and concise. `np.eye(vocab_size)[array]` is the right approach.

---

## Task 3 — Forward pass demo ✅

Shapes at each layer are correctly printed and explained. One minor note: the demo initializes a fresh untrained model, so the sample probabilities shown are ~0.038 per character (uniform random, 1/28). This is fine for demonstrating shapes, but noting it explicitly in the cell would make it clearer to the reader that the output is meaningless until training.

---

## Task 4 — Training ⚠️

**Hidden state should be reinitialized per batch, not per epoch.**

The current code:
```python
for e in range(epochs):
    h = net.init_hidden(batch_size, device)   # ← once per epoch
    for x, targets in train_dataloader:
        h = tuple([each.data for each in h])  # carries state across batches
        ...
```

Because `shuffle=True` is set on the DataLoader, each batch is a random sample of sequences from different parts of the dataset — they are not consecutive. Carrying the hidden state from one shuffled batch to the next passes context that is unrelated to the current input, which is incorrect. The hidden state should be reset at the start of every batch:

```python
for e in range(epochs):
    for x, targets in train_dataloader:
        h = net.init_hidden(batch_size, device)  # ← once per batch
        ...
```

This was flagged as a logical error in the course discussion and confirmed by the instructor.

---

## Task 5 — Name generation ✅

Prefix conditioning works correctly across all tested prefixes. Good variety of examples.

---

## Task 6 — Sampling ⚠️

**Code:** Correct. Top-k and temperature are both implemented properly.

**Written answer (cell 61):** The conclusion says *"Top-k=5 with temperature=0.5 gives the most consistently plausible dinosaur names"* — but the actual generated output in the cell shows temp=0.7 producing better results (`<megalopelta>`, `<archaeosaurus>`, `<alimimus>`, `<pachysaurus>`) compared to temp=0.5 which over-repeats the `-saurus` suffix. The written conclusion should match the experimental finding: **temp=0.7 is the sweet spot**, not 0.5.
