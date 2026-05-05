# Part 2 — Task 1: Build a PyTorch Dataset for RNN Training

## Goal

Prepare character-level sequence data from a dinosaur names dataset to train a CharRNN model.

## What was built

**`DinosDataset`** — a custom PyTorch Dataset that:
- Accepts either a file path (`.txt`) or a list of names
- Wraps each name with start `<` and end `>` tokens: `"anna"` → `"<anna>"`
- Concatenates all wrapped names into one long string
- Builds a vocabulary of unique characters and creates `token_to_id` / `id_to_token` mappings
- Converts the full text to a tensor of character IDs
- Returns sliding window pairs `(x, y)` where `y` is `x` shifted by one character — the next-character prediction target

The train/val split was done using HuggingFace `Dataset.train_test_split()` (80/20, seed=42).
A **shared vocabulary** was built from all names before splitting so both datasets use identical token IDs.

DataLoaders wrap both splits with `batch_size=64`, `drop_last=True`.

## Key design decisions

- `seq_len = 10` — each input/target sequence is 10 characters
- Shared vocab across train and val — prevents token ID mismatch between splits
- `drop_last=True` — ensures every batch is exactly 64 samples, required by the fixed hidden state shape in CharRNN

## Output

| | Value |
|---|---|
| Total names | 1,544 |
| Train names | 1,236 → 17,167 sequences → 268 batches |
| Val names | 309 → 4,277 sequences → 66 batches |
| Vocab size | 28 (`<>abcdefghijklmnopqrstuvwxyz`) |
| Batch shape | `x: (64, 10)`, `y: (64, 10)` |

**Example sliding window:**

| | Characters |
|---|---|
| Input `x` | `<aachenosa` |
| Target `y` | `aachenosau` |

## Saved to disk

| File | Contents |
|------|----------|
| `saved_models/dinos.txt` | Raw dinosaur names (downloaded once) |
| `saved_models/dinos_vocab.json` | Shared vocab + seq_len |
| `saved_models/dinos_train_names.json` | Train name list |
| `saved_models/dinos_val_names.json` | Val name list |
