# Part 2 — Task 2: One-Hot Encoding

## Goal

Implement a function to convert sequences of character IDs into one-hot encoded vectors, to be used as input to the LSTM instead of an embedding layer.

## What was built

```python
def one_hot_encode(array, vocab_size):
    array = np.array(array)
    return np.eye(vocab_size, dtype=np.float32)[array]
```

`np.eye(vocab_size)` is an identity matrix of shape `(vocab_size, vocab_size)`. Indexing it with character IDs directly produces the one-hot matrix — one line, no loops.

## Output

| | Shape | Description |
|---|---|---|
| Input | `(64, 10)` | Batch of 64 sequences, each 10 character IDs |
| Output | `(64, 10, 28)` | Each character ID expanded to a 28-dim one-hot vector |

**Verification:**
- Every row sums to exactly 1.0 ✓
- All values are 0 or 1 ✓
- Token ID and one-hot argmax match ✓

**Example** — character `b` (id=2, vocab size=28):

```
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```
