# Lesson 7 Homework — Implementation Plan

Source: `LLM_Architectures_hometask_2_Tel_Aviv.ipynb`
Course: Nebius Academy — LLM Architectures (Homework 2)

Two parts, 50 pts each:
- **Part 1**: AG News text classification — BPE/WordPiece → Word2Vec → MLP.
- **Part 2**: Char-level LSTM that generates dinosaur names.

---

## 0. Environment & Setup

Install once at the top of the notebook:
```python
!pip install --upgrade tokenizers gensim datasets torch joblib
```
- `tokenizers` (HuggingFace): trains BPE/WordPiece. Docs: https://huggingface.co/docs/tokenizers/quicktour
- `gensim`: trains Word2Vec.
- `datasets` (HuggingFace): loads `ag_news` and provides `train_test_split`.
- `torch` + `torch.utils.data`: model + dataloaders.
- `joblib`: simple parallel map used in Task 3 vectorization.

The first cell already loads AG News into `train_df`/`test_df` (120k / 7.6k rows, columns: `text`, `label`; 4 classes: World, Sports, Business, Sci/Tech).

---

## Parallelism Guide

Rule: tasks expected to run **> 2 minutes** get a parallel implementation. Shorter steps run sequentially.

| Step | Est. sequential time | Parallel approach | Expected speedup |
|------|---------------------|-------------------|-----------------|
| Task 1 — tokenize 120k sentences (batch encode) | ~10–20 s | `tokenizer.encode_batch()` already parallel (Rust) | built-in, no extra code |
| Task 2 — train 6 Word2Vec models | ~15–30 min | `ProcessPoolExecutor` — models are independent | ~5–6× on 6 cores |
| Task 3 — build X_train (sentence_to_vec × 120k) | ~3–8 min | `joblib.Parallel(n_jobs=-1)` | ~4–8× |
| Task 3 — MLP training (~10 epochs) | ~1–2 min | Sequential (single GPU/CPU loop, can't parallelise) | — |
| Part 2 — CharRNN training | ~5–15 min | Sequential (stateful RNN, can't parallelise epochs) | — |

**Pattern 1 — Word2Vec model training** (Task 2):

Train sequentially with `workers=4` per model. Gensim parallelises internally via Cython
threads — no Python executor needed and no concurrency errors.

```python
models = {}
for name, sg, dim in configs:
    models[name] = Word2Vec(sentences=tokenized_sentences, vector_size=dim,
                            window=5, min_count=2, sg=sg, workers=4, seed=42)
```

**Pattern 2 — Parallel map over large dataset** (Task 3):
```python
from joblib import Parallel, delayed

X_train = np.array(
    Parallel(n_jobs=-1)(
        delayed(sentence_to_vec)(t, w2v_model, embedding_dim)
        for t in train_tokens
    ), dtype=np.float32
)
```
`n_jobs=-1` uses all available cores. On Colab (2 vCPUs) this is modest; on a local machine with 8+ cores it matters a lot.

---

## Glossary (terms you'll meet)

| Term | One-line definition |
|------|---------------------|
| **Tokenization** | Splitting text into atomic units the model can map to integers. |
| **Subword tokenizer** | Splits unknown/rare words into known fragments (e.g. `unbelievable` → `un + believ + able`). Solves the "unknown word" problem of word-level. |
| **BPE** | Byte-Pair Encoding. Greedy: merges the most frequent adjacent token pair, repeats until vocab size hit. Used by GPT family. |
| **WordPiece** | Like BPE but merges the pair that maximises corpus likelihood, not raw frequency. Marks continuation pieces with `##`. Used by BERT. |
| **Vocabulary size** | Number of distinct tokens the tokenizer can emit. ~10k is a standard small choice. |
| **Embedding** | Dense low-dimensional float vector that represents a token (e.g. 100-d). |
| **Word2Vec** | Self-supervised algorithm for learning embeddings from raw text. Two flavours: |
| **CBOW** | Predict centre word from context words. Faster, smooths over rare words. |
| **Skip-gram** | Predict context words from centre word. Slower, better for rare words. |
| **MLP** | Multi-Layer Perceptron — a stack of `Linear → activation` layers, no recurrence. |
| **Pooling** | Reducing a variable-length sequence of vectors to one fixed-size vector (mean/sum/max). |
| **Confusion matrix** | `M[i,j]` = number of test items with true label `i` predicted as `j`. Diagonal = correct. |
| **One-hot encoding** | Vector of length `vocab_size`, all zeros except a single 1 at the token's index. |
| **RNN / LSTM** | Sequence model that consumes one token at a time and carries a hidden state. LSTM adds gates that fight vanishing gradients. |
| **Char-level model** | Vocabulary is individual characters (~28 here, including `<` and `>`). |
| **Softmax** | Turns logits into a probability distribution over the vocab. |
| **Top-k sampling** | At each step, restrict sampling to the `k` highest-probability tokens. |
| **Temperature** | Divide logits by `T` before softmax. `T<1` sharpens (greedier), `T>1` flattens (more random). |

---

# Part 1 — AG News Classification (50 pts)

Pipeline: `text → BPE tokens → token embeddings → pooling → MLP → 4 classes`.

## Task 1 — Tokenization (10 pts)

**Goal**: train one BPE and one WordPiece tokenizer on AG News *training text only*; tokenize 5 sample sentences with each.

**Steps**
1. Build a small training corpus iterator from `train_df["text"]`.
2. BPE:
   ```python
   from tokenizers import Tokenizer
   from tokenizers.models import BPE
   from tokenizers.trainers import BpeTrainer
   from tokenizers.pre_tokenizers import Whitespace
   bpe = Tokenizer(BPE(unk_token="[UNK]"))
   bpe.pre_tokenizer = Whitespace()
   bpe.train_from_iterator(train_df["text"], BpeTrainer(vocab_size=10000, special_tokens=["[UNK]"]))
   ```
3. WordPiece: same shape, swap `BPE`→`WordPiece`, `BpeTrainer`→`WordPieceTrainer`.
4. Tokenize the 5 given sentences via `tokenizer.encode(s).tokens` and print a comparison table.

**Question to answer (in markdown)**
- *BPE vs WordPiece*: BPE picks merges by **frequency**; WordPiece picks by **likelihood gain**. WordPiece marks subword continuations with `##`.
- *When to prefer*: BPE for general-purpose / GPT-like; WordPiece when you want explicit word-boundary cues (BERT-like).
- *Why both beat word-level*: handle out-of-vocabulary words, capture morphology, much smaller vocab.

**Save the tokenizer** — Task 2 needs it.

## Task 2 — Word2Vec embeddings (10 pts)

**Goal**: train 6 Word2Vec models (CBOW × {50,100,200} and Skip-gram × {50,100,200}) on the BPE-tokenized training corpus.

**Steps**
1. Tokenize training corpus — use `encode_batch` (already parallel, Rust-backed, ~10 s):
   ```python
   tokenized_sentences = [e.tokens for e in bpe_tokenizer.encode_batch(train_df["text"].tolist())]
   ```
2. **⚡ Parallel model training** (~15–30 min sequential → ~3–5 min parallel) — use `ProcessPoolExecutor` pattern from the Parallelism Guide above. Set `workers=2` inside each Word2Vec so the 6 processes don't fight over cores.
3. For 3 sample tokens (e.g. `"market"`, `"sport"`, `"film"` — pick tokens that *exist* in `model.wv`), print:
   - first 10 dims of the vector
   - `model.wv.most_similar(token, topn=5)`
4. Compare CBOW vs Skip-gram qualitatively on `most_similar` outputs.

**Questions to answer**
- Which (CBOW or Skip-gram) gave more semantically related neighbours? On AG News (short news headlines), CBOW often wins because each context window is small; mention what you saw.
- Which dimension you picked and why (more dims = richer but slower & risk overfit on small corpus).
- *Tokenizer vocab vs W2V vocab*: tokenizer holds 10k fixed tokens; W2V drops anything below `min_count`, so its vocab is **smaller**.

## Task 3 — MLP classifier (10 pts)

**Goal**: classify each AG News headline into one of 4 classes using pooled embeddings.

**Steps**
1. Pick your best W2V model from Task 2 — store its `vector_size` as `embedding_dim`.
2. Implement `sentence_to_vec(tokens, model, dim)` returning **mean-pooled** vector (sum & max are valid alternatives — pick one and justify). Skip tokens not in `model.wv`. Return zeros if all missing.
3. **⚡ Parallel vectorization** (~3–8 min sequential → ~1–2 min parallel) — use `joblib.Parallel` pattern from the Parallelism Guide above to build `X_train` and `X_test`. Then cast to `np.float32`.
   ```python
   # Also tokenize test set first:
   test_tokens = [e.tokens for e in bpe_tokenizer.encode_batch(test_df["text"].tolist())]
   ```
   Build `X_train`, `X_test` arrays with shape `(N, embedding_dim)`; labels are already integers 0–3.
4. Wrap in the provided `TextDataset`; pick `batch_size=64` (or 128).
5. Define `MLP` with `nn.Sequential`. Suggested starter:
   ```python
   nn.Linear(embedding_dim, 128), nn.ReLU(), nn.Dropout(0.2),
   nn.Linear(128, 64), nn.ReLU(),
   nn.Linear(64, 4)
   ```
6. `criterion = nn.CrossEntropyLoss()`; `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`.
7. Train ~5–10 epochs; print loss per epoch.

**State your design choices** (input rep, hidden layers/dims, activation, optimizer, lr, batch size, epochs).

**Question to answer**
- *Why MLP works here*: pooling already gave a fixed-size vector; classification is a static mapping.
- *Limitation vs RNN*: pooling **discards word order** — "dog bites man" and "man bites dog" share the same mean vector.

## Task 4 — Evaluation & error analysis (20 pts)

**Steps**
1. Eval loop: `model.eval()`, `with torch.no_grad():`, collect predictions for the test set.
2. Compute accuracy: `(preds == y_test).mean()`.
3. Confusion matrix via `sklearn.metrics.confusion_matrix`; visualise with `ConfusionMatrixDisplay` or `seaborn.heatmap`.
4. Print 3 misclassified examples — original text, true label name, predicted label name.
5. Discuss patterns (typical: confusion between Business and Sci/Tech because tech-business stories overlap).
6. Propose ONE interpretability method (no implementation needed). Suggested: **LIME** (perturb tokens and see how the prediction shifts) or **integrated gradients** on the input embeddings.

---

# Part 2 — Char-RNN Dinosaur Name Generator (50 pts)

Dataset: text file of dinosaur names, one per line. Source gist: https://gist.github.com/judahsemi/210d42e0410c122aa0e437861c46cf94

## Task 1 — `DinosDataset` (10 pts)

**Steps**
1. Load names (read file or accept list).
2. Wrap each: `"<" + name.strip().lower() + ">"`. The `<`/`>` tokens teach the model where names start/end.
3. Concatenate all wrapped names into one long string `self.text`.
4. Build `self.vocab = sorted(set(self.text))`; `token_to_id` & `id_to_token` dicts.
5. Encode full text to a list of ids → `self.data`.
6. `__len__`: `len(self.data) - self.seq_len` (sliding window).
7. `__getitem__(idx)`: return `x = ids[idx : idx+seq_len]`, `y = ids[idx+1 : idx+seq_len+1]` as `torch.long` tensors.
8. After defining the class:
   - Pick `seq_len=10`.
   - Split: use `torch.utils.data.random_split` with ~90/10 train/val.
   - Wrap in `DataLoader(..., batch_size=64, shuffle=True, drop_last=True, num_workers=2)` — `drop_last=True` matters because `train()` assumes full batches. `num_workers=2` prefetches batches in parallel (free speedup, no code change needed).

## Task 2 — `one_hot_encode` (5 pts)

```python
def one_hot_encode(array, vocab_size):
    arr = np.array(array)
    out = np.zeros((*arr.shape, vocab_size), dtype=np.float32)
    # Use np.indices or fancy indexing to set out[..., arr] = 1
    np.put_along_axis(out, arr[..., None], 1.0, axis=-1)
    return out
```
Verify: `encoded.shape == (batch, seq_len, vocab_size)`.

## Task 3 — Forward pass demo (5 pts)

`CharRNN` is already provided. You only need to:
1. Take a single batch from `train_loader`.
2. One-hot encode `x` → tensor.
3. Initialise hidden via `model.init_hidden(batch_size, device)`.
4. Walk through manually:
   - print `inputs.shape` (batch, seq_len, vocab_size)
   - `lstm_out, (h, c) = model.lstm(inputs, hc)`; print `lstm_out.shape`
   - after dropout + reshape + linear: print `logits.shape` (batch*seq_len, vocab_size)
5. `probs = F.softmax(logits, dim=1)`; print one row's sum (should be ≈1).

## Task 4 — Training (10 pts)

Function `train(...)` is provided. You only need to:
1. Detect device: `device = "cuda" if torch.cuda.is_available() else "cpu"`.
2. Instantiate: `model = CharRNN(vocab_size=len(train_data.vocab), n_steps=seq_len)`.
3. Call `train(model, train_loader, val_loader, epochs=20, batch_size=64, n_steps=seq_len, device=device)`.
4. Watch that train & val losses both go down — they should both end well below 2.0.

**Gotcha**: `train()` calls `targets.view(batch_size*n_steps)` — the dataloaders MUST yield exactly that batch shape, hence `drop_last=True` and a fixed `batch_size`.

## Task 5 — Generate names (5 pts)

`predict()` and `sample()` are provided. Just:
1. Generate ~10 names with prefix `"<"` (the default start token).
2. Try other prefixes: `"<a"`, `"<tyr"`, `"<bron"` — they seed the LSTM hidden state.
3. Show the outputs in a markdown cell.

## Task 6 — Top-k & temperature sampling (15 pts)

**Modify `predict`** to accept `top_k` and `temperature`:
```python
def predict(char, model, id_to_token, token_to_id, h=None, device='cpu',
            top_k=None, temperature=1.0):
    # ... encode + forward as before, get logits `out`
    out = out / temperature                    # temperature scaling
    p = F.softmax(out, dim=1).data.cpu().numpy().squeeze()
    if top_k is not None:
        top_idx = p.argsort()[-top_k:]         # keep k highest
        mask = np.zeros_like(p); mask[top_idx] = p[top_idx]
        p = mask / mask.sum()
    char_id = np.random.choice(len(p), p=p)
    return id_to_token[char_id], h
```
Propagate `top_k`/`temperature` into `sample()`.

**Compare** (in markdown):
- Plain random sampling vs `top_k=5` — top-k removes "tail noise" so names stay more dinosaur-like.
- Temperatures `0.5`, `1.0`, `1.5` — `0.5` produces very repetitive, common-pattern names; `1.5` produces wild/unreadable strings.
- A short paragraph of insights: "lower temperature = more conservative; top-k bounds risk; both reduce probability of rare-character glitches."

---

## Suggested Working Order

1. Setup cell (installs + imports).
2. Part 1 Task 1 (BPE + WordPiece). Save tokenizers.
3. Part 1 Task 2 (W2V x 6). Pick best, keep its handle.
4. Part 1 Task 3 (MLP). Save trained model.
5. Part 1 Task 4 (eval + analysis).
6. Part 2 Task 1 (Dataset + loaders).
7. Part 2 Task 2 (one_hot_encode) — verify shapes.
8. Part 2 Task 3 (manual forward pass).
9. Part 2 Task 4 (train CharRNN).
10. Part 2 Task 5 (sample names).
11. Part 2 Task 6 (top-k + temperature).

## Submission Checklist

- [ ] All TODOs replaced with real code (no `...` left).
- [ ] Every cell executed; outputs visible (notebook NOT cleared).
- [ ] All "write your answers" markdown cells have actual answers.
- [ ] Confusion matrix is rendered.
- [ ] At least 5 generated dinosaur names shown for each sampling strategy.
- [ ] No incomplete placeholders.
