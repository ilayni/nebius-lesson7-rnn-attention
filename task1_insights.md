# Task 1 Insights — Tokenization (BPE & WordPiece)

## AG News Dataset

**Labels** (4 classes, balanced):

| ID | Name | What it covers |
|----|------|----------------|
| 0 | World | International news, politics, conflicts |
| 1 | Sports | Athletic events, players, results |
| 2 | Business | Finance, markets, corporate news |
| 3 | Sci/Tech | Science, technology, software, gadgets |

**Data format**: Each sample has two fields:
- `text` — headline + short news snippet, typically 1–3 sentences (~50–150 words).
- `label` — integer 0–3.

**Size**: 120,000 train / 7,600 test. Classes are exactly balanced (30k train each).

---

## What the code does (step by step)

### 1. Install & imports
`!pip install -q tokenizers` gets the HuggingFace `tokenizers` library — a fast Rust-backed tokenizer trainer.

### 2. Train BPE
```
Tokenizer(BPE(unk_token="[UNK]"))  ← base model
.pre_tokenizer = Whitespace()      ← split on spaces first, then BPE merges
.train_from_iterator(train_df["text"], BpeTrainer(vocab_size=10_000))
```
- Processes ~120k texts.
- Starts from individual characters, greedily merges the most frequent pair until vocabulary hits 10,000.
- Result: vocab contains common full words + subpieces for rarer ones.

### 3. Train WordPiece
Same shape, different internals — merges are chosen by **likelihood gain** rather than raw count.
WordPiece also marks continuation subpieces with `##` (e.g., `"##ble"` means this piece is not a word start).

### 4. Tokenise 5 sentences & print table
`bpe_tokenizer.encode(sentence).tokens` returns a Python list of string tokens.
The code prints a side-by-side comparison so differences are visible.

---

## BPE vs WordPiece — key differences

| | BPE | WordPiece |
|---|---|---|
| Merge criterion | Most **frequent** pair | Maximises **likelihood** (mutual info) |
| Continuation marker | None | `##` prefix |
| Typical use | GPT / LLaMA | BERT / DistilBERT |
| Boundary recovery | Inferred by consumer | Explicit via `##` |

**Concrete example** (likely output on these sentences):

| Sentence | BPE | WordPiece |
|----------|-----|-----------|
| `unbelievable` | `["un", "believ", "able"]` | `["un", "##bel", "##ie", "##va", "##ble"]` |
| `microorganism` | `["micro", "organ", "ism"]` | `["micro", "##organ", "##ism"]` |
| `ultra-fast` | `["ultra", "-", "fast"]` | `["ultra", "-", "fast"]` |

WordPiece tends to produce finer splits on very rare words because the likelihood criterion is more conservative about creating large merged tokens.

## When to prefer each

**BPE** when:
- Building a GPT-style autoregressive model.
- Multilingual settings — BPE handles byte-level extensions (ByteBPE) well.
- You don't need explicit word-boundary cues.

**WordPiece** when:
- Building a BERT-style bidirectional model where `##` signals benefit attention masking.
- Morphologically rich languages (Turkish, Finnish) where finer suffixes matter.
- Downstream tasks that explicitly use the `##` prefix.

## Why both beat word-level tokenization

1. **No `[UNK]`** — any word, no matter how rare, can be split into known subpieces.
2. **Morphology sharing** — `run`, `runner`, `running` share the subpiece `run`; the model transfers what it learns about the root.
3. **Vocabulary efficiency** — 10k subword tokens cover the same corpus as ~100k+ word tokens; the embedding matrix is ~10× smaller.
4. **Better rare-word embeddings** — a word seen once gets a poor embedding; its subpieces each appear many times and get good embeddings that compose into a reasonable representation.

## Expected output when you run the cell

- `BPE vocab size: 10,000`
- `WordPiece vocab size: 10,000`
- A table of 5 sentences with their BPE and WordPiece tokenisations side by side.

Common thing to notice: short, common words (`the`, `in`, `a`) remain intact under both. Longer or rarer words (`microorganism`, `unbelievable`) get split, with WordPiece generally producing more `##`-marked fragments.
