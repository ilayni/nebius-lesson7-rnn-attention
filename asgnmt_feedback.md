# Assignment Grade — LLM Architectures Hometask 2

## Overall: 99 / 100

Strong, complete submission across both parts. All deliverables present, all written answers grounded in actual experimental output, and design choices clearly justified. The code is well-structured and standalone — most cells can run independently from disk-saved artifacts.

---

# Part 1 — AG News Classification (49.5 / 50)

## Task 1 — Tokenization (10 / 10)

- BPE and WordPiece both trained on the training corpus only, vocab size 10,000.
- 5 example sentences correctly tokenized side by side; the comparison table makes the difference between BPE and WordPiece visually obvious.
- Written answer covers all three required points: the difference (frequency vs likelihood, `##` marker), when to prefer each (BPE for left-to-right generation, WordPiece for full-sentence understanding), and why both beat word-level (no UNKs, shared subword representations, smaller vocabulary, better rare-word embeddings).
- The concrete examples (`unbelievable`, `uncertainty`) drawn from real output are a nice touch.

## Task 2 — Word2Vec (9.5 / 10)

- All 6 model variants trained (CBOW + Skip-gram × 50/100/200 dimensions).
- 3 example tokens with embeddings and most-similar words shown for each model.
- Comparison answers all four required questions: model choice (Skip-gram), why (semantic siblings vs co-occurrence neighbors using the `sport` example), embedding dimension chosen (100), and vocab-size difference explained correctly via `min_count=2`.
- The "Note on fragments in neighbors" explaining BPE subpieces is excellent attention to detail.

**Minor:** the `_GensimNoiseFilter` class is over-engineered — `warnings.filterwarnings("ignore")` would do the same in one line and not distract from the actual task.

## Task 3 — MLP (10 / 10)

- Mean pooling correctly implemented to produce fixed-size sentence vectors.
- MLP architecture (100 → 256 → 128 → 4) with ReLU and dropout — well within the 1–3 hidden layers requirement.
- Validation accuracy logged per epoch (improvement from 88.5% → 90.0%) — gives clear visibility into overfitting that pure train-loss doesn't.
- Design choices table covers all required items.
- MLP-vs-RNN limitation explained correctly with a concrete example ("Oil prices rise" vs "Rise in oil prices" → same vector).

## Task 4 — Evaluation (20 / 20)

- Test accuracy 90.03% with full confusion matrix.
- 3 misclassified examples shown with original text, true label, predicted label.
- Pattern analysis is the strongest part of the submission: correctly identifies Business ↔ Sci/Tech (201 errors) as the dominant failure mode, traces it to mean pooling losing token-level signal, and notes that one of the cited errors (Charles Schwab) is reasonable even for a human grader.
- LIME suggested as the attribution method, explained well and connected back to the specific MLP architecture.

---

# Part 2 — Char-RNN Dinosaur Names (50 / 50)

## Task 1 — DinosDataset (10 / 10)

- Accepts both file path and list of names.
- Start `<` and end `>` tokens correctly added; full text concatenated.
- All required attributes present: `vocab`, `token_to_id`, `id_to_token`.
- Sliding window sequences with proper `(x, y)` shifted pair.
- Train/val split done via HuggingFace `Dataset.train_test_split()` as required.
- **Shared vocabulary across train/val** — small but important design choice that prevents token-id mismatch between splits.
- DataLoaders use `drop_last=True` to keep batch sizes consistent for the LSTM hidden state.

## Task 2 — One-hot encoding (5 / 5)

Correct and concise: `np.eye(vocab_size)[array]` produces the right shape `(*input_shape, vocab_size)` in one line. Verified against all properties (sums to 1, values are 0/1, IDs match argmax).

## Task 3 — CharRNN forward pass (5 / 5)

Forward pass traced step by step through one-hot → LSTM → dropout → reshape → FC → softmax, with shapes printed at every step. Confirms output sums to 1 and shows per-character probabilities ≈ 1/28 (random) — a useful sanity check before training.

## Task 4 — Training (10 / 10)

- Model initialized and trained for 10 epochs, loss decreased from ~2.88 → ~1.45 (train) and ~2.86 → ~1.64 (val) — small train/val gap indicates no severe overfitting.
- **Bugs in the provided `train` function were correctly identified and fixed:**
  - `inputs.to(device)` → `inputs = inputs.to(device)` (`.to()` is not in-place)
  - `targets.view(batch_size*n_steps)` → `targets.view(-1)` (handles variable batch sizes)
  - Hidden state reinitialized per batch (because `shuffle=True`, batches aren't sequential — carrying state across shuffled batches would feed unrelated context)

## Task 5 — Name generation (5 / 5)

Multiple prefixes tested (`<`, `<t`, `<s`, `<a`, `<r`). Generated names look plausibly dinosaur-like — `-saurus`, `-raptor`, `-aurus` suffixes appear naturally, and prefix conditioning works correctly. A mix of short (`<tor>`) and long (`<anatipanocephales>`) names matches the natural variety in the training data.

## Task 6 — Sampling strategies (15 / 15)

- Both `top_k` and `temperature` implemented in `predict` and `sample`.
- Six strategies compared with 8 samples each, including a tested intermediate temperature (0.7).
- Written insights are grounded in the actual generated output: identifies temp=0.5 as too repetitive (everything ends in `-saurus`), temp=2.0 as too noisy (`<angongosthus>`, `<mentosauris>`), and temp=0.7 + top_k=5 as the empirical sweet spot — supported by concrete name examples (`<megalopelta>`, `<archaeosaurus>`, `<alimimus>`).

---

# Suggestions for future work

1. **Cell 13** — replace the custom `_GensimNoiseFilter` class with `warnings.filterwarnings("ignore")`. Same effect, ~20 fewer lines.
2. **Cell 14** — the original assignment question bullets are still above the answers, separated by `---`. Removing them would tighten the section.
3. **Part 2 Task 4** — adding a quick generation example *during* training (every N epochs) would make it easier to see qualitatively how the model is learning, complementing the loss curve.

None of these affect the grade — just polish for next time.
