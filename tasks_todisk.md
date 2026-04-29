# Save / Load Map

| Cell | Task heading | Saves | Loads from disk |
|------|-------------|-------|-----------------|
| **4** | Load the Data | `train_df.parquet`, `test_df.parquet` | loads from parquet on re-run |
| **8** | Task 1 — Tokenization (BPE or WordPiece) | `bpe_tokenizer.json`, `wp_tokenizer.json` | `train_df` if not in memory |
| **13** | Task 2 — Train Word Embeddings (Word2Vec) | `w2v_CBOW_50/100/200.model`, `w2v_SkipGram_50/100/200.model` | `bpe_tokenizer` if not in memory |
| **17** | Task 3 — MLP Architecture and Training | `mlp_model.pt`, `X_train/test.npy`, `y_train/test.npy` | `bpe_tokenizer`, `models` if not in memory |
| **21** | Task 4 — Evaluation and Error Analysis | — | `mlp_model.pt`, `X_test.npy`, `y_test.npy` |
| Part 2 TBD | Part 2 Tasks 1–4 — DinosDataset + CharRNN | `dinos_vocab.json`, `charrnn.pt` | Added when implemented |
| Part 2 TBD | Part 2 Tasks 5–6 — Generation + Sampling | — | `charrnn.pt`, `dinos_vocab.json` |

## Run-once cells (train and save)
Run these once in order, then never again:
1. Cell 4 — Load Data (downloads AG News, saves parquet)
2. Cell 8 — Task 1 Tokenization
3. Cell 13 — Task 2 Word2Vec
4. Cell 17 — Task 3 MLP

## Independent cells (load from disk)
Re-run any time after a kernel restart without running anything above:
- Cell 21 — Task 4 Evaluation
- All Part 2 cells once implemented
