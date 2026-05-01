# Feedback — Tasks 1–4

Overall this is a strong submission. The pipeline is complete, the code runs cleanly, and the written answers show genuine understanding. Below are the points worth addressing.

---

## Task 1 — Tokenization ✅

Code and answers are both solid. The comparison table is clear, the examples are concrete, and the BPE vs WordPiece explanation is accurate. Nothing meaningful to change here.

---

## Task 2 — Word2Vec

**Code — `_GensimNoiseFilter` class**

This 20-line stderr filter is overkill for an assignment. It works, but it buries the actual task logic. A one-liner does the same job:

```python
import warnings
warnings.filterwarnings("ignore")
```

Remove the filter class and use that instead.

**Written answers — structure of cell 14**

The original question bullet points (`* Which model performed better for this task`, etc.) were left in the answer cell above the `---` separator. Remove them — your answers stand on their own and the prompts add clutter.

**Written answers — content**

The Skip-gram vs CBOW comparison using the `sport` neighbors is a good concrete argument. The vocabulary size explanation (min_count=2 drops low-frequency tokens) is correct and well-articulated.

---

## Task 3 — MLP

**Code — redundant import at the bottom**

```python
import torch, numpy as np_
```

`numpy` is already imported as `np` at the top of the cell. Re-importing it as `np_` is confusing. Just use `np` throughout:

```python
np.save("saved_models/X_train.npy", X_train)
```

**Code — no validation accuracy during training**

The training loop only logs training loss. It gives no signal on whether the model is overfitting. Even a simple accuracy check on the test loader at the end of each epoch would make this much more informative.

**Written answers**

The design choices table is clear and well-justified. The MLP-vs-RNN limitation explanation is accurate ("Rise in oil prices" and "Oil prices rise" produce the same vector — good example).

---

## Task 4 — Evaluation and Error Analysis ✅

Code is correct, the confusion matrix is readable, and the 3 error examples are clearly displayed. The written analysis is the strongest part of the submission:

- The Business ↔ Sci/Tech confusion is correctly traced back to mean pooling losing token-level distinction.
- The LIME explanation is accurate and connects well to the specific model architecture (removing a token changes the mean-pooled vector, so attribution is natural).

One minor note: the 3 displayed errors are simply the first 3 misclassified examples in order — not necessarily the most representative ones. Worth mentioning in a real analysis that you'd want to sample across error types.

---

## Spelling

One typo in the assignment itself (not yours): the requirement says "confusion **metrix**" — you correctly wrote "confusion matrix" in your answer.

No spelling errors found in your written answers.
