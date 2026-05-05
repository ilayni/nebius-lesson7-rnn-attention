# Part 2 — Task 5: Generating Dinosaur Names

## Goal

Use the trained CharRNN model to generate new dinosaur names character by character, starting from a given prefix.

## How it works

Two functions work together:

**`predict(char, model, ...)`** — takes a single character, runs it through the model, and samples the next character from the output probability distribution.

**`sample(model, size, ..., prefix)`** — warms up the hidden state by feeding each character of the prefix through `predict`, then generates up to `size` more characters, stopping early if `>` (end token) is produced.

## Bug fixed

`inputs.to(device)` in `predict` was not assigned — same issue as in `train`. Fixed to `inputs = inputs.to(device)`.

## Generated names (seed=42)

| Prefix | Generated names |
|--------|----------------|
| `<` (random) | `<gurus>`, `<aurus>`, `<sanchingopelta>`, `<liaomaus>`, `<eepichasaurus>` |
| `<t` | `<totornis>`, `<telolimasaurus>`, `<tor>`, `<taurus>`, `<torosaurus>` |
| `<s` | `<sebachiosaurus>`, `<strosaurus>`, `<sarcouannia>`, `<saurus>`, `<shanasaurus>` |
| `<a` | `<aychurus>`, `<anatipanocephales>`, `<adrominosaurus>`, `<artosaurus>`, `<ar>` |
| `<r` | `<rinleraptor>`, `<raigewator>`, `<rickachitson>`, `<rixtongsaurus>`, `<rizberosaurus>` |

## Conclusions

- The model has clearly learned the structure of dinosaur names — suffixes like `saurus`, `raptor`, `aurus`, `opus` appear frequently, which matches real dinosaur naming conventions
- Prefix conditioning works: `<t` produces names starting with `t`, `<r` with `r`, etc. The hidden state warm-up effectively biases generation toward the chosen prefix
- Some names are very short (`<tor>`, `<ar>`) and some long (`<anatipanocephales>`), reflecting the natural variation in the training data
- Occasional odd combinations (`<rickachitson>`, `<raigewator>`) show the model hasn't fully memorized the data — it is genuinely generating new names rather than copying training examples
