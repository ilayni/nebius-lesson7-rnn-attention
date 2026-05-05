# Part 2 — Task 6: Sampling Strategies

## Goal

Extend `predict` and `sample` with two sampling controls — top-k and temperature — and compare how they affect the quality and variety of generated names.

## What was implemented

Both `predict` and `sample` now accept `top_k` and `temperature` parameters.

**Temperature** — divides the logits before softmax:
```python
p = F.softmax(out / temperature, dim=1)
```
- Low temperature (< 1.0): sharpens the distribution → model is more confident, picks higher-probability characters
- High temperature (> 1.0): flattens the distribution → rare characters get more chances, more creative output

**Top-k** — after computing probabilities, restricts sampling to the k most probable characters:
```python
top_k_idx = np.argsort(p)[-top_k:]
char = np.random.choice(top_k_idx, p=top_k_p / top_k_p.sum())
```
Eliminates very unlikely characters that would break the name's coherence.

## Generated names (seed=42, prefix=`<`, 8 samples per strategy)

**Random sampling (top_k=None, temp=1.0)**
`<gurus>`, `<aurus>`, `<sanchingopelta>`, `<liaomaus>`, `<eepichasaurus>`, `<yutrosaurus>`, `<scenatius>`, `<aspondylus>`

**Top-k=5 (temp=1.0)**
`<mosaurus>`, `<archaeiocephale>`, `<cratops>`, `<aurus>`, `<cephalisaurus>`, `<saurus>`, `<condochilengia>`, `<changyenglenglong>`

**Top-k=3 (temp=1.0)**
`<arconia>`, `<saurus>`, `<anglangosaurus>`, `<saraptor>`, `<stegosaurus>`, `<parasaurus>`, `<alamos>`, `<archaeosaurus>`

**Top-k=5, temp=0.5 (low)**
`<magulosaurus>`, `<anthosaurus>`, `<pelta>`, `<mingasaurus>`, `<palaeosaurus>`, `<albalosaurus>`, `<sinodon>`, `<antosaurus>`

**Top-k=5, temp=0.7 (mid)**
`<megalopelta>`, `<parasaurus>`, `<muchalodon>`, `<archaeosaurus>`, `<anasaurus>`, `<alimimus>`, `<pachysaurus>`, `<saurus>`

**Top-k=5, temp=2.0 (high)**
`<mursausurus>`, `<antops>`, `<sanghiuaceratops>`, `<amagaces>`, `<alenosausaurus>`, `<coloceratops>`, `<angongosthus>`, `<mentosauris>`

## Insights

**Random vs top-k**
Random sampling draws from the full vocabulary, so even very unlikely characters can occasionally be picked — producing odd combinations like `<gurus>` or `<eepichasaurus>`. Top-k restricts choice to the k most probable characters. With k=5 names are more coherent; with k=3 they are even cleaner but with less variety. k=1 degenerates to greedy search — always picking the most likely character — producing repetitive names.

**Temperature**
- **temp=0.5**: sharpens the distribution, model is decisive. Names are clean and dino-like (`<palaeosaurus>`, `<antosaurus>`) but can feel repetitive — all ending in `-saurus`
- **temp=0.7**: sweet spot. Coherent names with real variety — `<megalopelta>`, `<archaeosaurus>`, `<alimimus>`, `<pachysaurus>`. Mixes multiple suffix patterns naturally
- **temp=1.0**: baseline. Reasonable but occasionally odd (`<changyenglenglong>`, `<gurus>`)
- **temp=2.0**: too noisy. Names like `<angongosthus>` and `<mentosauris>` start to look garbled

**Best combination**
**Top-k=5 with temperature=0.7** gives the best balance — avoids unlikely characters (top-k) while maintaining enough variation to produce diverse, convincing dinosaur names.
