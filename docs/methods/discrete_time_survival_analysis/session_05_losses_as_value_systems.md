# Session 5 — Losses as value systems

Loss choice encodes what you consider a "good" survival model.

Reference code: `src/ehrsequencing/models/losses.py`, `examples/survival_analysis/train_behrt_survival.py`

---

## Learning goals

- Contrast NLL, ranking, and hybrid objectives.
- Understand failure modes of ranking loss.
- Tune `lambda_rank` pragmatically.

---

## 1) NLL (`DiscreteTimeSurvivalLoss`)

Belief: good model assigns correct probability mass to survival/event process.

Strengths:

- calibrated hazards
- principled likelihood-based objective

Weaknesses:

- may not maximize rank discrimination (C-index) directly

---

## 2) Pairwise ranking (`PairwiseRankingLoss`)

Belief: good model correctly orders earlier-event patients above later-event patients.

Strengths:

- directly aligns with discrimination metrics

Weaknesses:

- can be unstable with few comparable pairs
- can over-focus ranking and hurt calibration

### When ranking misbehaves

- small batch sizes
- heavy censoring
- narrow event-time spread

All three reduce effective comparable pairs and increase noisy gradients.

---

## 3) Hybrid (`HybridSurvivalLoss`)

`total = lambda_nll * NLL + lambda_rank * Ranking`

Belief: you want both calibrated probabilities and strong ordering.

This is usually the best default in applied settings.

---

## 4) Tuning `lambda_rank` (practical recipe)

Start with:

- `lambda_nll = 1.0`
- `lambda_rank = 0.05` to `0.1`

Then adjust by behavior:

- C-index low, calibration OK -> increase `lambda_rank` gradually
- calibration degrades / hazard curves noisy -> decrease `lambda_rank`

Recommended search:

- `{0.01, 0.05, 0.1, 0.2}`
- keep batch size and margin fixed while scanning

Track at least:

- C-index (discrimination)
- calibration proxy (e.g., Brier or risk curve sanity)

---

## 5) Takeaway

Losses are not interchangeable. Pick the one that matches your deployment value: ranking, calibration, or a controlled compromise.
