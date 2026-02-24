# Loss Functions for Survival Analysis

**Topics:** NLL vs C-index disconnect, pairwise ranking loss, hybrid loss, tuning `lambda_rank`

**Reference code:** `src/ehrsequencing/models/losses.py`, `examples/survival_analysis/train_behrt_survival.py`

---

## Table of Contents

1. [The Core Tension: Calibration vs Discrimination](#the-core-tension-calibration-vs-discrimination)
2. [Negative Log-Likelihood Loss](#negative-log-likelihood-loss)
3. [Why NLL Does Not Directly Optimize C-index](#why-nll-does-not-directly-optimize-c-index)
4. [Pairwise Ranking Loss](#pairwise-ranking-loss)
5. [Hybrid Loss](#hybrid-loss)
6. [Tuning lambda_rank](#tuning-lambda_rank)
7. [Practical Recommendations](#practical-recommendations)

---

## The Core Tension: Calibration vs Discrimination

Loss choice encodes what you consider a "good" survival model.

| Property | NLL | C-index |
|----------|-----|---------|
| Measures | Calibration (probability accuracy) | Discrimination (ranking quality) |
| Cares about | Absolute predicted probabilities | Relative ordering of predictions |
| Differentiable | Yes (smooth gradients) | No (discrete counting) |
| Optimized by | `DiscreteTimeSurvivalLoss` | `PairwiseRankingLoss` (surrogate) |

A model can be well-calibrated but poorly discriminative, or vice versa. Neither property implies the other.

---

## Negative Log-Likelihood Loss

**Class:** `DiscreteTimeSurvivalLoss`

For discrete-time survival, the NLL is derived from maximum likelihood:

$$\mathcal{L}_{\text{NLL}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \delta_i \log h_{t_i} + (1-\delta_i) \log S(t_i) \right]$$

where:
- $\delta_i = 1$ if event observed, $0$ if censored
- $h_{t_i}$ is the predicted hazard at event/censoring time $t_i$
- $S(t_i) = \prod_{j=1}^{t_i} (1 - h_j)$ is the survival probability

**For event patients ($\delta_i = 1$):** Penalizes low hazard at the observed event time — pushes $h_{t_i} \to 1$.

**For censored patients ($\delta_i = 0$):** Penalizes low survival probability up to censoring — pushes $S(t_i) \to 1$, meaning all $h_j$ for $j \leq t_i$ should be small.

**Strengths:** Calibrated hazard probabilities, principled likelihood-based objective, stable gradients.

**Weakness:** May not maximize rank discrimination (C-index) directly.

---

## Why NLL Does Not Directly Optimize C-index

### The Disconnect

**Example: Good NLL, Poor C-index**

```
Patient A: Event at t=2  →  predicted risk 0.51
Patient B: Event at t=5  →  predicted risk 0.50
Patient C: Censored t=10 →  predicted risk 0.49
```

NLL is low (predictions are well-calibrated). But C-index is near 0.5 — the model barely distinguishes A from B.

**Example: Poor NLL, Good C-index**

```
Patient A: Event at t=2  →  predicted risk 0.90
Patient B: Event at t=5  →  predicted risk 0.60
Patient C: Censored t=10 →  predicted risk 0.20
```

C-index is high (correct ordering). But NLL may be poor if the absolute probabilities are miscalibrated.

### Why C-index Is Not Directly Differentiable

C-index involves:
1. Counting concordant pairs (discrete operation)
2. Comparing predictions with indicator functions $\mathbb{1}[r_i > r_j]$ (discontinuous)

The gradient is zero almost everywhere — direct optimization with gradient descent is impossible. Ranking losses use smooth surrogates instead.

---

## Pairwise Ranking Loss

**Class:** `PairwiseRankingLoss`

For each comparable pair $(i, j)$ where patient $i$ had an event before patient $j$, penalize incorrect ordering:

$$\mathcal{L}_{\text{rank}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \max(0,\, \text{margin} - (r_i - r_j))$$

where $\mathcal{P}$ is the set of comparable pairs and $r_i$ is the predicted risk score.

**Strengths:** Directly aligns with discrimination metrics; improves C-index.

**Weaknesses:**
- Can be unstable with few comparable pairs
- Can over-focus on ranking and hurt calibration
- Noisy gradients when event-time spread is narrow

**When ranking misbehaves:**
- Small batch sizes → few comparable pairs → noisy gradients
- Heavy censoring → most pairs are incomparable
- Narrow event-time spread → many ties, ambiguous ordering

---

## Hybrid Loss

**Class:** `HybridSurvivalLoss`

$$\mathcal{L}_{\text{hybrid}} = \lambda_{\text{NLL}} \cdot \mathcal{L}_{\text{NLL}} + \lambda_{\text{rank}} \cdot \mathcal{L}_{\text{rank}}$$

Belief: you want both calibrated probabilities and strong ordering.

This is the recommended default in applied settings — it balances the two objectives and is more robust than either alone.

---

## Tuning lambda_rank

**Starting point:**

```python
lambda_nll  = 1.0
lambda_rank = 0.05   # conservative start
```

**Adjustment rules:**

| Observation | Action |
|-------------|--------|
| C-index low, calibration OK | Increase `lambda_rank` gradually |
| Calibration degrades / hazard curves noisy | Decrease `lambda_rank` |
| Training unstable | Reduce `lambda_rank`, increase batch size |

**Recommended search grid:** `{0.01, 0.05, 0.1, 0.2}`

Keep batch size and margin fixed while scanning. Track both C-index (discrimination) and a calibration proxy (Brier score or risk curve sanity check).

---

## Practical Recommendations

| Goal | Loss | Notes |
|------|------|-------|
| Best calibration (Brier score) | NLL only | Stable, principled |
| Best discrimination (C-index) | Hybrid, high `lambda_rank` | Watch for calibration degradation |
| General-purpose starting point | Hybrid, `lambda_rank=0.05` | Recommended default |
| Small dataset / heavy censoring | NLL only | Ranking loss too noisy |
| Large dataset, strong labels | Hybrid, `lambda_rank=0.1–0.2` | More pairs → stable ranking gradients |

**Key principle:** Losses are not interchangeable. Pick the one that matches your deployment value — ranking, calibration, or a controlled compromise.

---

**See also:** `loss_functions_and_optimization.md` for a deeper mathematical treatment including the full NLL derivation, implementation details, and experimental comparisons.

**Next:** `07_optimization_strategies.md` — frozen encoder vs LoRA vs full fine-tuning.
