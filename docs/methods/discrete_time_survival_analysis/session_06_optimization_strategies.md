# Session 6 — Optimization strategy: frozen vs LoRA vs full fine-tune

This session gives a practical decision framework for adapting BEHRT to survival.

Reference code: `src/ehrsequencing/models/behrt_survival.py`, `src/ehrsequencing/models/lora.py`, `examples/survival_analysis/train_behrt_survival.py`

---

## Learning goals

- Compare frozen, LoRA, and full fine-tuning.
- Understand overfitting/compute tradeoffs.
- Apply a training recipe by cohort size.

---

## 1) Strategy options

### A) Frozen encoder (`freeze_behrt=True`)

- Train only survival head.
- Fastest and most stable on tiny datasets.
- Lowest adaptation capacity.

Use when:

- very small cohorts
- fast baselines and ablations

### B) LoRA (`use_lora=True`)

- Freeze base weights; train low-rank adapters (+ optionally embeddings/head).
- Strong efficiency/performance tradeoff.
- Usually best default for practical transfer.

Use when:

- medium data scale
- limited GPU memory
- need better adaptation than frozen

### C) Full fine-tuning

- Update all BEHRT parameters.
- Highest capacity, highest compute, highest overfit risk.

Use when:

- large cohorts + strong regularization
- enough compute and tuning budget

---

## 2) Parameter tradeoff intuition

Order of trainable parameter count:

`Frozen << LoRA << Full`

As trainable count rises:

- adaptation flexibility increases
- compute/memory increases
- overfitting risk increases

---

## 3) Practical recipe by data regime

### Small cohort (e.g., < 2k patients)

- Start frozen + hybrid loss (`lambda_rank=0.05`)
- If underfitting, move to LoRA rank 8–16
- Strong early stopping

### Medium cohort (2k–20k)

- Start with LoRA rank 16
- Hybrid loss (`lambda_rank=0.1` as initial)
- Gradient clipping + weight decay

### Large cohort (> 20k)

- Compare LoRA vs full fine-tune
- Consider full fine-tune if LoRA saturates
- Use robust validation and checkpoint selection by C-index + calibration

---

## 4) Operational notes

- Use `model.get_trainable_parameters()` to verify actual training footprint.
- Keep truncation/masking consistent across train/val/test.
- Never compare strategies with different preprocessing pipelines.

---

## 5) Takeaway

Default recommendation for this project: **LoRA-first**, then escalate to full fine-tune only when data scale and validation evidence justify it.
