# BEHRT Pretraining Objectives

**Topics:** Masked Language Modeling (MLM), Next-Visit Prediction (NVP), what each objective teaches, joint training

**Reference code:** `src/ehrsequencing/models/behrt.py`

---

## Table of Contents

1. [Why Pretrain?](#why-pretrain)
2. [MLM Objective](#mlm-objective)
3. [Next-Visit Prediction Objective](#next-visit-prediction-objective)
4. [Joint vs Sequential Training](#joint-vs-sequential-training)
5. [Recommended Strategy](#recommended-strategy)

---

## Why Pretrain?

BEHRT pretraining learns general-purpose EHR representations from unlabeled patient sequences before any survival labels are introduced. The benefit:

- Survival-labeled data is scarce and expensive to curate
- Unlabeled EHR sequences are abundant
- Pretrained embeddings encode medical code semantics and temporal patterns that transfer to downstream tasks

Without pretraining, the model must learn both representation and survival prediction simultaneously from a small labeled dataset — a harder optimization problem prone to overfitting.

---

## MLM Objective

**Class:** `BEHRTForMLM`

**Mechanism:** Randomly mask 15% of code tokens. Predict the original code from bidirectional context.

```python
# Masked positions get a special [MASK] token
# Loss: cross-entropy over vocabulary, ignoring non-masked positions
loss = CrossEntropyLoss(logits[masked_positions], labels[masked_positions],
                        ignore_index=-100)
```

**What it tends to learn:**
- Local co-occurrence and substitutability among medical codes (e.g., diabetes codes cluster together)
- Cross-visit context dependencies (e.g., heart failure codes predict diuretic codes)
- Robust representations for sparse coding patterns (rare codes get context from neighbors)

**What gets updated:** Code embeddings + age/visit/position embeddings + transformer weights + MLM head

**Intuition:** Forces the model to understand medical code semantics from context, analogous to how BERT learns word meaning from surrounding words.

---

## Next-Visit Prediction Objective

**Class:** `BEHRTForNextVisitPrediction`

**Mechanism:** From the patient's representation (pooled over all tokens), predict the set of codes likely to appear in the next visit.

**Why multi-label:** A visit contains multiple diagnosis, procedure, and medication codes simultaneously. This is set prediction, not single-class classification.

```python
# Multi-hot target: 1 for each code present in next visit
loss = BCEWithLogitsLoss(logits, target_multi_hot)
```

**What it tends to learn:**
- Disease progression patterns and forward clinical trajectory
- Comorbidity co-occurrence at the visit level
- Patient-level risk context useful for downstream time-to-event tasks

**Key difference from MLM:** MLM learns within-sequence code semantics; NVP learns forward-in-time trajectory patterns. NVP is more directly aligned with survival prediction.

---

## Joint vs Sequential Training

The codebase provides separate classes (`BEHRTForMLM`, `BEHRTForNextVisitPrediction`) rather than a single joint class.

**Implication:** Embeddings are shaped by both objectives **only if** your training loop explicitly combines them:

```python
# Option A: Alternating batches
for batch in train_loader:
    if batch_idx % 2 == 0:
        loss = mlm_loss(model, batch)
    else:
        loss = nvp_loss(model, batch)

# Option B: Weighted multi-task sum
loss = lambda_mlm * mlm_loss + lambda_nvp * nvp_loss
```

If you train only MLM checkpoints, downstream survival fine-tuning starts from MLM-shaped embeddings — which is a reasonable default.

---

## Recommended Strategy

| Scenario | Recommendation |
|----------|---------------|
| Starting from scratch | MLM pretraining first (stable, broadly useful) |
| Have reliable visit transitions | Add NVP multi-task pretraining |
| Limited compute | MLM only, then fine-tune survival head |
| Evaluating transfer benefit | Track C-index + calibration with/without NVP |

**MLM builds rich contextual code semantics; NVP adds trajectory sensitivity.** Joint training can be powerful, but in this codebase it must be orchestrated intentionally in training scripts.

---

**Next:** `05_survival_head_and_aggregation.md` — the bridge from token-level transformer output to visit-level hazards.
