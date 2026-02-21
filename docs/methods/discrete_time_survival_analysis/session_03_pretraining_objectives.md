# Session 3 — BEHRT pretraining objectives

This session clarifies what MLM and Next-Visit Prediction (NVP) each teach the encoder.

Reference code: `src/ehrsequencing/models/behrt.py`

---

## Learning goals

- Understand what MLM optimizes in EHR sequences.
- Understand why NVP is multi-label.
- Understand objective interaction (separate vs joint training).

---

## 1) MLM objective (`BEHRTForMLM`)

MLM predicts masked code tokens from bidirectional context.

What it tends to learn:

- local co-occurrence and substitutability among medical codes
- cross-visit context dependencies
- robust token representations for sparse coding patterns

Loss: cross-entropy over vocabulary with `ignore_index=-100`.

---

## 2) Next-Visit objective (`BEHRTForNextVisitPrediction`)

NVP predicts the set of codes likely to appear in the next visit.

Why multi-label:

- a visit can contain multiple diagnosis/procedure/medication codes
- this is set prediction, not single-class classification

Loss: `BCEWithLogitsLoss` on multi-hot labels.

What it tends to learn:

- progression patterns and forward clinical trajectory
- patient-level risk context useful for downstream time-to-event tasks

---

## 3) Do MLM + NVP get enforced together?

Short answer in this repo: **not by default in one built-in joint class**.

Current implementation provides separate heads/classes:

- `BEHRTForMLM`
- `BEHRTForNextVisitPrediction`

So embeddings are shaped by both objectives **only if** your training loop explicitly combines them (e.g., alternating batches or weighted multi-task sum).

If you train only MLM checkpoints, downstream survival starts from MLM-shaped embeddings.

---

## 4) Recommended strategy

- Start with MLM pretraining (stable and broadly useful).
- If you have reliable longitudinal visit transitions, add NVP multi-task pretraining.
- Track downstream survival metrics (C-index + calibration) to verify transfer benefit.

---

## 5) Takeaway

MLM builds rich contextual code semantics; NVP adds trajectory sensitivity. Joint training can be powerful, but in this codebase it must be orchestrated intentionally in training scripts.
