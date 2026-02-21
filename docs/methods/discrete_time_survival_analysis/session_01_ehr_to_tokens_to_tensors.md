# Session 1 — EHR → tokens → tensors

This session explains how hierarchical EHR data becomes BEHRT-ready tensors without losing visit structure.

## Learning goals

- Understand flattening (`patient -> visits -> codes`) vs hierarchical representation.
- Understand the **two roles** of `visit_ids`.
- Avoid silent bugs from padding/masking mismatch.

---

## 1) Hierarchical data vs flat transformer input

In this codebase, survival data is flattened in `BEHRTSurvivalDataset.__getitem__`:

- `codes`: all codes across visits, concatenated into one token stream
- `ages`: age repeated per code token
- `visit_ids`: visit index repeated per code token
- `attention_mask`: 1 for real tokens, 0 for padding

Reference: `src/ehrsequencing/data/behrt_survival_dataset.py`

Why flatten?

- BEHRT encoder expects `[batch, seq_len]` token-aligned tensors.
- You keep visit boundaries via `visit_ids`, then recover visit-level signals later.

---

## 2) The dual role of `visit_ids`

`visit_ids` is one tensor with two jobs:

1. **Input-side temporal feature**
   - Passed into `BEHRTEmbedding.visit_embedding(...)`
   - Lets token representation know which visit it belongs to

2. **Output-side grouping key**
   - Reused in `BEHRTForSurvival.aggregate_visits(...)`
   - Groups token embeddings back into visit embeddings via `scatter_add_`

That is elegant but easy to misuse: if `visit_ids` is corrupted, both representation learning and aggregation fail together.

---

## 3) Tensor contract (what must align)

For each batch item, these must remain perfectly aligned by index:

- `codes[t]`
- `ages[t]`
- `visit_ids[t]`
- `attention_mask[t]`

If one gets shifted/truncated differently, model quality degrades with no immediate crash.

---

## 4) Padding discipline: the silent killer

Current padding convention:

- `codes` pad token = `0`
- `attention_mask` pad = `0`
- padded `visit_ids` currently set to `0`

Why this is safe here:

- In aggregation, padded positions are zeroed by `attention_mask` before `scatter_add_`.
- In transformer attention, `src_key_padding_mask = ~attention_mask.bool()` blocks padded tokens.

But you still must enforce:

- Never infer validity from `visit_ids == 0`.
- Always use `attention_mask` as the source of truth for valid tokens.

---

## 5) Practical checks before training

- Confirm `attention_mask.sum(dim=1)` equals number of non-pad tokens.
- Confirm `visit_ids[attention_mask==1]` is nondecreasing within each patient sequence.
- Confirm truncation keeps all tensors synchronized.

---

## 6) Walkthrough: where this shows up in code

- Flattening + labels: `data/behrt_survival_dataset.py`
- Embedding ingestion: `models/embeddings.py` (`BEHRTEmbedding.forward`)
- Transformer masking: `models/behrt.py` (`BEHRT.forward`)
- Visit aggregation: `models/behrt_survival.py` (`aggregate_visits`)

---

## 7) Takeaway

The modeling trick is simple:

- **Flatten early for transformer efficiency**
- **Reconstruct visits late for hazard prediction**

`visit_ids` and `attention_mask` are the invariants that make this possible.
