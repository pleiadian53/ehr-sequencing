# EHR Data → Tokens → Tensors

**Topics:** Flattening hierarchical EHR data, dual role of `visit_ids`, padding discipline, tensor contract

**Reference code:** `src/ehrsequencing/data/behrt_survival_dataset.py`

---

## Table of Contents

1. [The Transformation Problem](#the-transformation-problem)
2. [Hierarchical Data vs Flat Transformer Input](#hierarchical-data-vs-flat-transformer-input)
3. [The Dual Role of `visit_ids`](#the-dual-role-of-visit_ids)
4. [Tensor Contract](#tensor-contract)
5. [Padding Discipline](#padding-discipline)
6. [Hierarchical vs Flat Architecture Tradeoff](#hierarchical-vs-flat-architecture-tradeoff)
7. [Practical Checks Before Training](#practical-checks-before-training)
8. [Code Walkthrough](#code-walkthrough)

---

## The Transformation Problem

EHR data is naturally hierarchical:

```
Patient
  └── Visit 1 (age 62)
        ├── Code: 250.00 (Type 2 Diabetes)
        ├── Code: 401.9  (Hypertension)
        └── Code: E11.9  (Diabetes, unspecified)
  └── Visit 2 (age 62)
        ├── Code: 250.00
        └── Code: 428.0  (Heart Failure)
  └── Visit 3 (age 63)
        └── Code: 428.0
```

BEHRT expects flat token sequences: `[batch, seq_len]`. The challenge is to flatten this hierarchy without losing the visit structure needed for hazard prediction.

---

## Hierarchical Data vs Flat Transformer Input

`BEHRTSurvivalDataset.__getitem__` performs the flattening:

```python
# Hierarchical input
visits = [
    {'codes': [250, 401, 500], 'age': 62},   # Visit 0
    {'codes': [250, 428],      'age': 62},   # Visit 1
    {'codes': [428],           'age': 63},   # Visit 2
]

# Flat output tensors (seq_len = 6)
codes        = [250, 401, 500, 250, 428, 428]
ages         = [ 62,  62,  62,  62,  62,  63]
visit_ids    = [  0,   0,   0,   1,   1,   2]
attention_mask = [1,   1,   1,   1,   1,   1]
```

Visit structure is preserved implicitly through `visit_ids`. The transformer sees a flat stream; visit boundaries are recovered later during aggregation.

**Why flatten?**
- BEHRT encoder expects `[batch, seq_len]` token-aligned tensors
- Enables full cross-visit attention — code at visit 3 can attend to code at visit 1
- Compatible with BEHRT pre-training (pretrained weights transfer directly)
- Simpler batching than hierarchical approaches

---

## The Dual Role of `visit_ids`

`visit_ids` is one tensor with two distinct jobs:

**1. Input-side temporal feature**
- Passed into `BEHRTEmbedding.visit_embedding(visit_ids)`
- Lets each token's representation know which clinical encounter it belongs to
- Contributes to the embedding sum: `code + age + visit + position`

**2. Output-side grouping key**
- Reused in `BEHRTForSurvival.aggregate_visits(code_embeddings, visit_ids, ...)`
- Groups contextualized token embeddings back into visit-level representations via `scatter_add_`

This is elegant but load-bearing: **if `visit_ids` is corrupted, both representation learning and aggregation fail together** — often silently.

---

## Tensor Contract

For each batch item, these four tensors must remain perfectly aligned by index position:

| Tensor | Shape | Meaning at position `t` |
|--------|-------|------------------------|
| `codes` | `(B, L)` | Medical code ID |
| `ages` | `(B, L)` | Patient age at this code |
| `visit_ids` | `(B, L)` | Which visit this code belongs to |
| `attention_mask` | `(B, L)` | 1 = real code, 0 = padding |

If any tensor is shifted or truncated differently from the others, model quality degrades with no immediate crash.

**Truncation rule:** Always truncate all tensors together, keeping the most recent visits (right-truncation preserves recency).

---

## Padding Discipline

Sequences shorter than `max_seq_length` are right-padded:

```python
padding_len = max_seq_length - seq_len
codes         += [pad_token] * padding_len   # pad_token = 0
ages          += [0]         * padding_len
visit_ids     += [0]         * padding_len   # padded visit_ids = 0
attention_mask += [0]        * padding_len
```

**Why this is safe:**
- In aggregation: padded positions are zeroed by `attention_mask` before `scatter_add_`
- In transformer attention: `src_key_padding_mask = ~attention_mask.bool()` blocks padded tokens from being attended to

**Critical rule:** Never infer token validity from `visit_ids == 0`. Visit 0 is a real visit. Always use `attention_mask` as the sole source of truth for valid tokens.

### The Silent Killer

Padding bugs do not crash training. They silently corrupt gradients:

```python
# Bug: forgot to mask before aggregation
visit_emb = scatter_add(code_embeddings, visit_ids)  # ❌ padding leaks into visit 0

# Correct: mask first
masked = code_embeddings * attention_mask.unsqueeze(-1)
visit_emb = scatter_add(masked, visit_ids)            # ✅
```

---

## Hierarchical vs Flat Architecture Tradeoff

An alternative to flattening is a **hierarchical transformer**:

```
Visit 1 codes → [Code Encoder] → visit_1_repr
Visit 2 codes → [Code Encoder] → visit_2_repr   (shared weights)
...
[visit_1_repr, visit_2_repr, ...] → [Visit Encoder] → hazards
```

| Aspect | Flat (current) | Hierarchical |
|--------|---------------|--------------|
| Cross-visit attention | Full (all tokens attend to all) | None (visits are isolated in code encoder) |
| Pre-training compatibility | ✅ Direct BEHRT transfer | ❌ Requires separate pretraining |
| Sequence length scaling | O(L²) attention over all tokens | O(V²) over visits + O(C²) per visit |
| Batching complexity | Simple | Ragged tensors or double padding |
| Mask discipline | Load-bearing | Structurally isolated per visit |

**Current choice:** Flat. Pragmatic for the current data scale (7–10 visits avg, moderate codes/visit). Hierarchical becomes worth considering when visits are long (>20 codes) and sequences are long (>30 visits).

---

## Practical Checks Before Training

```python
# 1. Confirm non-pad token count
assert (attention_mask.sum(dim=1) == expected_token_counts).all()

# 2. Confirm visit_ids are non-decreasing within each sequence
for b in range(batch_size):
    valid = attention_mask[b].bool()
    ids = visit_ids[b][valid]
    assert (ids[1:] >= ids[:-1]).all(), "visit_ids not monotone"

# 3. Confirm all tensors have same shape
assert codes.shape == ages.shape == visit_ids.shape == attention_mask.shape
```

---

## Code Walkthrough

| Step | Location |
|------|----------|
| Flattening + label creation | `data/behrt_survival_dataset.py` → `__getitem__` |
| Embedding ingestion | `models/embeddings.py` → `BEHRTEmbedding.forward` |
| Transformer masking | `models/behrt.py` → `BEHRT.forward` |
| Visit aggregation | `models/behrt_survival.py` → `aggregate_visits` |

**Key takeaway:** Flatten early for transformer efficiency. Reconstruct visits late for hazard prediction. `visit_ids` and `attention_mask` are the invariants that make this possible.

---

**See also:** `01b_ehr_tokens_tensors.md` for a deeper treatment of the flattening operation and masking implementation details.

**Next:** `03_behrt_embeddings.md` — how BEHRT embeddings encode structured inductive bias.
