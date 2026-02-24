# Survival Head and Visit Aggregation

**Topics:** Token-to-visit pooling, scatter_add mechanics, hazard head, discrete-time interpretation

**Reference code:** `src/ehrsequencing/models/behrt_survival.py`

---

## Table of Contents

1. [Why Aggregate Tokens into Visits?](#why-aggregate-tokens-into-visits)
2. [scatter_add Aggregation](#scatter_add-aggregation)
3. [Hazard Head](#hazard-head)
4. [What "Hazard per Visit" Means](#what-hazard-per-visit-means)
5. [Visit Mask and Padding](#visit-mask-and-padding)
6. [Complete Forward Pass](#complete-forward-pass)

---

## Why Aggregate Tokens into Visits?

The BEHRT encoder outputs one contextualized vector per token: `(B, L, H)`.

But the survival target is event timing over **visit intervals** — `event_time` is a visit index, not a token index. To align model output with labels, you need:

```
token embeddings (B, L, H)  →  visit embeddings (B, V, H)  →  hazards (B, V)
```

Without aggregation, hazard semantics are misaligned with labels. A model predicting hazard per token would need to know which token within a visit to use — an arbitrary and unstable choice.

---

## scatter_add Aggregation

`aggregate_visits` implements vectorized mean pooling of token embeddings into visit bins:

```python
def aggregate_visits(self, code_embeddings, visit_ids, attention_mask):
    batch_size, seq_len, hidden_dim = code_embeddings.shape
    max_visit_id = visit_ids.max().item() + 1

    visit_embeddings = torch.zeros(batch_size, max_visit_id, hidden_dim, ...)
    visit_counts     = torch.zeros(batch_size, max_visit_id, ...)

    # Step 1: zero out padding tokens
    valid_mask = attention_mask.bool()
    masked_embeddings = code_embeddings * valid_mask.unsqueeze(-1)

    # Step 2: sum embeddings per visit
    visit_ids_expanded = visit_ids.unsqueeze(-1).expand(-1, -1, hidden_dim)
    visit_embeddings.scatter_add_(dim=1, index=visit_ids_expanded, src=masked_embeddings)

    # Step 3: count real tokens per visit
    visit_counts.scatter_add_(dim=1, index=visit_ids, src=valid_mask.float())

    # Step 4: mean pooling (avoid div by zero)
    visit_mask = (visit_counts > 0).float()
    visit_counts = torch.clamp(visit_counts, min=1.0)
    visit_embeddings = visit_embeddings / visit_counts.unsqueeze(-1)

    return visit_embeddings, visit_mask
```

**Geometrically:** Each visit embedding is the centroid of its token vectors in hidden space — the mean of all contextualized code representations within that visit.

**Complexity:** O(B·L) — linear in sequence length. The previous loop-based implementation was O(B·V·L).

**Key invariant:** Padding tokens are zeroed before `scatter_add_`. If this step is skipped, padding codes leak into visit 0 and corrupt its representation.

---

## Hazard Head

The hazard head maps each visit embedding to a scalar hazard probability:

```python
self.hazard_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.LayerNorm(hidden_dim // 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, 1),
    nn.Sigmoid()   # output ∈ (0, 1)
)

# Applied per visit
hazards = self.hazard_head(visit_embeddings).squeeze(-1)  # (B, V)
```

The sigmoid ensures hazards are in `(0, 1)`, consistent with the probabilistic interpretation $h_t = P(T = t \mid T \geq t)$.

---

## What "Hazard per Visit" Means

Time is discretized by **encounter index**, not wall-clock days.

```
Visit 0  →  h_0 = 0.05   (5% chance event at this visit)
Visit 1  →  h_1 = 0.08
Visit 2  →  h_2 = 0.15
...
```

**Implications:**
- Hazard step size is irregular in real time if visit spacing varies
- Interpretation is "risk at this observed visit step"
- Event and censoring labels must be aligned to visit index (not calendar time)

**If you need day-level interpretability:** Add explicit elapsed-time features (time-delta encoding) or move to calendar-time bins. See the discussion in `dev/methods/behrt_survival_analysis/` for future directions.

---

## Visit Mask and Padding

Not all patients have the same number of visits. `visit_mask` tracks which visit slots are real:

```python
visit_mask = (visit_counts > 0).float()   # (B, V): 1 for real visits, 0 for padding
hazards = hazards * visit_mask             # zero out hazards for padded visits
```

This mask is also passed to the loss function so padded visit slots do not contribute to the NLL computation.

---

## Complete Forward Pass

```python
def forward(self, codes, ages, visit_ids, attention_mask):
    # 1. BEHRT encoding: contextualize all tokens
    code_embeddings = self.behrt(
        codes=codes,
        ages=ages,
        visit_ids=visit_ids,
        attention_mask=attention_mask
    )  # (B, L, H)

    # 2. Aggregate tokens → visits
    visit_embeddings, visit_mask = self.aggregate_visits(
        code_embeddings, visit_ids, attention_mask
    )  # (B, V, H), (B, V)

    # 3. Predict hazard per visit
    hazards = self.hazard_head(visit_embeddings).squeeze(-1)  # (B, V)
    hazards = hazards * visit_mask                             # mask padding

    return hazards
```

**Core survival abstraction:**
- Contextualize at **token granularity** (full bidirectional attention)
- Decide at **visit granularity** (hazard prediction)

`scatter_add` is the key operation that bridges these two levels efficiently.

---

**See also:** `01a_visit_embeddings.md` for a deep treatment of the visit ID vs aggregated visit embedding distinction.

**Next:** `06_loss_functions.md` — how loss choice encodes what you consider a "good" survival model.
