# Hierarchical BEHRT: A Two-Level Architecture for EHR Survival

**Topics:** Hierarchical vs flat design, within-visit encoding, across-visit transformer, attention pooling, implementation blueprint

**Reference code:** `src/ehrsequencing/models/behrt_survival.py`, `src/ehrsequencing/models/behrt.py`

**Related docs:** `02_ehr_to_tokens.md`, `03_behrt_embeddings.md`, `08_architecture_decisions.md`

---

## Table of Contents

1. [The Core Tradeoff](#the-core-tradeoff)
2. [Notation and Tensor Shapes](#notation-and-tensor-shapes)
3. [Stage A: Within-Visit Encoder](#stage-a-within-visit-encoder)
4. [Stage B: Across-Visit Encoder](#stage-b-across-visit-encoder)
5. [Survival Head and Loss](#survival-head-and-loss)
6. [Why Hierarchy Helps: A Clinical Intuition](#why-hierarchy-helps-a-clinical-intuition)
7. [Flat vs Hierarchical: Side-by-Side](#flat-vs-hierarchical-side-by-side)
8. [Implementation Blueprint](#implementation-blueprint)
9. [Recommended Starting Point](#recommended-starting-point)

---

## The Core Tradeoff

The current BEHRT model is **flat**: all codes across all visits are concatenated into a single token sequence, and visit structure is recovered implicitly via `visit_id` embeddings and post-hoc `scatter_add` pooling.

A **hierarchical** model makes visit structure explicit at the architecture level:

```
Flat:        [code_1_v1, code_2_v1, code_1_v2, code_2_v2, ...]  →  Transformer  →  scatter_add  →  hazards
Hierarchical: [code_1_v1, code_2_v1]  →  VisitEncoder  →  visit_1_repr  ┐
              [code_1_v2, code_2_v2]  →  VisitEncoder  →  visit_2_repr  ┤  →  TimelineEncoder  →  hazards
              ...                                                         ┘
```

The model no longer needs to infer that certain tokens belong to the same visit — that structure is baked into the data pipeline and the two-stage encoder.

---

## Notation and Tensor Shapes

| Symbol | Meaning | Shape |
|--------|---------|-------|
| B | Batch size (patients) | — |
| V | Max visits per patient (padded) | — |
| C | Max codes per visit (padded) | — |
| d | Embedding / hidden dimension | — |
| **c** | Code IDs | (B, V, C) |
| **a** | Age at each visit | (B, V) |
| **m**^code | Code padding mask (1=real, 0=pad) | (B, V, C) |
| **m**^visit | Visit padding mask (1=real, 0=pad) | (B, V) |
| **u** | Visit embeddings (Stage A output) | (B, V, d) |
| **z** | Contextualized visit states (Stage B output) | (B, V, d) |
| **h** | Discrete-time hazard per visit | (B, V) |

The key structural difference from the flat model: inputs are shaped `(B, V, C)` instead of `(B, L)`. The two levels of padding — codes within visits, visits within patients — are tracked by two separate masks.

---

## Stage A: Within-Visit Encoder

Stage A compresses the `C` code vectors within each visit into a single visit embedding **u**_{i,v}.

### Embedding Layer

For each code token, sum the code embedding with an intra-visit positional embedding (and optionally a type embedding for dx/proc/rx/lab):

$$\mathbf{e}_{i,v,k} = \mathbf{E}^{\text{code}}[\mathbf{c}_{i,v,k}] + \mathbf{E}^{\text{pos(code)}}[k] + \mathbf{E}^{\text{type}}[\text{optional}]$$

This gives a tensor of shape `(B, V, C, d)`.

### Aggregation Options

Three options exist for compressing `C` code vectors into one visit vector, in increasing order of expressiveness:

#### Option A: Mean Pooling

$$\mathbf{u}_{i,v} = \frac{1}{\sum_k m^{\text{code}}_{i,v,k}} \sum_k m^{\text{code}}_{i,v,k} \cdot \mathbf{e}_{i,v,k}$$

Simple, fast, no extra parameters. All codes contribute equally. A reasonable baseline.

#### Option B: Attention Pooling (Learned Importance)

$$\alpha_{i,v,k} = \text{softmax}_k\!\left(\mathbf{q}^\top \tanh(\mathbf{W}\, \mathbf{e}_{i,v,k})\right)$$

$$\mathbf{u}_{i,v} = \sum_k \alpha_{i,v,k} \cdot \mathbf{e}_{i,v,k}$$

The result is a **weighted average** of code embeddings, where the weights are learned by a global query vector **q** (shape `d`, shared across all visits and patients). **q** acts as a content-based importance scorer: "how much does this code look like an important code?" Each code is scored independently against **q**, then the scores are normalized within the visit via softmax.

**Crucially, this is not transformer self-attention.** There is only one query — a single learned parameter — so there is no consolidation problem. Each code gets exactly one scalar weight, derived from its own embedding alone. No averaging of attention weights is needed.

```python
class AttnPoolingVisitEncoder(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Linear(d, d)
        self.q = nn.Parameter(torch.randn(d))

    def forward(self, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # e: (B, V, C, d),  mask: (B, V, C)
        scores = (self.q * torch.tanh(self.W(e))).sum(-1)  # (B, V, C)
        scores = scores.masked_fill(~mask.bool(), float('-inf'))

        # Guard: if an entire visit is padded, all scores are -inf → softmax → NaN.
        # Replace all-padding visits with uniform zero weights before softmax.
        all_pad = ~mask.bool().any(dim=-1, keepdim=True)   # (B, V, 1)
        scores = scores.masked_fill(all_pad.expand_as(scores), 0.0)

        alpha = torch.softmax(scores, dim=-1)               # (B, V, C)
        # Zero out weights for all-padding visits (softmax of zeros = uniform, not zero)
        alpha = alpha * mask.float()

        return (alpha.unsqueeze(-1) * e).sum(dim=2)         # (B, V, d)
```

> **Implementation note:** The guard above handles the edge case where a visit slot is entirely padding (all `mask` values false for that visit). Without it, `softmax([-inf, ..., -inf])` produces `NaN`, which propagates silently through the loss. In practice, the visit mask (`m^visit`) should prevent padded visit slots from contributing to the loss, but the NaN guard makes the forward pass numerically safe regardless.

#### Option C: Mini-Transformer per Visit

Reshape visits into a batch dimension, run a small transformer encoder with the code padding mask, then pool the output:

```python
# Reshape: (B, V, C, d) → (B*V, C, d)
e_flat = e.view(B * V, C, d)
mask_flat = mask.view(B * V, C)

out = self.code_transformer(e_flat, src_key_padding_mask=~mask_flat.bool())
# out: (B*V, C, d)

# Pool to visit vector — mean or CLS token
u_flat = (out * mask_flat.unsqueeze(-1)).sum(1) / mask_flat.sum(1, keepdim=True)
u = u_flat.view(B, V, d)  # (B, V, d)
```

This gives codes within a visit full bidirectional attention over each other — the most expressive option. The tradeoff is compute: you run a transformer `B*V` times per forward pass.

**Practical recommendation:** Option B (attention pooling) gives the best efficiency-to-expressiveness ratio and is the recommended starting point.

---

## Stage B: Across-Visit Encoder

Stage B treats the visit embeddings **u** as tokens in a second-level sequence model, learning how the patient's clinical trajectory evolves across visits.

### Visit-Level Embeddings

Before the across-visit transformer, add structural signals at the visit level:

$$\mathbf{x}_{i,v} = \mathbf{u}_{i,v} + \mathbf{E}^{\text{pos(visit)}}[v] + \mathbf{E}^{\text{age}}[\text{bin}(a_{i,v})] + \mathbf{E}^{\Delta t}[\text{bin}(\Delta t_{i,v})]$$

The $\Delta t$ (time-delta) embedding is optional but highly valuable for EHR data, where visit spacing is irregular. A gap of 3 days vs 3 years between visits carries very different clinical meaning, and this embedding lets the model learn that distinction explicitly.

### Transformer over Visits

```python
z = self.visit_transformer(x, src_key_padding_mask=~visit_mask.bool())
# z: (B, V, d)
```

The visit mask ensures padded visit slots do not contribute to attention. The output **z** is a sequence of contextualized visit representations — each visit now "knows" about all other visits in the patient's history.

---

## Survival Head and Loss

The survival head and loss are identical to the flat model:

$$h_{i,v} = \sigma(g(\mathbf{z}_{i,v}))$$

where $g$ is the MLP hazard head. The discrete-time NLL loss is:

$$\log L_i = \sum_{v < t_i} \log(1 - h_{i,v}) + \delta_i \log(h_{i,t_i})$$

Padded visits are masked out using **m**^visit. The existing `DiscreteTimeSurvivalLoss` in `losses.py` can be reused directly — `sequence_mask = visit_mask` maps naturally onto its interface.

---

## Why Hierarchy Helps: A Clinical Intuition

Consider a visit containing: `sepsis`, `vasopressor`, `intubation`.

A within-visit encoder can learn that these three codes co-occur as an "acute shock state" — a single dense vector capturing the clinical gestalt of the encounter. The across-visit encoder then learns transitions between these visit-level states:

```
stable chronic  →  mild deterioration  →  acute shock  →  ICU admission  →  death
```

A flat model *can* learn this, but it must use the same attention mechanism to simultaneously learn:
- code co-occurrence patterns within visits
- long-range temporal dynamics across visits

These are different inductive problems at different scales. Hierarchy gives each level the right scope.

---

## Flat vs Hierarchical: Side-by-Side

| Aspect | Flat (current) | Hierarchical |
|--------|---------------|--------------|
| Input shape | `(B, L)` where L = V·C | `(B, V, C)` |
| Attention scope | All tokens attend to all | Codes attend within visit; visits attend across timeline |
| Attention complexity | O((V·C)²) | O(V·C² + V²) |
| Visit structure | Implicit via `visit_id` | Explicit in architecture |
| Δt encoding | Token-level (approximate) | Visit-level (natural) |
| Pre-training transfer | Direct BEHRT transfer | Requires separate pretraining |
| Batching | Simple | Ragged tensors or double padding |
| Mask discipline | Single mask | Two masks (code + visit) |
| Scalability | Bottleneck at long sequences | Better for dense/long visits |

**Concrete scale example:** With V=50 visits and C=30 codes/visit, the flat sequence length is L=1500. Flat attention is O(1500²) = 2.25M operations per head. Hierarchical is O(50·30² + 50²) = 47,500 — roughly 47× cheaper.

---

## Implementation Blueprint

The hierarchical model decomposes cleanly into five modules:

**Module 1: `CodeEmbedding`**
- `nn.Embedding(n_codes, d, padding_idx=0)` — code identity
- `nn.Embedding(max_codes_per_visit, d)` — intra-visit position
- Optional: type embedding (dx / proc / rx / lab)

**Module 2: `VisitEncoder`** (choose one)
- `MeanPoolingVisitEncoder` — baseline
- `AttnPoolingVisitEncoder` — recommended
- `TransformerVisitEncoder` — most expressive

Input: `(B, V, C, d)` + code mask `(B, V, C)`
Output: `(B, V, d)`

**Module 3: `VisitTimeEmbedding`**
- `nn.Embedding(max_visits, d)` — visit index
- `AgeEmbedding` — binned age at visit
- Optional: `DeltaTEmbedding` — binned time gap since last visit

**Module 4: `TimelineEncoder`**
- Transformer over visits
Input: `(B, V, d)` + visit mask `(B, V)`
Output: `(B, V, d)`

**Module 5: `SurvivalHead`**
- MLP → sigmoid
Output: `(B, V)` hazards

**Module 6: `DiscreteTimeSurvivalLoss`**
- Reuse `losses.py` as-is — `sequence_mask = visit_mask`

---

## Recommended Starting Point

The minimal viable hierarchical variant with the best empirical return:

```
Within-visit:  AttnPoolingVisitEncoder  (Option B)
Across-visit:  Transformer
Visit signals: positional + age + Δt embedding
Loss:          DiscreteTimeSurvivalLoss (unchanged)
```

This gives you:
- Explicit visit-level representations (cleaner scientific narrative)
- Learned code importance within visits (more interpretable than mean)
- Time-gap awareness at the visit level (better hazard calibration)
- ~47× cheaper attention for typical EHR sequence lengths

The most important implementation detail is **consistent mask discipline**: the two-level `(m^code, m^visit)` masking must be applied correctly at every stage. Silent bugs in hierarchical models almost always live in mask propagation, not in the model logic itself.

---

**See also:**
- `02_ehr_to_tokens.md` — flat tokenization design and why it was chosen
- `03_behrt_embeddings.md` — embedding design in the flat model
- `08_architecture_decisions.md` §4 — flat vs hierarchical comparison summary
