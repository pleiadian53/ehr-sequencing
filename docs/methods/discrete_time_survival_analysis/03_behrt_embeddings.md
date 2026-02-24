# BEHRT Embeddings as Structured Inductive Bias

**Topics:** Embedding design, age binning, visit vs positional embedding, sinusoidal vs learned, visit ID embedding vs aggregated visit embedding

**Reference code:** `src/ehrsequencing/models/embeddings.py`

---

## Table of Contents

1. [Embedding Sum as Model Prior](#embedding-sum-as-model-prior)
2. [Age Embedding: Why Binned, Not Continuous](#age-embedding-why-binned-not-continuous)
3. [Visit Embedding vs Positional Embedding](#visit-embedding-vs-positional-embedding)
4. [Sinusoidal vs Learned Positional Encoding](#sinusoidal-vs-learned-positional-encoding)
5. [Visit ID Embedding vs Aggregated Visit Embedding](#visit-id-embedding-vs-aggregated-visit-embedding)
6. [Practical Tuning Notes](#practical-tuning-notes)

---

## Embedding Sum as Model Prior

BEHRT uses additive embeddings:

```
token_embedding = code_emb + age_emb + visit_emb + position_emb
```

This is not just an implementation detail — it encodes structural assumptions about what matters:

- **code identity** — which medical concept this token represents
- **age context** — the patient's biological stage at this code
- **visit membership** — which clinical encounter this code belongs to
- **sequence order** — absolute position in the flattened token stream

Each component is a learned lookup table (or fixed sinusoidal function for position). The sum is passed through LayerNorm and dropout before entering the transformer.

**Why summation?** Additive composition is a standard prior in NLP (BERT uses the same pattern). It assumes the contributions of each axis are approximately independent and additive in embedding space. See `dev/models/pretrain_finetune/05_embedding_summation_and_quality_analysis.md` for mathematical justification.

---

## Age Embedding: Why Binned, Not Continuous

`AgeEmbedding` discretizes continuous age into bins before embedding:

```python
# Default: 5-year bins
age_bin = age // age_bin_size   # e.g., age=67 → bin=13
embedding = self.embedding(age_bin)
```

**Pros:**
- Robust to noisy or imprecise age values in EHR data
- Easier optimization than raw continuous feature injection
- Captures coarse life-stage effects (pediatric / adult / elderly)
- Consistent with how BEHRT was originally designed

**Tradeoff:**
- Loses within-bin resolution (all ages 65–69 map to the same embedding)

**When to adjust:** If the outcome hazard is sensitive to narrow age windows (e.g., pediatric dosing thresholds), reduce `age_bin_size` to 1–2 years. Watch for sparsity in older age bins with small cohorts.

---

## Visit Embedding vs Positional Embedding

These two embeddings are **not redundant** — they encode different axes:

| Embedding | Question answered | Granularity |
|-----------|------------------|-------------|
| `VisitEmbedding` | "Which clinical encounter does this token belong to?" | Visit-level |
| `PositionalEmbedding` | "Where is this token in the flattened stream?" | Token-level |

**Why both matter:**

Two tokens can share the same visit ID but differ in token position (e.g., the 1st vs 3rd code within visit 2). Two tokens can have similar position indices but belong to different visits or different patients.

Removing either weakens temporal structure. Visit embedding helps the transformer learn intra-visit vs inter-visit attention patterns. Positional embedding preserves token-level ordering within a visit.

---

## Sinusoidal vs Learned Positional Encoding

`BEHRTEmbedding` supports both modes:

```python
# Learned (default): position embedding is a trainable nn.Embedding
use_sinusoidal=False

# Fixed: position encoding is a non-trainable sinusoidal buffer
use_sinusoidal=True
```

With sinusoidal encoding:
- **Fixed:** positional basis (not updated during training)
- **Learned:** code, age, visit embeddings + transformer weights + task heads

**When to use sinusoidal:**
- Stronger generalization on sequence lengths not seen during training
- Less positional overfitting on small cohorts
- Domain shift across institutions (sinusoidal is more stable)

**When to use learned:**
- Larger cohorts where positional patterns are worth learning
- Sequences with consistent length distributions

---

## Visit ID Embedding vs Aggregated Visit Embedding

This is the most important conceptual distinction in the BEHRT survival architecture. There are **two different things** called "visit embedding":

### Visit ID Embedding (Input-Side)

```python
# In BEHRTEmbedding.forward()
visit_emb = self.visit_embedding(visit_ids)   # lookup table
token_emb = code_emb + age_emb + visit_emb + pos_emb
```

- **What it is:** A learned lookup table mapping visit index → embedding vector
- **When it runs:** Before the transformer, as part of input construction
- **What it encodes:** "This token belongs to visit #3" — purely structural metadata
- **Medical content:** None — it knows nothing about what codes are in visit #3

### Aggregated Visit Embedding (Output-Side)

```python
# In BEHRTForSurvival.aggregate_visits()
visit_embeddings = scatter_add(contextualized_code_embeddings, visit_ids)
visit_embeddings /= visit_counts   # mean pooling
```

- **What it is:** Mean pool of BEHRT-contextualized code embeddings within a visit
- **When it runs:** After the transformer, as part of hazard prediction
- **What it encodes:** "Visit #3 contained these conditions, in this patient context"
- **Medical content:** Rich — it reflects the full bidirectional context from the transformer

### The Full Data Flow

```
codes (tokens)
    ↓
BEHRTEmbedding:
    code_emb + age_emb + visit_emb + pos_emb   ← visit_emb is structural only
    ↓
Transformer encoder (self-attention across all codes)
    ↓
code_embeddings: (B, L, H)   ← each code now "knows about" neighboring codes
    ↓
aggregate_visits (scatter_add mean-pool)
    ↓
visit_embeddings: (B, V, H)  ← content-aware visit representation
    ↓
hazard head → hazards: (B, V)
```

`VisitEmbedding` helps the transformer learn **intra-visit vs inter-visit attention patterns** during encoding. `aggregate_visits` produces the actual **content-rich visit representation** used for hazard prediction. They serve complementary roles at different stages.

---

## Practical Tuning Notes

**Small cohort (< 2K patients):**
- Keep embedding defaults
- Avoid over-parameterizing (large `embedding_dim` with little data)
- Use sinusoidal positional encoding to reduce overfitting

**Larger cohort (> 10K patients):**
- Learned positional embeddings often pay off
- Consider increasing `embedding_dim` if vocabulary is large

**Domain shift across institutions:**
- Sinusoidal positional encoding is more stable
- Consider freezing age and visit embeddings if source/target age distributions differ

---

**See also:** `01a_visit_embeddings.md` for a deep treatment of the visit ID vs aggregated visit embedding distinction, including scatter_add mechanics and gradient flow analysis.

**Next:** `04_pretraining_objectives.md` — what MLM and Next-Visit Prediction teach the encoder.
