# Session 2 — BEHRT embeddings as structured inductive bias

This session focuses on why BEHRT embedding design matters for survival performance.

## Learning goals

- Understand age binning and its tradeoffs.
- Distinguish visit embedding from positional embedding.
- Understand what changes when `use_sinusoidal=True`.

Reference code: `src/ehrsequencing/models/embeddings.py`

---

## 1) Embedding sum as model prior

BEHRT uses:

`token_embedding = code + age + visit + position`

This is not just implementation detail; it encodes assumptions:

- code identity matters
- age context matters
- visit membership matters
- absolute sequence order matters

---

## 2) Age embedding: why binned, not continuous

`AgeEmbedding` discretizes age with `age_bin_size=5` years by default.

Pros:

- robust to noisy age values
- easier optimization than raw continuous feature injection
- captures coarse life-stage effects (pediatric/adult/elderly)

Tradeoff:

- loses within-bin resolution

If outcome hazard is sensitive to narrow age windows, consider smaller bin size (e.g., 1–2 years), but watch sparsity.

---

## 3) Visit embedding vs positional embedding (not redundant)

They encode different axes:

- **Visit embedding**: "which clinical encounter does this token belong to?"
- **Positional embedding**: "where is this token in flattened stream?"

Why both matter:

- Two tokens can share visit id but differ in token position.
- Two tokens can share similar position indices but belong to different visits/patients.

Removing either weakens temporal structure.

---

## 4) What changes with `use_sinusoidal=True`

In `BEHRTEmbedding`:

- `use_sinusoidal=False` (default): position embedding is learned (`nn.Embedding`)
- `use_sinusoidal=True`: position encoding is fixed (buffer, not trainable)

So with sinusoidal:

- **fixed**: positional basis
- **learned**: code, age, visit embeddings + transformer + task heads

Use sinusoidal when you want stronger generalization on sequence lengths/order patterns and less positional overfitting.

---

## 5) Practical tuning notes

- Small cohort: keep defaults, avoid over-parameterizing embeddings.
- Larger cohort: learned positional embeddings often pay off.
- Domain shift across institutions: sinusoidal can be more stable.

---

## 6) Takeaway

Embedding design is your first and strongest inductive bias. In BEHRT survival, age+visit+position is what makes flat token streams clinically structured.
