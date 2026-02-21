# Session 4 — Survival head + visit aggregation

This session explains the bridge from token-level transformer output to visit-level hazards.

Reference code: `src/ehrsequencing/models/behrt_survival.py`

---

## Learning goals

- Why hazard should be predicted at visit level.
- What `scatter_add` aggregation computes geometrically.
- What "hazard per visit" means for time discretization.

---

## 1) Why aggregate tokens into visits?

The BEHRT encoder outputs one vector per token (`[B, L, d]`).

But the survival target here is event timing over visit intervals. So you need:

- token -> visit pooling
- visit embeddings -> hazard trajectory `[B, T]`

Without aggregation, hazard semantics are misaligned with labels (`event_time` in visit index space).

---

## 2) What `aggregate_visits` is doing

Inside `aggregate_visits(...)`:

1. zero out padded tokens via `attention_mask`
2. `scatter_add_` token embeddings into bins indexed by `visit_ids`
3. `scatter_add_` visit token counts
4. divide sums by counts (mean pooling)

Geometrically: each visit embedding is the centroid of its token vectors in hidden space.

---

## 3) Hazard head

Hazard head is a small MLP + sigmoid per visit embedding:

`visit_embedding_t -> hazard_t in (0,1)`

Then hazards are masked with `visit_mask` to remove padded visits.

---

## 4) What "hazard per visit" means

Time is discretized by encounter index, not wall-clock days.

Implications:

- hazard step size is irregular in real time if visit spacing varies
- interpretation is "risk at this observed visit step"
- event/censor labels must be aligned to visit index

If you need day-level interpretability, add explicit elapsed-time features or move to calendar-time bins.

---

## 5) Takeaway

The core survival abstraction here is:

- contextualize at token granularity
- decide at visit granularity

`scatter_add` is the key algebra that preserves both efficiency and structure.
