# BEHRT for Discrete-Time Survival Analysis: Documentation

**Last Updated:** 2026-02-21
**Purpose:** Comprehensive guide to BEHRT-based survival modeling on EHR data

---

## Overview

This documentation series explains how BEHRT (Bidirectional Encoder Representations from Transformers for EHRs) is adapted for discrete-time survival analysis. Documents are organized into two tracks:

- **Tutorial track (`01`–`08`):** Concise, concept-focused documents in logical reading order — from survival math foundations through architecture decisions.
- **Deep-reference track (`00_`, `00a_`, `00b_`):** Comprehensive treatments of the core architecture, visit embedding design, and data representation. Read these for full mathematical detail and implementation depth.

**Target audience:** Researchers, ML engineers, and data scientists working with EHR survival models.

---

## Tutorial Track (Recommended Reading Order)

### [01_survival_framework.md](./01_survival_framework.md)
Survival math foundations: hazard function, survival function, censoring, label preparation, synthetic data generation, evaluation metrics (C-index, Brier score).

### [02_ehr_to_tokens.md](./02_ehr_to_tokens.md)
How hierarchical EHR data becomes BEHRT-ready tensors: flattening, dual role of `visit_ids`, tensor contract, padding discipline, flat vs hierarchical tradeoff.

### [03_behrt_embeddings.md](./03_behrt_embeddings.md)
BEHRT embedding design as structured inductive bias: age binning, visit vs positional embedding, sinusoidal vs learned, visit ID embedding vs aggregated visit embedding.

### [04_pretraining_objectives.md](./04_pretraining_objectives.md)
What MLM and Next-Visit Prediction each teach the encoder, joint vs sequential training, recommended pretraining strategy.

### [05_survival_head_and_aggregation.md](./05_survival_head_and_aggregation.md)
The bridge from token-level transformer output to visit-level hazards: `scatter_add` mechanics, hazard head, discrete-time interpretation, visit mask.

### [06_loss_functions.md](./06_loss_functions.md)
Loss choice as a value system: NLL vs C-index disconnect, pairwise ranking loss, hybrid loss, tuning `lambda_rank`, practical recommendations.

### [07_optimization_strategies.md](./07_optimization_strategies.md)
Frozen encoder vs LoRA vs full fine-tuning: parameter tradeoffs, practical recipes by data regime, operational notes.

### [08_architecture_decisions.md](./08_architecture_decisions.md)
LSTM baseline architecture, BEHRT vs LSTM comparison, flat vs hierarchical transformer tradeoffs, benchmark framework and expected performance.

### [09_hierarchical_architecture.md](./09_hierarchical_architecture.md)
Deep dive into hierarchical BEHRT: two-stage encoder design (within-visit + across-visit), attention pooling mechanics, Δt embedding, implementation blueprint, and flat vs hierarchical comparison.

---

## Deep-Reference Track

### [00_behrt_model_overview.md](./00_behrt_model_overview.md)
Complete end-to-end pipeline: learning hierarchy (5 levels), all training objectives, loss function comparison tables, training strategy decision matrix, optimization loop, gradient flow, common pitfalls.

### [00a_visit_embeddings.md](./00a_visit_embeddings.md)
Deep dive into the two "visit embeddings": visit ID embedding (input-side lookup) vs aggregated visit embedding (output-side representation). Includes step-by-step `scatter_add` explanation, gradient flow analysis, and common misconceptions.

### [00b_ehr_tokens_tensors.md](./00b_ehr_tokens_tensors.md)
Deep dive into flattening hierarchical EHR data: forest/vine analogy, attention mask implementation, padding bug examples, testing correctness, design tradeoff analysis.

---

## Quick Navigation

### By Topic

**Survival math:**
- Framework and notation → `01_survival_framework.md`
- Label preparation → `01_survival_framework.md` §4
- Evaluation metrics → `01_survival_framework.md` §6

**Data processing:**
- Flattening overview → `02_ehr_to_tokens.md`
- Flattening deep dive → `00b_ehr_tokens_tensors.md`
- Padding discipline → `02_ehr_to_tokens.md` §5, `00b_ehr_tokens_tensors.md` §4

**Embeddings:**
- Embedding design overview → `03_behrt_embeddings.md`
- Visit ID vs aggregated visit embedding → `03_behrt_embeddings.md` §5, `00a_visit_embeddings.md`
- scatter_add mechanics → `05_survival_head_and_aggregation.md` §2, `00a_visit_embeddings.md` §4

**Pretraining:**
- MLM and NVP objectives → `04_pretraining_objectives.md`
- Full pipeline overview → `00_behrt_model_overview.md` §2

**Survival head:**
- Aggregation + hazard head → `05_survival_head_and_aggregation.md`
- Full architecture detail → `00_behrt_model_overview.md` §1.4–1.5

**Loss functions:**
- Overview and tuning → `06_loss_functions.md`
- Full mathematical treatment → `06_loss_functions.md`

**Training strategy:**
- Frozen / LoRA / full → `07_optimization_strategies.md`
- Decision matrix → `00_behrt_model_overview.md` §3

**Architecture decisions:**
- BEHRT vs LSTM → `08_architecture_decisions.md` §3
- Flat vs hierarchical (overview) → `08_architecture_decisions.md` §4, `02_ehr_to_tokens.md` §6
- Hierarchical BEHRT (deep dive) → `09_hierarchical_architecture.md`

### By Experience Level

**New to BEHRT survival:**
1. `01_survival_framework.md` — understand the task
2. `02_ehr_to_tokens.md` — understand the data
3. `03_behrt_embeddings.md` — understand the model inputs
4. `05_survival_head_and_aggregation.md` — understand the model outputs
5. `07_optimization_strategies.md` — understand how to train

**Implementing or debugging:**
1. `00b_ehr_tokens_tensors.md` — data pipeline deep dive
2. `00a_visit_embeddings.md` — aggregation deep dive
3. `00_behrt_model_overview.md` — full pipeline reference

**Choosing architecture or loss:**
1. `06_loss_functions.md` — loss function tradeoffs
2. `07_optimization_strategies.md` — training strategy
3. `08_architecture_decisions.md` — BEHRT vs LSTM, flat vs hierarchical
4. `09_hierarchical_architecture.md` — hierarchical BEHRT design and implementation

---

## Common Questions

**Q: What is the difference between visit ID embedding and aggregated visit embedding?**
A: See `03_behrt_embeddings.md` §5 for a concise answer, or `00a_visit_embeddings.md` for the full treatment.

**Q: Why flatten instead of using a hierarchical transformer?**
A: See `02_ehr_to_tokens.md` §6 and `08_architecture_decisions.md` §4.


**Q: Which loss function should I use?**
A: Start with hybrid loss (`lambda_rank=0.05`). See `06_loss_functions.md` §7 for decision guidance.

**Q: Which training strategy should I use?**
A: LoRA by default for medium cohorts. See `07_optimization_strategies.md` §5.

**Q: Why does NLL not directly optimize C-index?**
A: See `06_loss_functions.md` §3.

**Q: How do I verify my data pipeline is correct?**
A: See `02_ehr_to_tokens.md` §7 for practical checks.

---

## Code References

| Component | File |
|-----------|------|
| Dataset and flattening | `src/ehrsequencing/data/behrt_survival_dataset.py` |
| Embeddings | `src/ehrsequencing/models/embeddings.py` |
| BEHRT encoder | `src/ehrsequencing/models/behrt.py` |
| Survival model | `src/ehrsequencing/models/behrt_survival.py` |
| Loss functions | `src/ehrsequencing/models/losses.py` |
| LoRA | `src/ehrsequencing/models/lora.py` |
| Training script | `examples/survival_analysis/train_behrt_survival.py` |
| Validation script | `examples/survival_analysis/validate_behrt_survival.py` |

---

## Related Documentation

- `dev/workflow/BEHRT_SURVIVAL_ANALYSIS_DESIGN.md` — original design specification
- `dev/workflow/ROADMAP.md` — project roadmap and phase status
- `dev/models/pretrain_finetune/` — BEHRT pre-training documentation
- `dev/methods/discrete_time_survival_analysis/` — internal design notes and session logs

## Changelog

| Date | Changes |
|------|---------|
| 2026-02-21 | Added 09_hierarchical_architecture.md; consolidated into unified 01–08 tutorial track; retired session_0X files; moved POLISHING_SUMMARY.md to dev/ |
| 2026-02-03 | Initial documentation release: 01_behrt_model_overview.md, 01a_visit_embeddings.md, 01b_ehr_tokens_tensors.md, README |
