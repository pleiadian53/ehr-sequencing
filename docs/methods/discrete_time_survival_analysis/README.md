# BEHRT for Discrete-Time Survival Analysis: Documentation

**Last Updated:** 2026-02-03  
**Purpose:** Comprehensive guide to BEHRT-based survival modeling

---

## Overview

This documentation series explains how BEHRT (Bidirectional Encoder Representations from Transformers) is adapted for discrete-time survival analysis on EHR data. The documents progress from high-level architecture to low-level implementation details.

**Target audience:** Researchers, ML engineers, and data scientists working with EHR survival models.

---

## Document Series

### [01_behrt_model_overview.md](./01_behrt_model_overview.md)

**What:** Complete optimization pipeline from tokens to gradients

**Topics covered:**
- Learning hierarchy (embeddings → transformer → task heads)
- Training objectives (MLM, next-visit, survival)
- Loss functions (NLL, ranking, hybrid)
- Training strategies (frozen, LoRA, full fine-tune)
- Optimization loop and gradient flow

**Read this if:**
- New to BEHRT survival models
- Want end-to-end pipeline understanding
- Need to choose training strategy
- Deciding which loss function to use

**Key takeaway:** BEHRT learns contextual representations through pre-training, then adapts to survival prediction via task-specific heads and losses.

---

### [01a_visit_embeddings.md](./01a_visit_embeddings.md)

**What:** Deep dive into two conceptually different "visit embeddings"

**Topics covered:**
- Visit ID embeddings (input-side lookup)
- Aggregated visit embeddings (output-side representation)
- Direct comparison and key differences
- scatter_add mechanics and implementation
- Gradient flow and optimization implications

**Read this if:**
- Confused about "visit embedding" terminology
- Need to understand aggregation process
- Implementing visit-level models
- Debugging aggregation issues

**Key takeaway:** Visit ID embedding = "this token is from visit 3" (input metadata). Aggregated visit embedding = "visit 3 contains these conditions" (output representation).

---

### [01b_ehr_tokens_tensors.md](./01b_ehr_tokens_tensors.md)

**What:** How hierarchical EHR data becomes flat token sequences

**Topics covered:**
- Flattening hierarchical data (patient → visits → codes)
- Preserving structure through visit_ids
- Dual role of visit_ids (feature + index)
- Attention masking and padding discipline
- Implementation best practices

**Read this if:**
- Want to understand data preprocessing
- Need to implement flattening logic
- Debugging padding/masking issues
- Comparing hierarchical vs flat architectures

**Key takeaway:** Flattening is pragmatic and enables BEHRT pre-training, but requires rigorous attention masking to preserve hierarchical structure.

---

## Quick Navigation

### By Topic

**Architecture:**
- Model components → `01_behrt_model_overview.md` (Section 1)
- Embedding layers → `01_behrt_model_overview.md` (Section 1.1-1.2)
- Visit aggregation → `01a_visit_embeddings.md` (Section 2)

**Data Processing:**
- Flattening hierarchy → `01b_ehr_tokens_tensors.md` (Section 2)
- Attention masking → `01b_ehr_tokens_tensors.md` (Section 4)
- Padding handling → `01b_ehr_tokens_tensors.md` (Section 4)

**Training:**
- Pre-training objectives → `01_behrt_model_overview.md` (Section 2.1-2.2)
- Survival losses → `01_behrt_model_overview.md` (Section 2.3)
- Training strategies → `01_behrt_model_overview.md` (Section 3)

**Implementation:**
- scatter_add mechanics → `01a_visit_embeddings.md` (Section 4)
- Masking best practices → `01b_ehr_tokens_tensors.md` (Section 9)
- Common pitfalls → `01_behrt_model_overview.md` (Section 8)

### By Use Case

**"I want to..."**

- **Understand the full pipeline** → Read `01_behrt_model_overview.md`
- **Implement visit aggregation** → Read `01a_visit_embeddings.md`
- **Preprocess EHR data** → Read `01b_ehr_tokens_tensors.md`
- **Choose a loss function** → `01_behrt_model_overview.md` (Section 2.3)
- **Choose training strategy** → `01_behrt_model_overview.md` (Section 3)
- **Debug padding issues** → `01b_ehr_tokens_tensors.md` (Section 4)
- **Understand gradient flow** → `01a_visit_embeddings.md` (Section 6)

### By Experience Level

**Beginner (new to project):**
1. Start with `01_behrt_model_overview.md` (skip math details first pass)
2. Read `01b_ehr_tokens_tensors.md` (understand data processing)
3. Skim `01a_visit_embeddings.md` (come back when confused about embeddings)

**Intermediate (familiar with basics):**
1. Deep dive into `01_behrt_model_overview.md` (all sections)
2. Study `01a_visit_embeddings.md` (understand visit representations)
3. Review `01b_ehr_tokens_tensors.md` (implementation details)

**Advanced (modifying codebase):**
1. Use as reference for implementation decisions
2. Consult gradient flow sections for optimization debugging
3. Review best practices sections before major changes

---

## Common Questions

### Q: What's the difference between visit ID embedding and aggregated visit embedding?

**A:** Visit ID embedding is an **input feature** (learned lookup table added to tokens before transformer). Aggregated visit embedding is an **output representation** (computed from transformer outputs by grouping tokens).

**See:** `01a_visit_embeddings.md` (complete explanation)

### Q: Why flatten hierarchical EHR data instead of using a hierarchical model?

**A:** Flattening enables:
- Use of pre-trained BEHRT
- Full cross-visit attention
- Simpler architecture
- Better code reuse

**Trade-off:** Requires careful attention masking.

**See:** `01b_ehr_tokens_tensors.md` (Section 5)

### Q: Which survival loss should I use?

**A:** Depends on your evaluation metric:
- **NLL:** Best for calibration (Brier score)
- **Ranking:** Best for discrimination (C-index)
- **Hybrid:** Balances both (recommended starting point)

**See:** `01_behrt_model_overview.md` (Section 2.3)

### Q: Should I use frozen BEHRT, LoRA, or full fine-tuning?

**A:** Depends on dataset size and resources:
- **Frozen:** Small data (< 1K patients), fast prototyping
- **LoRA:** Standard scenario (1K-10K patients), best efficiency/performance
- **Full:** Large data (> 10K patients), maximum flexibility

**See:** `01_behrt_model_overview.md` (Section 3)

### Q: How do I handle padding correctly?

**A:** Three critical steps:
1. Create attention mask: `mask = (codes != 0).long()`
2. Pass to transformer: `model(..., attention_mask=mask)`
3. Mask before aggregation: `hidden * mask.unsqueeze(-1)`

**See:** `01b_ehr_tokens_tensors.md` (Section 4)

### Q: What is scatter_add doing?

**A:** Vectorized group-by-sum operation that aggregates tokens into visits:

```
For each token i: add hidden_states[i] to visit_embedding[visit_ids[i]]
Then divide by visit size (mean pooling)
```

**See:** `01a_visit_embeddings.md` (Section 4)

---

## Implementation Checklist

### For New Implementations

- [ ] Read `01_behrt_model_overview.md` (understand full pipeline)
- [ ] Read `01b_ehr_tokens_tensors.md` (data preprocessing)
- [ ] Read `01a_visit_embeddings.md` (aggregation logic)
- [ ] Review code: `src/ehrsequencing/models/behrt_survival.py`
- [ ] Test attention masking correctness
- [ ] Validate visit aggregation with small examples
- [ ] Monitor both calibration and discrimination metrics

### For Debugging

**Padding issues:**
- [ ] Check attention mask creation
- [ ] Verify mask passed to transformer
- [ ] Validate masking before aggregation
- [ ] Test: padding should not affect outputs

**Aggregation issues:**
- [ ] Print visit_ids and attention_mask
- [ ] Check scatter_add index dimensions
- [ ] Verify division by visit counts
- [ ] Test: same visit different patients should aggregate differently

**Training issues:**
- [ ] Check loss computation (masked properly?)
- [ ] Monitor loss components (NLL vs ranking)
- [ ] Verify gradient flow (check grad norms)
- [ ] Validate metrics (C-index, Brier score)

---

## Related Documentation

### Pre-training and Fine-tuning

- **`dev/models/pretrain_finetune/`** - BEHRT pre-training documentation
- **`dev/models/pretrain_finetune/01_behrt_model_design.md`** - Architecture details
- **`dev/models/pretrain_finetune/07_lora_deep_dive.md`** - LoRA comprehensive guide

### Embeddings

- **`dev/models/pretrain_finetune/05_embedding_summation_and_quality_analysis.md`** - Why sum embeddings

### Benchmarking

- **`dev/models/pretrain_finetune/06_benchmarking_updates.md`** - Evaluation strategies

---

## Code References

### Core Models

| File | Description |
|------|-------------|
| `src/ehrsequencing/models/behrt.py` | BEHRT base architecture |
| `src/ehrsequencing/models/embeddings.py` | Embedding layers |
| `src/ehrsequencing/models/behrt_survival.py` | Survival model implementation |
| `src/ehrsequencing/models/losses.py` | Survival loss functions |

### Training Scripts

| File | Description |
|------|-------------|
| `examples/survival_analysis/train_lstm.py` | Full training pipeline |
| `examples/survival_analysis/train_lstm_demo.py` | Demo/quick start |

### Utilities

| File | Description |
|------|-------------|
| `src/ehrsequencing/models/lora.py` | LoRA implementation |
| `src/ehrsequencing/benchmarks/` | Evaluation metrics |

---

## Coming Soon

### Planned Documentation

1. **02_survival_losses.md** - Mathematical derivation of survival losses
2. **03_evaluation_metrics.md** - Comprehensive guide to survival metrics
3. **04_training_recipes.md** - Practical training configurations
4. **05_troubleshooting.md** - Common issues and solutions

### Planned Examples

1. **Minimal working example** - 50-line script demonstrating key concepts
2. **Complete training pipeline** - Production-ready training script
3. **Custom loss functions** - How to implement new survival losses
4. **Multi-task learning** - Pre-training + survival jointly

---

## Mathematical Notation Reference

### Common Symbols

| Symbol | Meaning | Typical Dimensions |
|--------|---------|-------------------|
| B | Batch size | - |
| L | Sequence length (tokens) | - |
| T | Number of visits | - |
| d | Embedding dimension | - |
| \|V\| | Vocabulary size | - |
| c_i | Code ID at position i | ∈ {0, ..., \|V\|-1} |
| v_i | Visit ID at position i | ∈ {0, ..., T-1} |
| x_i | Token embedding | ∈ ℝ^d |
| H | Contextual hidden states | ∈ ℝ^(B×L×d) |
| V_t | Visit representation | ∈ ℝ^d |
| h_t | Hazard at visit t | ∈ (0, 1) |
| E^code | Code embedding matrix | ∈ ℝ^(\|V\|×d) |
| E^visit | Visit ID embedding matrix | ∈ ℝ^(T_max×d) |

### Set Notation

| Notation | Meaning |
|----------|---------|
| I_{b,t} | Set of token indices in visit t for patient b |
| 𝟙(·) | Indicator function (1 if true, 0 if false) |
| Σ_{i ∈ I} | Sum over indices in set I |

---

## New: 6-session tutorial track (2026-02-21)

If you want a workshop-style progression, start here:

- `tutorial_sessions_index.md`
- Session 1: `session_01_ehr_to_tokens_to_tensors.md`
- Session 2: `session_02_behrt_embeddings_and_inductive_bias.md`
- Session 3: `session_03_pretraining_objectives.md`
- Session 4: `session_04_survival_head_and_visit_aggregation.md`
- Session 5: `session_05_losses_as_value_systems.md`
- Session 6: `session_06_optimization_strategies.md`

## Changelog

| Date | Changes |
|------|---------|
| 2026-02-03 | Initial documentation release |
| - | Published 01_behrt_model_overview.md |
| - | Published 01a_visit_embeddings.md |
| - | Published 01b_ehr_tokens_tensors.md |
| - | Created README with navigation |

---

## Feedback

This documentation is actively maintained. If you find:
- ✅ Errors or inconsistencies
- ✅ Unclear explanations
- ✅ Missing topics
- ✅ Implementation bugs

Please:
1. Check the code references to verify current implementation
2. Review related documentation for additional context
3. Open an issue with specific questions

---

**Status:** Active documentation, regularly updated  
**Version:** 1.0  
**Last Updated:** 2026-02-03
