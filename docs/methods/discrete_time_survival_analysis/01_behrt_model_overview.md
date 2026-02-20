# BEHRT for Discrete-Time Survival Analysis: Model Overview

**Last Updated:** 2026-02-03  
**Topics:** Model architecture, optimization objectives, training strategies

---

## Overview

This document provides a comprehensive overview of how BEHRT (Bidirectional Encoder Representations from Transformers for EHRs) is adapted for discrete-time survival analysis. We'll walk through the complete optimization pipeline: tokens → embeddings → sequence model → task heads → loss → gradients → parameter updates.

**Key insight:** EHR-seq trains a BEHRT encoder to turn medical code sequences into contextual representations, then learns task-specific heads by minimizing losses that encode either probability quality (NLL), ranking quality (pairwise), or both (hybrid).

---

## 1. The Learning Hierarchy

BEHRT learns a hierarchy of representations, each serving a specific purpose in the modeling pipeline.

### 1.1 Medical Code Embeddings

**What:** A trainable lookup table mapping discrete medical codes to continuous vectors.

**Implementation:**
```python
# In BEHRTEmbedding
code_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
```

**Role:** These vectors are trainable parameters updated by backpropagation, similar to word embeddings in NLP.

**Dimensions:**
- Input: Code ID ∈ {0, 1, ..., vocab_size-1}
- Output: Embedding vector ∈ ℝ^d where d = embedding_dim

### 1.2 Temporal Embeddings

**What:** Additional embeddings that inject temporal context into the representation.

**Components:**

BEHRT uses **additive** temporal embeddings:

```
x = e_code + e_age + e_visit + e_pos
```

Where:
- **Age embedding**: Binned age → embedding vector
- **Visit embedding**: Visit index → embedding vector  
- **Positional embedding**: Sequence position → embedding vector (learnable or sinusoidal)

**Implementation:**
```python
# From embeddings.py
embeddings = code_emb + age_emb + visit_emb + pos_emb
embeddings = self.layer_norm(embeddings)
embeddings = self.dropout(embeddings)
```

**Why summation?** See `dev/models/pretrain_finetune/05_embedding_summation_and_quality_analysis.md` for mathematical justification.

### 1.3 Sequence Representations

**What:** Contextual hidden states from transformer encoder.

**Architecture:**
```python
# BEHRT wraps nn.TransformerEncoder
self.encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=config.num_layers
)
```

**Transformation:**
- Input: Token-level embeddings H_in ∈ ℝ^(B×L×d)
- Output: Contextual hidden states H ∈ ℝ^(B×L×d)

Where:
- B = batch size
- L = sequence length (number of tokens)
- d = embedding dimension

**What it learns:** Through self-attention, the encoder captures:
- Medical code semantics and relationships
- Temporal context and disease progression
- Co-morbidity structure
- Long-range dependencies in patient history

### 1.4 Visit-Level Representations

**What:** Aggregated representations for survival modeling.

**Why needed:** BEHRT produces **token-level** embeddings, but survival analysis requires **visit-level** hazards.

**Aggregation method:**

For each visit t in patient b:

```
V_{b,t} = (1/|I_{b,t}|) * Σ_{i ∈ I_{b,t}} H_{b,i}
```

Where:
- I_{b,t} = set of token indices belonging to visit t
- V_{b,t} ∈ ℝ^d is the visit representation
- This is mean pooling over tokens within a visit

**Implementation:** Uses vectorized `scatter_add` for efficiency (see `01b_ehr_tokens_tensors.md` for details).

### 1.5 Task Heads

**What:** Task-specific prediction layers built on top of BEHRT representations.

**Available heads:**

| Task Head | Input | Output | Loss Function |
|-----------|-------|--------|---------------|
| **MLM Head** | Token embeddings H | Logits over vocab | CrossEntropyLoss |
| **Next-Visit Head** | Patient embedding (CLS/mean/max) | Logits over vocab | BCEWithLogitsLoss |
| **Survival Head** | Visit embeddings V | Per-visit hazard h_t ∈ (0,1) | Survival losses (see below) |

---

## 2. Training Objectives

### 2.1 Pre-training: Masked Language Modeling (MLM)

**Goal:** Learn contextual code representations by predicting masked tokens.

**Forward pass:**
1. BEHRT produces hidden states H ∈ ℝ^(B×L×d)
2. MLM head maps H → logits ∈ ℝ^(B×L×|V|)

**Loss:**
```python
loss = CrossEntropyLoss(logits[masked_positions], labels[masked_positions])
```

**What gets trained:** Code embeddings + temporal embeddings + transformer weights + MLM head

**Intuition:** Forces the model to understand medical code semantics and relationships from context.

**Reference:** `src/ehrsequencing/models/behrt.py`

### 2.2 Pre-training: Next Visit Prediction

**Goal:** From patient representation, predict codes in the next visit.

**Forward pass:**
1. Pool patient embedding (CLS token / mean / max pooling)
2. Head outputs logits over vocabulary
3. Loss: BCEWithLogitsLoss against multi-hot vector of next-visit codes

**What it learns:** Trajectory patterns and disease progression ("what tends to come next")

**Loss formulation:**
```python
# Multi-label classification
loss = BCEWithLogitsLoss(logits, target_multi_hot)
```

**Reference:** `src/ehrsequencing/models/behrt.py`

### 2.3 Downstream: Discrete-Time Survival Analysis

**Goal:** Predict time-to-event with visit-level hazards.

**Model output:** For each visit t, predict hazard h_t ∈ (0,1) via sigmoid activation.

#### Loss Option 1: Negative Log-Likelihood (NLL)

**Implementation:** `DiscreteTimeSurvivalLoss`

**Formulation:**

For each patient with event/censoring time T:

```
Log-likelihood = Σ_{t<T} log(1 - h_t) + δ * log(h_T)
```

Where:
- δ = 1 if event observed, 0 if censored
- First term: survived visits before T
- Second term: event occurred at T (if observed)

**Objective:** Minimize negative mean log-likelihood across batch.

**Intuition:** Trains **calibrated hazard probabilities** (probabilities that behave like probabilities).

**Reference:** `src/ehrsequencing/models/losses.py`

#### Loss Option 2: Pairwise Ranking Loss

**Implementation:** `PairwiseRankingLoss`

**Formulation:**

For comparable pairs (i, j) where:
- Patient i had event at time t_i
- Patient j survived beyond t_i (or censored later)

Penalize if: cumulative_risk(i) < cumulative_risk(j)

**Objective:** Maximize concordance (C-index).

**Intuition:** Trains **discrimination** (correct ordering of risk), often improving C-index at the expense of calibration.

**Reference:** `src/ehrsequencing/models/losses.py`

#### Loss Option 3: Hybrid (NLL + Ranking)

**Implementation:** `HybridSurvivalLoss`

**Formulation:**
```
L = λ_nll * L_nll + λ_rank * L_rank
```

**Returns:** Both components for logging.

**Philosophy:** Acknowledges that:
- NLL trains probability quality
- Ranking trains ordering quality
- Both are valuable for different reasons

**Tuning:** Adjust λ_rank to balance calibration vs discrimination.

**Reference:** `src/ehrsequencing/models/losses.py`

---

## 3. Training Strategies

### 3.1 Parameter Update Policies

**In `BEHRTForSurvival`, three training modes are supported:**

| Strategy | What's Trainable | When to Use | Trade-offs |
|----------|------------------|-------------|------------|
| **Frozen BEHRT** | Only hazard head | Small datasets, fast prototyping | Most stable, least flexible |
| **LoRA** | Low-rank adapters + head | Medium datasets, GPU constraints | Best efficiency/performance balance |
| **Full Fine-tune** | All parameters | Large datasets, maximum flexibility | Most expressive, most overfit-prone |

**Implementation detail:** These strategies directly change the optimization landscape by controlling which parameters receive gradients.

**Reference:** `src/ehrsequencing/models/behrt_survival.py`

### 3.2 Frozen BEHRT Strategy

**Configuration:**
```python
model = BEHRTForSurvival(config, freeze_behrt=True)
```

**What happens:**
- BEHRT encoder parameters frozen
- Only hazard head receives gradients
- Fastest training, lowest memory

**Use cases:**
- Limited training data (< 1000 patients)
- Quick experimentation
- When pre-trained representations are already good

### 3.3 LoRA Strategy

**Configuration:**
```python
from ehrsequencing.models.lora import apply_lora_to_behrt

model = BEHRTForSurvival(config)
model.behrt = apply_lora_to_behrt(
    model.behrt,
    rank=8,
    lora_attention=True,
    train_embeddings=True,
    train_head=True
)
```

**What happens:**
- BEHRT encoder weights frozen
- Low-rank adapters (B, A matrices) injected and trained
- 98% parameter reduction with minimal performance loss

**Use cases:**
- Standard training scenario (1K-10K patients)
- GPU memory constraints
- Need task adaptation with efficiency

**Details:** See `dev/models/pretrain_finetune/07_lora_deep_dive.md`

### 3.4 Full Fine-tune Strategy

**Configuration:**
```python
model = BEHRTForSurvival(config, freeze_behrt=False)
```

**What happens:**
- All parameters receive gradients
- Most flexible, highest capacity
- Highest memory usage and overfit risk

**Use cases:**
- Large datasets (> 10K patients)
- Domain very different from pre-training
- Maximum performance required

---

## 4. The Optimization Loop

**Standard training loop for any task:**

```python
# 1. Batch data
batch = dataloader.next()  # codes, ages, visit_ids, masks, labels

# 2. Forward pass
outputs = model(**batch)  # logits or hazards

# 3. Compute loss
loss = loss_fn(outputs, targets)

# 4. Backpropagation
loss.backward()  # Compute ∇_θ L

# 5. Parameter update
optimizer.step()  # θ_new = θ_old - lr * ∇_θ L
optimizer.zero_grad()

# 6. Evaluate metrics
metrics = compute_metrics(outputs, targets)  # C-index, calibration, etc.
```

**Key components:**
- **Optimizer:** Typically AdamW with weight decay
- **Learning rate:** Often with warm-up and decay schedule
- **Early stopping:** Monitor validation C-index or loss
- **Metrics:** Loss, C-index, Brier score, calibration curves

**Reference:** `examples/survival_analysis/train_lstm.py` for complete training loops

---

## 5. Architecture Summary

### Complete Pipeline

```
EHR Data (visits → codes)
    ↓
Flatten to token sequence
    ↓
Token embeddings (code + age + visit + position)
    ↓
Transformer encoder (BEHRT)
    ↓
Task-specific head
    ↓
Loss function
    ↓
Gradients → Parameter updates
```

### Information Flow

**Pre-training (MLM):**
```
Tokens → BEHRT → Token representations → MLM head → CrossEntropy loss
```

**Pre-training (Next-Visit):**
```
Tokens → BEHRT → Patient representation → NVP head → BCE loss
```

**Survival analysis:**
```
Tokens → BEHRT → Visit aggregation → Hazard head → Survival loss
```

### Mathematical Notation Summary

| Symbol | Meaning | Dimensions |
|--------|---------|------------|
| B | Batch size | - |
| L | Sequence length (tokens) | - |
| T | Number of visits | - |
| d | Embedding dimension | - |
| \|V\| | Vocabulary size | - |
| c_i | Code ID at position i | ∈ {0, ..., \|V\|-1} |
| x_i | Token embedding | ∈ ℝ^d |
| H | Contextual hidden states | ∈ ℝ^(B×L×d) |
| V_t | Visit representation | ∈ ℝ^d |
| h_t | Hazard at visit t | ∈ (0, 1) |

---

## 6. Key Design Decisions

### Why Flatten Visits?

**Alternative:** Hierarchical model (visit encoder → patient encoder)

**Chosen approach:** Flatten visits into single sequence

**Rationale:**
- ✅ Single transformer (simpler architecture)
- ✅ Full cross-visit attention
- ✅ Compatible with BEHRT pre-training
- ✅ Simpler batching logic
- ⚠️ Requires careful attention masking

**See also:** `01b_ehr_tokens_tensors.md` for detailed explanation

### Why Mean Pooling for Visits?

**Alternatives:** Attention-based pooling, max pooling, CLS token

**Chosen approach:** Mean pooling via scatter_add

**Rationale:**
- ✅ Fast and differentiable
- ✅ Treats all codes equally (democratic)
- ✅ Stable gradients
- ⚠️ May lose fine-grained information
- ⚠️ Sensitive to visit size variation

**Future work:** Attention-based visit pooling for improved representation

### Why Multiple Loss Functions?

**Philosophy:** Different losses optimize for different objectives

| Loss | What It Optimizes | Evaluation Metric |
|------|------------------|------------------|
| NLL | Probability calibration | Brier score, calibration curves |
| Ranking | Risk discrimination | C-index, AUC |
| Hybrid | Both calibration and discrimination | All metrics |

**Recommendation:** Start with NLL, add ranking if C-index is critical, tune λ_rank carefully.

---

## 7. Implementation Checklist

### For Training

- [ ] Choose pre-training strategy (MLM, next-visit, or both)
- [ ] Select survival loss (NLL, ranking, or hybrid)
- [ ] Choose training strategy (frozen, LoRA, or full)
- [ ] Set up proper attention masking
- [ ] Validate visit aggregation correctness
- [ ] Monitor both calibration and discrimination metrics

### For Evaluation

- [ ] Compute C-index (discrimination)
- [ ] Compute Brier score (calibration)
- [ ] Plot calibration curves
- [ ] Plot survival curves
- [ ] Check for overfitting (train vs validation gap)
- [ ] Validate on held-out test set

---

## 8. Common Pitfalls

### 1. Attention Mask Errors

**Problem:** Padding tokens contaminate representations

**Solution:** Always use attention masks correctly
```python
attention_mask = (codes != 0).long()  # 1 for real tokens, 0 for padding
```

### 2. Visit Aggregation Bugs

**Problem:** Including padding in visit representations

**Solution:** Mask padding before aggregation (see `01b_ehr_tokens_tensors.md`)

### 3. Loss Imbalance in Hybrid

**Problem:** One loss dominates, other becomes meaningless

**Solution:** 
- Monitor both loss components
- Normalize losses to similar scales
- Tune λ_rank gradually (start with 0.1-0.3)

### 4. Overfitting with Full Fine-tune

**Problem:** Model memorizes training data

**Solution:**
- Use LoRA instead
- Increase dropout
- Add more regularization
- Get more data

---

## 9. Next Steps

### Detailed Documentation

1. **`01a_visit_embeddings.md`** - Deep dive into visit ID embeddings vs aggregated embeddings
2. **`01b_ehr_tokens_tensors.md`** - How hierarchical EHR data is flattened to tensors
3. **`02_survival_losses.md`** - Mathematical derivation of survival losses (coming soon)
4. **`03_evaluation_metrics.md`** - Comprehensive guide to survival metrics (coming soon)

### Related Documentation

- **Pre-training:** `dev/models/pretrain_finetune/` - BEHRT pre-training strategies
- **LoRA:** `dev/models/pretrain_finetune/07_lora_deep_dive.md`
- **Embeddings:** `dev/models/pretrain_finetune/05_embedding_summation_and_quality_analysis.md`

---

## References

**Code:**
- `src/ehrsequencing/models/behrt.py` - BEHRT architecture
- `src/ehrsequencing/models/embeddings.py` - Embedding layers
- `src/ehrsequencing/models/behrt_survival.py` - Survival model
- `src/ehrsequencing/models/losses.py` - Survival loss functions

**Examples:**
- `examples/survival_analysis/train_lstm.py` - Training examples
- `examples/survival_analysis/train_lstm_demo.py` - Demo script

**Papers:**
- BEHRT: Li et al. (2019). "BEHRT: Transformer for Electronic Health Records"
- Transformers: Vaswani et al. (2017). "Attention is All You Need"
- LoRA: Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"

---

**Last Updated:** 2026-02-03  
**Status:** Complete and ready for use
