# Phase 3: Sequence Encoders Implementation Plan

**Last Updated:** January 29, 2026  
**Status:** Active - Modern transformer-based approach for 2026  
**Priority:** High - Pivoting from Phase 2 (Med2Vec) to modern architectures

---

## Overview

Implement transformer-based patient encoders (BEHRT, Transformer) with support for 3 model size tiers to enable local development (M1 MacBook Pro 16GB) and cloud GPU training (RunPods A40).

---

## Model Size Tiers

### Small (Local Development - M1 16GB)
**Target**: Fast iteration, debugging, architecture validation  
**Hardware**: MacBook Pro M1, 16GB RAM, MPS acceleration  
**Dataset**: 100-200 patients  
**Training time**: 10-30 minutes

**Model specifications**:
```python
SMALL_CONFIG = {
    'vocab_size': 1000,
    'embedding_dim': 64,
    'hidden_dim': 128,
    'num_layers': 2,
    'num_heads': 4,
    'dropout': 0.1,
    'max_seq_length': 50,
    'batch_size': 16,
    'parameters': ~500K
}
```

### Medium (Local/Small GPU)
**Target**: Realistic performance estimates, hyperparameter exploration  
**Hardware**: M1 (slower) or small GPU (RTX 3060, etc.)  
**Dataset**: 500-1000 patients  
**Training time**: 1-2 hours

**Model specifications**:
```python
MEDIUM_CONFIG = {
    'vocab_size': 1000,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 4,
    'num_heads': 8,
    'dropout': 0.1,
    'max_seq_length': 100,
    'batch_size': 32,
    'parameters': ~2-3M
}
```

### Large (Cloud GPU - A40)
**Target**: Production models, best performance  
**Hardware**: RunPods A40 (40GB VRAM)  
**Dataset**: 1000+ patients  
**Training time**: 2-4 hours

**Model specifications**:
```python
LARGE_CONFIG = {
    'vocab_size': 1000,
    'embedding_dim': 256,
    'hidden_dim': 512,
    'num_layers': 6,
    'num_heads': 8,
    'dropout': 0.1,
    'max_seq_length': 200,
    'batch_size': 64,
    'parameters': ~10-15M
}
```

---

## Architecture Implementations

### 1. BEHRT (BERT for EHR)

**Paper**: "BEHRT: Transformer for Electronic Health Records" (2019)  
**Key innovation**: Age + visit + position embeddings for temporal modeling

**Components**:
```
Input: [CLS] code1 code2 [SEP] code3 code4 [SEP] ...
       ↓
Age Embeddings: [0] [45] [45] [0] [46] [46] [0] ...
Visit Embeddings: [0] [1] [1] [0] [2] [2] [0] ...
Position Embeddings: [0] [1] [2] [3] [4] [5] [6] ...
       ↓
Transformer Encoder (6 layers)
       ↓
[CLS] representation → Patient embedding
```

**Files to create**:
- `src/ehrsequencing/models/behrt.py` - BEHRT architecture
- `src/ehrsequencing/models/embeddings.py` - Age/visit/position embeddings
- `examples/pretrain_finetune/train_behrt_demo.py` - Small model demo
- `examples/pretrain_finetune/train_behrt.py` - Production training
- `notebooks/03_sequence_encoders/01_behrt.ipynb` - Tutorial

### 2. Transformer Encoder

**Standard transformer encoder** with EHR-specific modifications:
- Visit-level attention (codes within visit)
- Sequence-level attention (visits across time)
- Hierarchical encoding option

**Components**:
```
Input: Patient sequence (visits × codes)
       ↓
Code Embeddings
       ↓
Visit-level Transformer (aggregate codes → visit vectors)
       ↓
Sequence-level Transformer (aggregate visits → patient vector)
       ↓
Patient embedding
```

**Files to create**:
- `src/ehrsequencing/models/transformer.py` - Transformer encoder
- `src/ehrsequencing/models/attention.py` - Attention mechanisms
- `examples/pretrain_finetune/train_transformer_demo.py` - Small model demo
- `examples/pretrain_finetune/train_transformer.py` - Production training
- `notebooks/03_sequence_encoders/02_transformer.ipynb` - Tutorial

### 3. Patient LSTM (Baseline)

**Keep existing LSTM** as baseline for comparison:
- Already implemented in `survival_lstm.py`
- Extract encoder component
- Compare with transformer models

---

## Pre-training Objectives

### 1. Masked Language Modeling (MLM)

**Objective**: Predict masked medical codes from context

```python
# Example
Input:  [CLS] 250.00 [MASK] 401.9 [SEP] 250.00 272.0 [MASK] [SEP]
Target:        272.0                              401.9

Loss: CrossEntropyLoss on masked positions
```

**Implementation**:
- Random masking: 15% of codes
- 80% replace with [MASK], 10% random code, 10% unchanged
- Predict original code from context

### 2. Next Visit Prediction (NVP)

**Objective**: Predict codes in next visit from history

```python
# Example
Input:  [CLS] visit1_codes [SEP] visit2_codes [SEP]
Target: visit3_codes

Loss: Binary cross-entropy (multi-label)
```

**Implementation**:
- Use first N-1 visits to predict visit N
- Multi-label classification (multiple codes per visit)
- Helps learn temporal patterns

### 3. Contrastive Learning (Optional)

**Objective**: Similar patients should have similar embeddings

```python
# Example
Anchor: Patient A embedding
Positive: Patient A (different time window)
Negative: Random patient B

Loss: Contrastive loss (bring anchor-positive closer, push anchor-negative apart)
```

**Implementation**:
- SimCLR-style contrastive learning
- Data augmentation: time windows, code dropout
- Learn robust patient representations

---

## Implementation Phases

### Week 1: Core Architectures (Local Development)

**Day 1-2: BEHRT Implementation**
- [ ] Create `src/ehrsequencing/models/embeddings.py`
  - Age embedding (continuous → discrete bins)
  - Visit embedding (sequential visit IDs)
  - Position embedding (standard transformer)
- [ ] Create `src/ehrsequencing/models/behrt.py`
  - BEHRT model class
  - Forward pass with temporal embeddings
  - Support for 3 size configs
- [ ] Test on small dataset (100 patients)
- [ ] Verify memory usage and speed

**Day 3-4: Transformer Encoder**
- [ ] Create `src/ehrsequencing/models/transformer.py`
  - Standard transformer encoder
  - Hierarchical option (visit-level + sequence-level)
  - Support for 3 size configs
- [ ] Create `src/ehrsequencing/models/attention.py`
  - Multi-head attention
  - Masked attention for variable-length sequences
- [ ] Test on small dataset
- [ ] Compare with BEHRT

**Day 5: Pre-training Objectives**
- [ ] Create `src/ehrsequencing/pretraining/mlm.py`
  - Masked language modeling
  - Masking strategy (15% random)
  - Loss computation
- [ ] Create `src/ehrsequencing/pretraining/nvp.py`
  - Next visit prediction
  - Multi-label classification
  - Loss computation
- [ ] Test both objectives on small dataset

### Week 2: Training Scripts & Evaluation

**Day 6-7: Training Scripts**
- [ ] Create `examples/pretrain_finetune/train_behrt_demo.py`
  - Small model, quick iteration
  - MLM pre-training
  - Minimal configuration
- [ ] Create `examples/pretrain_finetune/train_behrt.py`
  - Production script with 3 size configs
  - Early stopping, LR scheduling
  - Checkpoint saving
  - Training history logging
- [ ] Create similar scripts for Transformer

**Day 8-9: Evaluation & Visualization**
- [ ] Create `src/ehrsequencing/evaluation/embeddings.py`
  - Patient embedding extraction
  - Clustering quality (silhouette score)
  - Nearest neighbor retrieval
- [ ] Create visualization tools
  - t-SNE/UMAP projection
  - Attention weight visualization
  - Embedding space exploration
- [ ] Create `notebooks/03_sequence_encoders/01_behrt.ipynb`
  - Tutorial on BEHRT architecture
  - Pre-training walkthrough
  - Evaluation and visualization

**Day 10: Documentation**
- [ ] Create `docs/methods/sequence-encoders.md`
  - BEHRT architecture explanation
  - Transformer encoder details
  - Pre-training objectives
  - Model size recommendations
- [ ] Update `README.md` with Phase 3 progress
- [ ] Create usage examples

---

## Training Strategy

### Local Development (Small Models)

**Purpose**: Architecture validation, debugging, quick iteration

**Workflow**:
1. Implement model architecture
2. Test on 100-200 patients
3. Verify forward/backward pass works
4. Check memory usage (< 8GB)
5. Ensure training completes in < 30 minutes
6. Debug any issues

**Example command**:
```bash
python examples/pretrain_finetune/train_behrt_demo.py \
    --data_dir ~/work/loinc-predictor/data/synthea/all_cohorts/ \
    --model_size small \
    --epochs 10 \
    --batch_size 16
```

### Cloud GPU Training (Large Models)

**Purpose**: Production models, best performance

**Workflow**:
1. Validate locally first (small model)
2. Acquire RunPods A40 instance
3. Train large model on 1000+ patients
4. Pre-train with MLM + NVP
5. Evaluate patient embeddings
6. Save model for downstream tasks

**Example command**:
```bash
# On RunPods A40
python examples/pretrain_finetune/train_behrt.py \
    --data_dir /workspace/loinc-predictor/data/synthea/large_cohort_1000/ \
    --model_size large \
    --epochs 100 \
    --batch_size 64 \
    --pretraining mlm \
    --early_stopping_patience 10
```

---

## Evaluation Metrics

### Pre-training Metrics

**Masked Language Modeling**:
- MLM accuracy: % of masked codes correctly predicted
- Top-5 accuracy: True code in top 5 predictions
- Loss convergence

**Next Visit Prediction**:
- Multi-label F1 score
- Precision/Recall per code
- Loss convergence

### Embedding Quality

**Clustering**:
- Silhouette score (higher = better separation)
- Davies-Bouldin index (lower = better clustering)
- Cluster by disease, compare with ground truth

**Similarity**:
- Nearest neighbor retrieval
- Patient similarity by diagnosis
- Temporal trajectory similarity

**Visualization**:
- t-SNE/UMAP projection
- Color by disease/age/outcome
- Verify meaningful structure

---

## Downstream Tasks

After pre-training, use patient embeddings for:

### 1. Disease Progression (Phase 4)
- Use pre-trained encoder as feature extractor
- Fine-tune on survival prediction
- Compare with Phase 1.5 LSTM baseline
- **Expected improvement**: C-index 0.53 → 0.65+

### 2. Disease Subtyping (Phase 5)
- Cluster patient embeddings
- Discover disease subtypes
- Characterize phenotypes
- **Expected**: 5+ meaningful subtypes

### 3. Clinical Prediction Tasks
- Diagnosis prediction (AUC)
- Readmission prediction
- Mortality prediction
- **Expected**: AUC > 0.85

---

## Success Criteria

### Phase 3 Completion

**Technical**:
- [ ] BEHRT implemented with 3 size configs
- [ ] Transformer encoder implemented with 3 size configs
- [ ] MLM and NVP pre-training objectives working
- [ ] Small models train locally in < 30 minutes
- [ ] Large models train on A40 in < 4 hours
- [ ] Patient embeddings show meaningful structure (silhouette > 0.4)

**Deliverables**:
- [ ] Model implementations in `src/ehrsequencing/models/`
- [ ] Training scripts in `examples/pretrain_finetune/`
- [ ] Tutorial notebook in `notebooks/03_sequence_encoders/`
- [ ] Documentation in `docs/methods/`
- [ ] Pre-trained models saved for downstream tasks

**Performance**:
- [ ] MLM accuracy > 40% (random = 0.1%)
- [ ] NVP F1 score > 0.3
- [ ] Patient embeddings cluster by disease
- [ ] Better than Phase 1.5 LSTM baseline

---

## Cost Estimation

### Local Development (Free)
- Model implementation: 5-7 days
- Small model training: Unlimited iterations
- Architecture validation: Free

### Cloud GPU Training (RunPods A40 @ $0.40/hr)

| Task | Sessions | Hours | Cost |
|------|----------|-------|------|
| BEHRT pre-training (MLM) | 1 | 3-4 | $1.20-$1.60 |
| BEHRT pre-training (NVP) | 1 | 3-4 | $1.20-$1.60 |
| Transformer pre-training | 1 | 3-4 | $1.20-$1.60 |
| Hyperparameter sweeps | 2 | 6-8 | $2.40-$3.20 |
| **Total Phase 3** | **5** | **15-20** | **$6.00-$8.00** |

**Total budget (Phases 1.5-3)**: $7-9 (Phase 1.5) + $6-8 (Phase 3) = **$13-17**

---

## Med2Vec Baseline (Optional)

**If time permits**, implement quick Med2Vec baseline:

**Effort**: 1-2 days  
**Purpose**: Comparison with modern transformers  
**Implementation**:
- Simple skip-gram architecture
- Negative sampling
- Train on 1000 patients
- Compare embedding quality with BEHRT

**Expected result**: Med2Vec < BEHRT < Transformer (in performance)

---

## Timeline

### Week 1 (Local Development)
- Days 1-2: BEHRT implementation
- Days 3-4: Transformer implementation
- Day 5: Pre-training objectives

### Week 2 (Training & Evaluation)
- Days 6-7: Training scripts
- Days 8-9: Evaluation & visualization
- Day 10: Documentation

### Week 3 (Cloud GPU Training)
- Acquire RunPods A40
- Pre-train BEHRT (MLM + NVP)
- Pre-train Transformer
- Evaluate embeddings
- Save models for Phase 4

**Total**: 3 weeks (2 weeks local, 1 week cloud)

---

## Next Steps

**Immediate (Today)**:
1. Create directory structure for Phase 3
2. Start BEHRT implementation (`src/ehrsequencing/models/behrt.py`)
3. Implement temporal embeddings (`src/ehrsequencing/models/embeddings.py`)

**This Week**:
1. Complete BEHRT and Transformer architectures
2. Implement pre-training objectives
3. Test on small dataset locally
4. Create demo training scripts

**Next Week**:
1. Create production training scripts
2. Develop evaluation tools
3. Create tutorial notebook
4. Write documentation

**Week 3**:
1. Acquire RunPods A40
2. Pre-train large models
3. Evaluate embeddings
4. Prepare for Phase 4

---

## References

**BEHRT**:
- Paper: "BEHRT: Transformer for Electronic Health Records" (Li et al., 2019)
- Key idea: Age + visit + position embeddings for temporal EHR modeling

**Transformer**:
- Paper: "Attention is All You Need" (Vaswani et al., 2017)
- Application to EHR: Hierarchical attention for visit/sequence levels

**Pre-training**:
- MLM: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- Contrastive: "A Simple Framework for Contrastive Learning" (Chen et al., 2020)

---

**Document Version:** 1.0  
**Next Review:** After Week 1 (architecture implementation)  
**Maintained By:** EHR Sequencing Team
