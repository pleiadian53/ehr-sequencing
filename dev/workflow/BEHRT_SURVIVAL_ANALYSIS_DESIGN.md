# BEHRT for Survival Analysis: Design Document

## Current State: BEHRT (MLM Pre-training Only)

✅ **What BEHRT Currently Does:**
1. **Bidirectional Transformer** (like BERT)
2. **MLM Pre-training** - Predicts masked medical codes
3. **Self-supervised Learning** - Learns from EHR sequences without labels
4. **Contextualized Representations** - Rich embeddings capturing temporal context
5. **Model Class:** `BEHRTForMLM` in `src/ehrsequencing/models/behrt.py`

## What's Needed: BEHRT for Survival Analysis

To use BEHRT for survival analysis (readmission, mortality, disease onset), we need to add a **survival prediction head** on top of the pre-trained BEHRT encoder.

---

## Architecture Comparison

### Current LSTM Approach (`DiscreteTimeSurvivalLSTM`)

```
Input: Visit Codes [batch, num_visits, codes_per_visit]
  ↓
Code Embeddings (learned from scratch)
  ↓
Mean Pooling per Visit
  ↓
LSTM over Visits
  ↓
Hazard Head (Linear + Sigmoid)
  ↓
Output: Hazards [batch, num_visits]
```

**Key characteristics:**
- Learns embeddings from scratch for survival task
- Unidirectional (causal) - only sees past visits
- Simple architecture, fast training
- No pre-training phase

---

### Proposed BEHRT Approach (`BEHRTForSurvival`)

```
Input: Visit Codes [batch, seq_length]
  ↓
BEHRT Encoder (pre-trained with MLM)
  ├─ Code Embeddings (pre-trained)
  ├─ Age Embeddings
  ├─ Visit Embeddings
  ├─ Segment Embeddings
  └─ Bidirectional Transformer Layers
  ↓
Contextualized Representations [batch, seq_length, hidden_dim]
  ↓
Visit-level Aggregation (pool codes within each visit)
  ↓
Survival Prediction Head (Linear + Sigmoid)
  ↓
Output: Hazards [batch, num_visits]
```

**Key characteristics:**
- Uses pre-trained BEHRT embeddings (transfer learning)
- Bidirectional context (sees past AND future within sequence)
- Richer representations (age, visit, segment info)
- Two-stage training: pre-train MLM → fine-tune survival

---

## Implementation Requirements

### 1. Create `BEHRTForSurvival` Model

**File:** `src/ehrsequencing/models/behrt_survival.py`

**Architecture:**
```python
class BEHRTForSurvival(nn.Module):
    """
    BEHRT model adapted for discrete-time survival analysis.
    
    Uses pre-trained BEHRT encoder + survival prediction head.
    """
    
    def __init__(self, behrt_encoder, hidden_dim, dropout=0.1):
        super().__init__()
        
        # Pre-trained BEHRT encoder
        self.behrt = behrt_encoder
        
        # Survival prediction head
        self.survival_head = nn.Sequential(
            nn.Linear(behrt_encoder.config.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Hazard in (0, 1)
        )
    
    def forward(self, codes, ages, visit_ids, attention_mask, visit_boundaries):
        """
        Args:
            codes: [batch, seq_length] - flattened codes across visits
            ages: [batch, seq_length] - age at each code
            visit_ids: [batch, seq_length] - visit index for each code
            attention_mask: [batch, seq_length] - valid positions
            visit_boundaries: [batch, num_visits, 2] - (start, end) indices for each visit
        
        Returns:
            hazards: [batch, num_visits] - hazard at each visit
        """
        # Get BEHRT representations
        hidden_states = self.behrt(
            codes, ages=ages, visit_ids=visit_ids, 
            attention_mask=attention_mask
        )  # [batch, seq_length, hidden_dim]
        
        # Aggregate codes within each visit (mean pooling)
        visit_representations = self._aggregate_visits(
            hidden_states, visit_boundaries
        )  # [batch, num_visits, hidden_dim]
        
        # Predict hazards
        hazards = self.survival_head(visit_representations)  # [batch, num_visits, 1]
        hazards = hazards.squeeze(-1)  # [batch, num_visits]
        
        return hazards
```

---

### 2. Visit Aggregation Strategy

**Challenge:** BEHRT operates on flattened code sequences, but survival analysis needs visit-level predictions.

**Solution:** Aggregate BEHRT representations within visit boundaries.

```python
def _aggregate_visits(self, hidden_states, visit_boundaries):
    """
    Aggregate code-level representations to visit-level.
    
    Args:
        hidden_states: [batch, seq_length, hidden_dim]
        visit_boundaries: [batch, num_visits, 2] - (start, end) for each visit
    
    Returns:
        visit_reps: [batch, num_visits, hidden_dim]
    """
    batch_size, num_visits, _ = visit_boundaries.shape
    hidden_dim = hidden_states.shape[-1]
    
    visit_reps = []
    for i in range(batch_size):
        patient_visits = []
        for j in range(num_visits):
            start, end = visit_boundaries[i, j]
            if end > start:  # Valid visit
                # Mean pool codes in this visit
                visit_codes = hidden_states[i, start:end, :]
                visit_rep = visit_codes.mean(dim=0)
            else:  # Padding visit
                visit_rep = torch.zeros(hidden_dim, device=hidden_states.device)
            patient_visits.append(visit_rep)
        visit_reps.append(torch.stack(patient_visits))
    
    return torch.stack(visit_reps)
```

---

### 3. Training Strategy

**Two-Stage Approach (Recommended):**

#### Stage 1: Pre-train BEHRT with MLM
```bash
# Already done with train_behrt_demo.py
python examples/pretrain_finetune/train_behrt_demo.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --realistic-data
```

**Output:** Pre-trained BEHRT checkpoint with learned embeddings

#### Stage 2: Fine-tune for Survival
```bash
# New script: train_behrt_survival.py
python examples/survival_analysis/train_behrt_survival.py \
    --pretrained-behrt checkpoints/behrt_large_mlm/best_model.pt \
    --task readmission \
    --freeze-encoder  # Optional: freeze BEHRT, train only survival head
    --epochs 50
```

**Training Options:**

1. **Freeze BEHRT, train only survival head** (fast, less overfitting)
   ```python
   for param in model.behrt.parameters():
       param.requires_grad = False
   ```

2. **Fine-tune entire model** (slower, potentially better performance)
   ```python
   # All parameters trainable
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
   ```

3. **Use LoRA for efficient fine-tuning** (best of both worlds)
   ```python
   model.behrt = apply_lora_to_behrt(model.behrt, rank=16)
   ```

---

### 4. Data Preparation

**Challenge:** BEHRT expects flattened sequences, survival LSTM expects visit-grouped data.

**Solution:** Create adapter to convert between formats.

```python
class BEHRTSurvivalDataset(Dataset):
    """
    Dataset for BEHRT survival analysis.
    
    Converts visit-grouped data to BEHRT format while preserving visit boundaries.
    """
    
    def __init__(self, visit_sequences, survival_labels):
        self.visit_sequences = visit_sequences
        self.survival_labels = survival_labels
    
    def __getitem__(self, idx):
        visits = self.visit_sequences[idx]  # List of visits
        
        # Flatten codes across visits
        codes = []
        ages = []
        visit_ids = []
        visit_boundaries = []
        
        current_pos = 0
        for visit_idx, visit in enumerate(visits):
            visit_codes = visit['codes']
            visit_age = visit['age']
            
            start = current_pos
            end = current_pos + len(visit_codes)
            visit_boundaries.append([start, end])
            
            codes.extend(visit_codes)
            ages.extend([visit_age] * len(visit_codes))
            visit_ids.extend([visit_idx] * len(visit_codes))
            
            current_pos = end
        
        return {
            'codes': torch.tensor(codes),
            'ages': torch.tensor(ages),
            'visit_ids': torch.tensor(visit_ids),
            'visit_boundaries': torch.tensor(visit_boundaries),
            'hazards': torch.tensor(self.survival_labels[idx]),
            'event_indicator': torch.tensor(self.survival_labels[idx]['event'])
        }
```

---

## Implementation Plan

### Phase 1: Core Model (Week 1)

**Files to Create:**
1. `src/ehrsequencing/models/behrt_survival.py` - BEHRTForSurvival model
2. `src/ehrsequencing/data/behrt_survival_dataset.py` - Data adapter
3. `examples/survival_analysis/train_behrt_survival.py` - Training script

**Tasks:**
- [ ] Implement BEHRTForSurvival class
- [ ] Implement visit aggregation logic
- [ ] Create data adapter for BEHRT format
- [ ] Add survival head with proper initialization

---

### Phase 2: Training Pipeline (Week 2)

**Files to Create:**
1. `examples/survival_analysis/train_behrt_survival.py` - Main training script
2. `examples/survival_analysis/compare_behrt_lstm.py` - Comparison script

**Tasks:**
- [ ] Load pre-trained BEHRT checkpoint
- [ ] Implement fine-tuning loop
- [ ] Add LoRA support for efficient fine-tuning
- [ ] Implement evaluation metrics (C-index, calibration)
- [ ] Create comparison framework with LSTM baseline

---

### Phase 3: Benchmarking (Week 3)

**Files to Create:**
1. `examples/survival_analysis/benchmark_survival_models.py` - Comprehensive comparison

**Tasks:**
- [ ] Compare BEHRT vs LSTM on readmission task
- [ ] Compare BEHRT vs LSTM on mortality task
- [ ] Ablation study: pre-training vs from-scratch
- [ ] Ablation study: frozen vs fine-tuned BEHRT
- [ ] Generate comparison reports

---

## Expected Performance Gains

### Why BEHRT Should Outperform LSTM:

1. **Pre-trained Representations**
   - BEHRT learns from large unlabeled EHR data (MLM)
   - LSTM learns embeddings from scratch on small survival dataset
   - **Expected gain:** 10-20% improvement in C-index

2. **Bidirectional Context**
   - BEHRT sees full sequence context (past + future)
   - LSTM only sees past (causal)
   - **Expected gain:** Better representation quality

3. **EHR-Specific Features**
   - BEHRT uses age, visit, segment embeddings
   - LSTM only uses code embeddings
   - **Expected gain:** Better temporal modeling

4. **Transfer Learning**
   - BEHRT transfers knowledge from MLM pre-training
   - LSTM starts from random initialization
   - **Expected gain:** Faster convergence, less overfitting

---

## Comparison Framework

### Metrics to Compare:

1. **Discrimination:** C-index (concordance index)
2. **Calibration:** Calibration curves, Brier score
3. **Time-dependent Performance:** AUC at 7, 14, 30 days
4. **Efficiency:** Training time, parameter count
5. **Generalization:** Train-val gap

### Experimental Setup:

```python
# Baseline: LSTM from scratch
lstm_model = DiscreteTimeSurvivalLSTM(
    vocab_size=5000,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2
)

# BEHRT Variant 1: Frozen encoder
behrt_frozen = BEHRTForSurvival(
    behrt_encoder=pretrained_behrt,
    hidden_dim=256
)
for param in behrt_frozen.behrt.parameters():
    param.requires_grad = False

# BEHRT Variant 2: Fine-tuned with LoRA
behrt_lora = BEHRTForSurvival(
    behrt_encoder=apply_lora_to_behrt(pretrained_behrt, rank=16),
    hidden_dim=256
)

# BEHRT Variant 3: Full fine-tuning
behrt_full = BEHRTForSurvival(
    behrt_encoder=pretrained_behrt,
    hidden_dim=256
)

# Compare all models
results = compare_survival_models(
    models=[lstm_model, behrt_frozen, behrt_lora, behrt_full],
    train_data=train_data,
    val_data=val_data,
    test_data=test_data
)
```

---

## Key Design Decisions

### 1. Bidirectional vs Causal

**Question:** BEHRT is bidirectional, but survival analysis is causal (predict future from past). Is this a problem?

**Answer:** No, because:
- We only use BEHRT to generate **representations** of past visits
- The survival head makes **causal predictions** (hazard at visit t uses data through visit t)
- Bidirectional context helps learn better representations, but prediction is still causal

**Implementation:**
```python
# At prediction time for visit t:
# 1. Use BEHRT to encode all codes through visit t (bidirectional context within visit t)
# 2. Aggregate to visit-level representation
# 3. Predict hazard at visit t using only data through visit t
```

### 2. Visit Aggregation Method

**Options:**
1. **Mean pooling** (simple, works well)
2. **Max pooling** (captures salient features)
3. **Attention pooling** (learnable, more expressive)
4. **CLS token** (add special token at visit boundaries)

**Recommendation:** Start with mean pooling, add attention if needed.

### 3. Pre-training Task Alignment

**Question:** MLM pre-training predicts masked codes. Does this help survival prediction?

**Answer:** Yes, because:
- MLM forces BEHRT to learn semantic relationships between codes
- These relationships are useful for predicting outcomes
- Similar to how BERT pre-training helps downstream NLP tasks

---

## Success Criteria

### Minimum Viable Product (MVP):

- [ ] BEHRTForSurvival model implemented and tested
- [ ] Training script runs without errors
- [ ] Model achieves C-index ≥ LSTM baseline on readmission task
- [ ] Comparison report generated

### Stretch Goals:

- [ ] BEHRT outperforms LSTM by ≥10% C-index
- [ ] BEHRT shows better calibration than LSTM
- [ ] BEHRT generalizes better (smaller train-val gap)
- [ ] LoRA fine-tuning matches full fine-tuning performance
- [ ] Comprehensive ablation study completed

---

## Next Steps

1. **Implement BEHRTForSurvival model** (1-2 days)
2. **Create data adapter** (1 day)
3. **Write training script** (1-2 days)
4. **Run initial experiments** (1 day)
5. **Compare with LSTM baseline** (1 day)
6. **Iterate and optimize** (ongoing)

---

## References

**BEHRT Paper:**
- Li et al. (2020). "BEHRT: Transformer for Electronic Health Records"
- Key insight: Bidirectional pre-training learns better EHR representations

**Survival Analysis:**
- Your current LSTM implementation in `examples/survival_analysis/`
- Discrete-time survival loss and C-index metrics

**Transfer Learning:**
- Howard & Ruder (2018). "Universal Language Model Fine-tuning for Text Classification"
- Shows pre-training + fine-tuning outperforms training from scratch
