# Discrete-Time Survival LSTM: Architecture Walkthrough

**Date:** January 22, 2026  
**Source:** `src/ehrsequencing/models/survival_lstm.py`  
**Purpose:** Understanding the LSTM architecture for discrete-time survival analysis

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Forward Pass Walkthrough](#forward-pass-walkthrough)
4. [Comparison with LSTMBaseline](#comparison-with-lstmbaseline)
5. [Integration with VisitEncoder](#integration-with-visitencoder)
6. [Extensions and Variants](#extensions-and-variants)

---

## Overview

The `DiscreteTimeSurvivalLSTM` is a specialized LSTM architecture for modeling time-to-event outcomes (disease progression, readmission, death) from visit-grouped EHR sequences.

### Key Differences from Standard Classification

| Aspect | Standard Classification | Survival Analysis |
|--------|------------------------|-------------------|
| **Output** | Single prediction per patient | Hazard at **each visit** |
| **Label** | Binary (0/1) or multi-class | Event time + censoring indicator |
| **Loss** | Cross-entropy | Survival likelihood |
| **Handles censoring** | No | Yes (explicitly) |
| **Temporal** | Often static | Inherently temporal |

---

## Architecture Components

### Full Architecture Diagram

```
Patient EHR Sequence:
┌───────────────────────────────────────────────────────────┐
│ Visit 1          Visit 2          Visit 3                 │
│ [Code A,         [Code B,         [Code C,                │
│  Code B,          Code D,          Code E,                │
│  Code C]          Code E]          Code F]                │
└───────────────────────────────────────────────────────────┘
        ↓ (1) Code Embedding
┌───────────────────────────────────────────────────────────┐
│ [[emb_A, emb_B, emb_C],                                   │
│  [emb_B, emb_D, emb_E],                                   │
│  [emb_C, emb_E, emb_F]]                                   │
└───────────────────────────────────────────────────────────┘
        ↓ (2) Visit Aggregation (Mean Pooling)
┌───────────────────────────────────────────────────────────┐
│ [visit_vec_1, visit_vec_2, visit_vec_3]                   │
│  [128-dim]      [128-dim]      [128-dim]                  │
└───────────────────────────────────────────────────────────┘
        ↓ (3) LSTM (Temporal Modeling)
┌───────────────────────────────────────────────────────────┐
│ [hidden_1, hidden_2, hidden_3]                            │
│  [256-dim]  [256-dim]  [256-dim]                          │
└───────────────────────────────────────────────────────────┘
        ↓ (4) Hazard Head (Per-Visit Prediction)
┌───────────────────────────────────────────────────────────┐
│ [h_1, h_2, h_3]   ← Hazards in (0, 1)                     │
│  0.05  0.12  0.28 ← Increasing risk over time             │
└───────────────────────────────────────────────────────────┘
```

---

### Component 1: Code Embedding

```python
self.embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=0
)
```

**Purpose:** Map discrete medical codes to continuous vector space

**Example:**
```python
vocab_size = 10000  # 10K unique medical codes
embedding_dim = 128

# Input: code indices
visit_codes = torch.tensor([[42, 523, 1200, 0]])  # 3 codes + padding
# Shape: [1, 1, 4]  (batch=1, visits=1, codes=4)

# Output: embeddings
code_embeddings = embedding(visit_codes)
# Shape: [1, 1, 4, 128]

# Padding (index 0) gets zero vector
code_embeddings[0, 0, 3] = [0, 0, ..., 0]  # All zeros
```

**Why `padding_idx=0`?**
- Ensures padding tokens don't contribute to visit representation
- Zero embeddings are ignored in mean pooling

---

### Component 2: Visit Aggregation (Mean Pooling)

```python
# Expand mask for broadcasting: [B, V, C, 1]
visit_mask_expanded = visit_mask.unsqueeze(-1).float()

# Sum embeddings and normalize by number of codes
visit_vectors = (embeddings * visit_mask_expanded).sum(dim=2)  # [B, V, E]
num_codes_per_visit = visit_mask.sum(dim=2, keepdim=True).clamp(min=1)  # [B, V, 1]
visit_vectors = visit_vectors / num_codes_per_visit  # [B, V, E]
```

**What this does:**
- Aggregates variable number of codes per visit into fixed-size visit vector
- Uses **masked mean** to handle padding correctly

**Step-by-step:**

```python
# Visit with 3 real codes + 1 padding
visit_codes = [[42, 523, 1200, 0]]  # Indices
code_embeddings = [
    [0.5, -0.3, 0.8, ..., 0.2],   # Code 42
    [0.3, 0.6, -0.2, ..., 0.5],   # Code 523
    [0.7, 0.1, 0.4, ..., -0.1],   # Code 1200
    [0.0, 0.0, 0.0, ..., 0.0]     # Padding
]
visit_mask = [[1, 1, 1, 0]]  # 3 real, 1 padding

# Step 1: Zero out padding
masked = embeddings * visit_mask_expanded
# Padding row becomes all zeros

# Step 2: Sum
sum_embeddings = masked.sum(dim=2)
# = emb_42 + emb_523 + emb_1200 + [0, 0, ..., 0]

# Step 3: Count real codes
count = visit_mask.sum() = 3

# Step 4: Average
visit_vector = sum_embeddings / 3
# = mean([emb_42, emb_523, emb_1200])
```

**Why mean pooling?**
- Simple and effective
- Permutation invariant (code order doesn't matter)
- Handles variable visit sizes

---

### Component 3: LSTM (Temporal Modeling)

```python
self.lstm = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    num_layers=num_layers,
    batch_first=True,
    dropout=dropout if num_layers > 1 else 0,
    bidirectional=False  # Causal!
)
```

**Purpose:** Model temporal dependencies across visits

**Input/Output:**
```python
# Input: visit vectors
visit_vectors: [batch, num_visits, embedding_dim]
               [32, 10, 128]

# Output: hidden states at each visit
lstm_out: [batch, num_visits, hidden_dim]
          [32, 10, 256]

# Each hidden state encodes history up to that visit
lstm_out[:, 0, :] = patient state after visit 1
lstm_out[:, 1, :] = patient state after visit 2
...
lstm_out[:, 9, :] = patient state after visit 10
```

**Why `bidirectional=False`?**
- **Causal modeling:** Hazard at visit $t$ should only use data through visit $t$
- Bidirectional would "cheat" by seeing future visits
- Critical for deployment: can't predict based on future data

---

### Component 4: Hazard Prediction Head

```python
self.hazard_head = nn.Sequential(
    nn.Linear(lstm_output_dim, hidden_dim),  # 256 → 256
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 1),                # 256 → 1
    nn.Sigmoid()  # Ensure hazard in (0, 1)
)
```

**Purpose:** Map LSTM hidden states to hazard probabilities

**Architecture breakdown:**
```
Input: LSTM hidden state [256]
    ↓
Linear: 256 → 256  (expressiveness)
    ↓
ReLU: nonlinearity
    ↓
Dropout: regularization (0.1)
    ↓
Linear: 256 → 1  (single hazard value)
    ↓
Sigmoid: ensure output in (0, 1)
    ↓
Output: hazard h_t ∈ (0, 1)
```

**Why two layers?**
- First layer: Learns hazard-relevant features from hidden state
- Second layer: Maps to single probability
- More expressive than direct `Linear(256, 1)`

**Why Sigmoid?**
```python
# Hazard must be valid probability
0 < h_t < 1

# Sigmoid ensures this:
sigmoid(x) = 1 / (1 + exp(-x))

# Maps (-∞, ∞) → (0, 1)
```

**Output per visit:**
```python
hazards = model(visit_sequences)
# Shape: [32, 10]

# Example for patient 0:
hazards[0] = [0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
#             └────────────────────┬──────────────────────┘
#                  Increasing risk over time
```

---

## Forward Pass Walkthrough

### Complete Example

```python
# Setup
batch_size = 2
num_visits = 3
max_codes_per_visit = 4
vocab_size = 10000
embedding_dim = 128
hidden_dim = 256

model = DiscreteTimeSurvivalLSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)

# Input data
visit_codes = torch.tensor([
    # Patient 1
    [[42, 523, 1200, 0],      # Visit 1: 3 codes
     [42, 523, 1500, 2000],   # Visit 2: 4 codes
     [100, 200, 0, 0]],       # Visit 3: 2 codes
    # Patient 2
    [[50, 60, 0, 0],          # Visit 1: 2 codes
     [70, 80, 90, 0],         # Visit 2: 3 codes
     [0, 0, 0, 0]]            # Visit 3: padding
])
# Shape: [2, 3, 4]

visit_mask = torch.tensor([
    [[1, 1, 1, 0],
     [1, 1, 1, 1],
     [1, 1, 0, 0]],
    [[1, 1, 0, 0],
     [1, 1, 1, 0],
     [0, 0, 0, 0]]
])
# Shape: [2, 3, 4]

sequence_mask = torch.tensor([
    [1, 1, 1],  # Patient 1: 3 real visits
    [1, 1, 0]   # Patient 2: 2 real visits (last is padding)
])
# Shape: [2, 3]
```

---

### Step 1: Embed Codes

```python
embeddings = model.embedding(visit_codes)
# Shape: [2, 3, 4, 128]
#        [batch, visits, codes, embed_dim]

# Each code gets 128-dim embedding
# Padding (code 0) gets zero vector
```

---

### Step 2: Aggregate Codes Within Visits

```python
# Patient 1, Visit 1
visit_embeddings = embeddings[0, 0, :, :]  # [4, 128]
# [emb_42, emb_523, emb_1200, zeros]

visit_mask_for_this_visit = [1, 1, 1, 0]

# Masked mean
visit_vector = (visit_embeddings * visit_mask).sum(dim=0) / 3
# = mean([emb_42, emb_523, emb_1200])
# Shape: [128]

# After processing all visits:
visit_vectors = [
    # Patient 1
    [visit_vec_1_1, visit_vec_1_2, visit_vec_1_3],  # 3 visits
    # Patient 2
    [visit_vec_2_1, visit_vec_2_2, zeros]  # 2 visits + padding
]
# Shape: [2, 3, 128]
```

---

### Step 3: LSTM Processing

```python
lstm_out, (h_n, c_n) = model.lstm(visit_vectors)
# lstm_out: [2, 3, 256]

# Patient 1:
# lstm_out[0, 0] = hidden state after processing visit 1
# lstm_out[0, 1] = hidden state after processing visits 1-2
# lstm_out[0, 2] = hidden state after processing visits 1-3

# Patient 2:
# lstm_out[1, 0] = hidden state after visit 1
# lstm_out[1, 1] = hidden state after visit 2
# lstm_out[1, 2] = zeros (padding visit, but still computed)
```

---

### Step 4: Hazard Prediction

```python
hazards = model.hazard_head(lstm_out).squeeze(-1)
# Shape: [2, 3]

# Before masking:
hazards = [
    [0.05, 0.12, 0.28],  # Patient 1: increasing risk
    [0.08, 0.15, 0.03]   # Patient 2: hazard at padding visit (will be masked)
]

# After masking by sequence_mask:
hazards = hazards * sequence_mask
hazards = [
    [0.05, 0.12, 0.28],  # Patient 1: all real
    [0.08, 0.15, 0.00]   # Patient 2: last visit zeroed out
]
```

---

### Step 5: Interpretation

**Patient 1 hazards:** `[0.05, 0.12, 0.28]`

- Visit 1: 5% chance of event by next visit
- Visit 2: 12% chance of event by next visit (risk increasing)
- Visit 3: 28% chance of event by next visit (risk accelerating)

**Survival probability:**
```python
S_1 = (1 - 0.05) = 0.95
S_2 = (1 - 0.05) × (1 - 0.12) = 0.95 × 0.88 = 0.836
S_3 = 0.836 × (1 - 0.28) = 0.602

# Patient has ~60% chance of being event-free through visit 3
```

---

## Comparison with LSTMBaseline

### Similarities

| Component | LSTMBaseline | SurvivalLSTM |
|-----------|--------------|--------------|
| **Embedding** | ✅ Same | ✅ Same |
| **Visit aggregation** | ✅ Mean/attention | ✅ Mean only |
| **LSTM** | ✅ Same architecture | ✅ Same architecture |
| **Bidirectional** | Optional (default False) | Must be False |

---

### Key Differences

#### 1. Output Head

**LSTMBaseline:**
```python
# Single prediction per patient (uses final hidden state)
final_hidden = hidden[-1]  # Last LSTM state
logits = self.fc(final_hidden)  # [batch, output_dim]
predictions = self.activation(logits)  # [batch, 1] or [batch, K]
```

**SurvivalLSTM:**
```python
# Hazard at EVERY visit
hazards = self.hazard_head(lstm_out)  # [batch, num_visits]
# Every visit gets a prediction
```

**Visual comparison:**
```
LSTMBaseline:
Visit 1 ──> Visit 2 ──> Visit 3 ──> [FINAL PREDICTION]
                                     ↑
                                     One output

SurvivalLSTM:
Visit 1 ──> Visit 2 ──> Visit 3
  ↓           ↓           ↓
 h_1         h_2         h_3
  ↓           ↓           ↓
Three outputs (hazards at each visit)
```

---

#### 2. Loss Function

**LSTMBaseline:**
```python
# Standard cross-entropy or MSE
loss = nn.CrossEntropyLoss()(predictions, labels)
# Labels: [batch] - one label per patient
```

**SurvivalLSTM:**
```python
# Survival likelihood
loss = DiscreteTimeSurvivalLoss()(
    hazards,           # [batch, num_visits]
    event_times,       # [batch] - time of event/censoring
    event_indicators,  # [batch] - 1=event, 0=censored
    sequence_mask      # [batch, num_visits]
)
```

---

#### 3. Training Data

**LSTMBaseline:**
```python
# Simple labels
data = {
    'visit_sequences': [...],
    'label': 1  # Binary: did event occur?
}
```

**SurvivalLSTM:**
```python
# Survival data with censoring
data = {
    'visit_sequences': [...],
    'event_time': 5,        # Event at visit 5
    'event_indicator': 1,   # 1 = observed event
                             # 0 = censored (unknown after last visit)
}
```

---

#### 4. Prediction at Inference

**LSTMBaseline:**
```python
# One prediction
prediction = model(sequence)  # Scalar or vector
# "Will this patient progress?"
```

**SurvivalLSTM:**
```python
# Hazard at each visit
hazards = model(sequence)  # [num_visits]
# "What is risk at each visit?"

# Can compute survival curve
survival_probs = model.predict_survival(sequence)
# S_t = P(no event through visit t)
```

---

## Integration with VisitEncoder

**Your observation is correct!** The survival LSTM uses simple mean pooling, but could benefit from the more sophisticated `VisitEncoder` from `lstm_baseline.py`.

### Current Implementation (Mean Pooling)

```python
# survival_lstm.py (lines 117-124)
visit_mask_expanded = visit_mask.unsqueeze(-1).float()
visit_vectors = (embeddings * visit_mask_expanded).sum(dim=2)
num_codes_per_visit = visit_mask.sum(dim=2, keepdim=True).clamp(min=1)
visit_vectors = visit_vectors / num_codes_per_visit
```

**Limitations:**
- All codes weighted equally
- No learned importance weighting
- Fixed aggregation strategy

---

### Improved Implementation with VisitEncoder

```python
from ehrsequencing.models.lstm_baseline import VisitEncoder

class DiscreteTimeSurvivalLSTMWithAttention(nn.Module):
    """
    Survival LSTM with attention-based visit encoding.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        visit_aggregation: str = 'attention',  # 'mean', 'attention', 'max'
    ):
        super().__init__()
        
        # Code embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Visit encoder (with attention!)
        self.visit_encoder = VisitEncoder(
            embedding_dim=embedding_dim,
            aggregation=visit_aggregation,  # Can use attention
            dropout=dropout
        )
        
        # LSTM (same as before)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Hazard head (same as before)
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, visit_codes, visit_mask, sequence_mask):
        batch_size, num_visits, max_codes = visit_codes.shape
        
        # Embed codes: [B, V, C, E]
        embeddings = self.embedding(visit_codes)
        
        # Reshape for visit encoder: [B*V, C, E]
        embeddings_flat = embeddings.view(batch_size * num_visits, max_codes, embedding_dim)
        visit_mask_flat = visit_mask.view(batch_size * num_visits, max_codes)
        
        # Encode visits with attention: [B*V, E]
        visit_vectors = self.visit_encoder(embeddings_flat, visit_mask_flat)
        
        # Reshape back: [B, V, E]
        visit_vectors = visit_vectors.view(batch_size, num_visits, embedding_dim)
        
        # LSTM (same as before)
        lstm_out, _ = self.lstm(visit_vectors)
        
        # Hazard prediction (same as before)
        hazards = self.hazard_head(lstm_out).squeeze(-1)
        hazards = hazards * sequence_mask.float()
        
        return hazards
```

---

### Benefits of VisitEncoder Integration

#### 1. Learned Code Importance

**With attention:**
```python
# Model learns which codes are important for survival
Visit with codes: [Diabetes, HbA1c, Metformin refill]

# Attention weights (learned):
weights = [0.60, 0.35, 0.05]
#          ↑     ↑      ↑
#       Important Moderately Not very
#       (diagnosis) (lab) (routine med)

visit_vector = 0.60*emb_diabetes + 0.35*emb_HbA1c + 0.05*emb_metformin
```

**With mean pooling:**
```python
# All codes weighted equally
weights = [0.33, 0.33, 0.33]  # Uniform
```

---

#### 2. Flexible Aggregation Strategies

```python
# Can choose aggregation method
model = DiscreteTimeSurvivalLSTMWithAttention(
    aggregation='attention'  # or 'mean', 'max', 'sum'
)

# Experiment to see what works best for your data
```

---

#### 3. Interpretability

**Attention weights reveal what the model focuses on:**

```python
# Extract attention weights
attention_weights = model.visit_encoder.attention(code_embeddings)
# Shape: [batch, codes, 1]

# For patient who progressed:
# Attention might focus on:
# - Diagnostic codes (high weights)
# - Abnormal lab values (high weights)
# - Routine visits (low weights)

# For stable patient:
# - All codes get moderate weights (no red flags)
```

---

### Comparison: Mean vs. Attention

**Experiment setup:**
```python
# Train two models
model_mean = DiscreteTimeSurvivalLSTM(aggregation='mean')
model_attn = DiscreteTimeSurvivalLSTMWithAttention(aggregation='attention')

# Compare on same data
results = {
    'Mean Pooling': {
        'C-index': 0.72,
        'AUC': 0.75
    },
    'Attention': {
        'C-index': 0.76,  # Better!
        'AUC': 0.78       # Better!
    }
}
```

**When attention helps most:**
- Complex visits with many codes (20+)
- Heterogeneous code types (diagnoses + labs + meds)
- Sparse coding (many irrelevant codes)

**When mean pooling suffices:**
- Simple visits (3-5 codes)
- Homogeneous coding (all diagnoses)
- Small datasets (attention might overfit)

---

## Extensions and Variants

### 1. Time-Aware Survival LSTM

**Motivation:** Visit spacing matters (weekly visits vs. yearly visits)

```python
class DiscreteTimeSurvivalLSTMWithTime(DiscreteTimeSurvivalLSTM):
    """Incorporates time features."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, time_encoding_dim=16):
        super().__init__(vocab_size, embedding_dim, hidden_dim)
        
        # Time feature encoder
        self.time_encoder = nn.Sequential(
            nn.Linear(3, time_encoding_dim),  # 3 time features
            nn.ReLU(),
            nn.Linear(time_encoding_dim, time_encoding_dim)
        )
        
        # Update LSTM to accept time features
        self.lstm = nn.LSTM(
            input_size=embedding_dim + time_encoding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
    
    def forward(self, visit_codes, visit_mask, sequence_mask, time_features):
        # time_features: [batch, num_visits, 3]
        # Features: [days_since_last_visit, days_since_first_visit, visit_index]
        
        # Encode visits (same as before)
        visit_vectors = self.encode_visits(visit_codes, visit_mask)
        
        # Encode time
        time_encoded = self.time_encoder(time_features)
        
        # Concatenate
        lstm_input = torch.cat([visit_vectors, time_encoded], dim=-1)
        
        # LSTM with time info
        lstm_out, _ = self.lstm(lstm_input)
        
        # Hazards
        hazards = self.hazard_head(lstm_out).squeeze(-1)
        return hazards * sequence_mask.float()
```

**Time features:**
```python
# Patient visits at: day 0, 90, 180, 200
time_features = [
    [0, 0, 0],       # Visit 1: (delta=0, total=0, index=0)
    [90, 90, 1],     # Visit 2: (delta=90, total=90, index=1)
    [90, 180, 2],    # Visit 3: (delta=90, total=180, index=2)
    [20, 200, 3]     # Visit 4: (delta=20, total=200, index=3)
]
```

---

### 2. Competing Risks LSTM

**Motivation:** Multiple event types (progression, death, readmission)

```python
class CompetingRisksSurvivalLSTM(nn.Module):
    """
    Multiple cause-specific hazards.
    """
    def __init__(self, vocab_size, num_event_types, embedding_dim, hidden_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Separate hazard head for each event type
        self.hazard_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            for _ in range(num_event_types)
        ])
    
    def forward(self, visit_codes, visit_mask, sequence_mask):
        # Encode visits
        visit_vectors = self.encode_visits(visit_codes, visit_mask)
        
        # LSTM
        lstm_out, _ = self.lstm(visit_vectors)
        
        # Cause-specific hazards
        hazards = []
        for head in self.hazard_heads:
            h = head(lstm_out).squeeze(-1)  # [batch, visits]
            hazards.append(h)
        
        # Stack: [batch, visits, num_event_types]
        hazards = torch.stack(hazards, dim=-1)
        
        # Mask
        hazards = hazards * sequence_mask.unsqueeze(-1).float()
        
        return hazards
```

**Usage:**
```python
# Event types: [progression, death, readmission]
model = CompetingRisksSurvivalLSTM(vocab_size=10000, num_event_types=3)

hazards = model(visit_sequences)
# Shape: [batch, visits, 3]

# hazards[:, :, 0] = progression hazards
# hazards[:, :, 1] = death hazards
# hazards[:, :, 2] = readmission hazards
```

---

### 3. Hierarchical Survival LSTM

**Motivation:** Model both within-visit and across-visit structure

```python
class HierarchicalSurvivalLSTM(nn.Module):
    """
    Two-level LSTM: codes → visit → sequence
    """
    def __init__(self, vocab_size, embedding_dim, code_hidden_dim, visit_hidden_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Level 1: LSTM over codes within visit
        self.code_lstm = nn.LSTM(
            embedding_dim, code_hidden_dim,
            batch_first=True
        )
        
        # Level 2: LSTM over visit sequence
        self.visit_lstm = nn.LSTM(
            code_hidden_dim, visit_hidden_dim,
            batch_first=True
        )
        
        # Hazard head
        self.hazard_head = nn.Sequential(
            nn.Linear(visit_hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, visit_codes, visit_mask, sequence_mask):
        batch_size, num_visits, max_codes = visit_codes.shape
        
        # Embed codes
        embeddings = self.embedding(visit_codes)  # [B, V, C, E]
        
        # Process each visit with code-level LSTM
        visit_vectors = []
        for v in range(num_visits):
            codes_this_visit = embeddings[:, v, :, :]  # [B, C, E]
            
            # LSTM over codes
            code_lstm_out, (h_final, _) = self.code_lstm(codes_this_visit)
            
            # Use final hidden state as visit representation
            visit_vec = h_final.squeeze(0)  # [B, code_hidden_dim]
            visit_vectors.append(visit_vec)
        
        visit_vectors = torch.stack(visit_vectors, dim=1)  # [B, V, code_hidden]
        
        # LSTM over visits
        visit_lstm_out, _ = self.visit_lstm(visit_vectors)  # [B, V, visit_hidden]
        
        # Hazards
        hazards = self.hazard_head(visit_lstm_out).squeeze(-1)
        return hazards * sequence_mask.float()
```

---

## Summary

### Key Takeaways

1. **Survival LSTM is specialized for time-to-event prediction**
   - Outputs hazard at each visit (not just final prediction)
   - Handles censoring explicitly
   - Uses survival likelihood loss

2. **Similar to LSTMBaseline but with key differences**
   - Visit aggregation: same approach (currently mean pooling)
   - LSTM: same architecture (must be unidirectional)
   - Output head: different (per-visit hazards vs. single prediction)

3. **Can benefit from VisitEncoder integration**
   - Replace mean pooling with attention
   - Learn code importance for survival
   - Improve interpretability

4. **Multiple extensions available**
   - Time-aware: incorporate visit spacing
   - Competing risks: multiple event types
   - Hierarchical: model codes and visits separately

---

### Recommended Architecture

**For most use cases:**
```python
model = DiscreteTimeSurvivalLSTMWithAttention(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.1,
    visit_aggregation='attention'  # Use attention!
)
```

**For irregular visit spacing:**
```python
model = DiscreteTimeSurvivalLSTMWithTime(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=256,
    time_encoding_dim=16
)
```

**For multiple outcomes:**
```python
model = CompetingRisksSurvivalLSTM(
    vocab_size=10000,
    num_event_types=3,  # progression, death, readmission
    embedding_dim=128,
    hidden_dim=256
)
```

The survival LSTM architecture is flexible and can be adapted to various clinical prediction tasks while maintaining the core principles of discrete-time survival analysis!
