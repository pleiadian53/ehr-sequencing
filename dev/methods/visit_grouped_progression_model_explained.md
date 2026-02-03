# Visit-Grouped Progression Model: Architecture Deep Dive

**Date:** January 20, 2026  
**Source:** `docs/methods/pretrained-models-and-disease-progression.md`  
**Purpose:** Understanding hierarchical visit-grouped models for disease progression

---

## Overview

The **VisitGroupedProgressionModel** is a hierarchical architecture designed for modeling disease progression (e.g., CKD staging) from Electronic Health Records. It operates at two levels:

1. **Level 1 (Within-Visit):** Aggregate medical codes within each visit into a visit embedding
2. **Level 2 (Across-Visits):** Model the temporal sequence of visits to predict disease progression

**Key Innovation:** Leverages pre-trained code embeddings (e.g., from CEHR-BERT) and respects the natural structure of clinical care (visits).

---

## Architectural Overview

### Visual Structure

```
Patient Timeline:
┌─────────────────────────────────────────────────────────────┐
│ Visit 1        Visit 2        Visit 3        Visit 4        │
│ Day 0          Day 90         Day 180        Day 365        │
├─────────────────────────────────────────────────────────────┤
│ [ICD:E11.9,    [ICD:E11.65,   [ICD:E11.22,   [ICD:N18.3,   │
│  LOINC:2160-0, LOINC:2160-0,  LOINC:2160-0,  LOINC:2160-0, │
│  RX:860975]    RX:860975,     RX:860975,     RX:1234567,   │
│                RX:1545653]    LOINC:4548-4]  ICD:I10]      │
└─────────────────────────────────────────────────────────────┘
                    ↓ Level 1: Visit Encoder
┌─────────────────────────────────────────────────────────────┐
│ Visit Embedding 1  Visit Embedding 2  Visit Embedding 3  ...│
│ [v₁ ∈ ℝ²⁵⁶]       [v₂ ∈ ℝ²⁵⁶]       [v₃ ∈ ℝ²⁵⁶]            │
└─────────────────────────────────────────────────────────────┘
                    ↓ Add Time Features
┌─────────────────────────────────────────────────────────────┐
│ [v₁, t₁]          [v₂, t₂]          [v₃, t₃]              │
│ t₁ = (0, 0)       t₂ = (90, 90)     t₃ = (180, 90)        │
│ (days from start, days from prev)                          │
└─────────────────────────────────────────────────────────────┘
                    ↓ Level 2: LSTM Sequence Model
┌─────────────────────────────────────────────────────────────┐
│ Hidden State 1    Hidden State 2    Hidden State 3         │
│ [h₁ ∈ ℝ⁵¹²]       [h₂ ∈ ℝ⁵¹²]       [h₃ ∈ ℝ⁵¹²]            │
└─────────────────────────────────────────────────────────────┘
                    ↓ Prediction Heads
┌─────────────────────────────────────────────────────────────┐
│ Stage: [1,2,3,4,5]  Stage: [1,2,3,4,5]  Stage: [1,2,3,4,5] │
│ Time: 180 days      Time: 90 days       Time: 45 days      │
└─────────────────────────────────────────────────────────────┘
```

---

## Code Walkthrough

### Constructor: Model Initialization

```python
def __init__(
    self,
    pretrained_code_embeddings,  # From CEHR-BERT
    code_embed_dim=128,
    visit_embed_dim=256,
    hidden_dim=512,
    num_stages=5,  # CKD stages 1-5
    dropout=0.1
):
```

**Parameters Explained:**

| Parameter | Type | Purpose | Example |
|-----------|------|---------|---------|
| `pretrained_code_embeddings` | Tensor [vocab_size, embed_dim] | Pre-trained code embeddings from CEHR-BERT/Med-BERT | Shape: [30000, 128] |
| `code_embed_dim` | int | Dimension of code embeddings | 128 (typical) |
| `visit_embed_dim` | int | Dimension of visit-level embeddings | 256 (higher than code-level) |
| `hidden_dim` | int | LSTM hidden state dimension | 512 (captures complex progression patterns) |
| `num_stages` | int | Number of disease stages to predict | 5 (CKD stages 1-5) |
| `dropout` | float | Dropout rate for regularization | 0.1 (10% dropout) |

**Why these dimensions?**
- `code_embed_dim` (128): Pre-trained dimension from foundation model
- `visit_embed_dim` (256): **Larger** than code-level because visit embeddings must summarize multiple codes
- `hidden_dim` (512): Even **larger** to capture complex temporal dependencies across visits

**Hierarchy:** Code (128) → Visit (256) → Hidden State (512)

---

### Component 1: Pre-trained Code Embeddings

```python
# Level 1: Code embeddings (pre-trained)
self.code_embeddings = nn.Embedding.from_pretrained(
    pretrained_code_embeddings,
    freeze=False  # Allow fine-tuning
)
```

**What this does:**
1. Loads pre-trained embeddings from CEHR-BERT/Med-BERT
2. Each medical code → dense vector (e.g., ICD10:E11.9 → [0.23, -0.45, ..., 0.12])
3. `freeze=False`: Allows fine-tuning on your specific task

**Example:**
```python
# Pre-trained embeddings from CEHR-BERT
pretrained = load_cehrbert_embeddings()  # Shape: [30000 codes, 128 dim]

# Create embedding layer
code_embeddings = nn.Embedding.from_pretrained(pretrained, freeze=False)

# Usage
code_idx = torch.tensor([42, 100, 523])  # Three codes
embeds = code_embeddings(code_idx)  # Shape: [3, 128]
```

**Why pre-trained?**
- Learned from millions of patients
- Captures semantic relationships (diabetes codes cluster together)
- Much better than random initialization

**Why `freeze=False`?**
- Fine-tune embeddings for disease-specific patterns
- E.g., CKD-specific code relationships may differ from general population

---

### Component 2: Visit Encoder (Within-Visit Aggregation)

```python
# Level 1: Within-visit aggregation
self.visit_encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=code_embed_dim,
        nhead=4,
        dim_feedforward=code_embed_dim * 4,
        dropout=dropout,
        batch_first=True
    ),
    num_layers=2
)
```

**Purpose:** Aggregate codes within a single visit into a visit-level representation.

**Architecture:**
- **Transformer Encoder**: Captures relationships between codes in the same visit
- **2 layers**: Enough to model code interactions without overfitting
- **4 attention heads**: Can attend to multiple code relationships simultaneously
- **Feed-forward dim = 4 × embed_dim**: Standard Transformer architecture

**Why Transformer for within-visit?**
- **Set encoding**: Order of codes within a visit often doesn't matter
- **Attention**: Some codes more important (e.g., diagnosis > medication refill)
- **Flexible**: Handles variable number of codes per visit

**Example:**
```python
# Visit with 5 codes
visit_codes = [ICD:E11.9, LOINC:2160-0, RX:860975, ICD:I10, LOINC:4548-4]
code_embeds = code_embeddings(visit_codes)  # [5, 128]

# Transformer encodes relationships
# E.g., E11.9 (diabetes) + LOINC:4548-4 (HbA1c) → strong attention
visit_repr = visit_encoder(code_embeds)  # [5, 128] (same shape, but contextualized)
```

---

### Component 3: Visit Projection

```python
# Project to visit embedding
self.visit_projection = nn.Linear(code_embed_dim, visit_embed_dim)
```

**Purpose:** Project from code-level dimension (128) to visit-level dimension (256).

**Why needed?**
- Visit embeddings need higher capacity than individual codes
- Creates hierarchical representation: codes (128) → visits (256)

**Shape transformation:**
```python
pooled_codes = torch.randn(batch_size, 128)  # Aggregated codes
visit_embed = visit_projection(pooled_codes)  # [batch_size, 256]
```

---

### Component 4: LSTM Sequence Model

```python
# Level 2: Visit sequence modeling
self.visit_lstm = nn.LSTM(
    input_size=visit_embed_dim + 2,  # +2 for time features
    hidden_size=hidden_dim,
    num_layers=2,
    dropout=dropout,
    batch_first=True,
    bidirectional=False  # Causal for prediction
)
```

**Purpose:** Model temporal sequence of visits to predict disease progression.

**Key parameters:**

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `input_size` | `visit_embed_dim + 2` (258) | Visit embedding (256) + 2 time features |
| `hidden_size` | 512 | Captures complex temporal patterns |
| `num_layers` | 2 | Stack two LSTM layers for deeper temporal modeling |
| `dropout` | 0.1 | Between LSTM layers to prevent overfitting |
| `batch_first=True` | True | Input shape: [batch, sequence, features] |
| `bidirectional` | False | **Causal**: Only look at past visits (for prediction) |

**Why bidirectional=False?**
- **Causal modeling**: At visit t, we can only use visits 0 to t (not future visits)
- **Real-world prediction**: When predicting progression at visit 3, we don't know what happens at visit 4
- Bidirectional would be for retrospective analysis, not prospective prediction

**Input shape:**
```python
# After encoding all visits
visit_embeds_with_time = [
    [visit_1_embed (256-dim), days_from_start, days_from_prev],  # 258-dim
    [visit_2_embed (256-dim), days_from_start, days_from_prev],  # 258-dim
    ...
]
# Shape: [batch_size, num_visits, 258]
```

---

### Component 5: Prediction Heads

```python
# Progression prediction head
self.stage_classifier = nn.Linear(hidden_dim, num_stages)
self.time_to_progression = nn.Linear(hidden_dim, 1)  # Days until next stage
```

**Two prediction tasks:**

1. **Stage Classification**: What stage is the patient currently in?
   - Input: LSTM hidden state [batch, visits, 512]
   - Output: Logits over stages [batch, visits, 5] (for CKD stages 1-5)

2. **Time-to-Progression**: How long until the next stage?
   - Input: LSTM hidden state [batch, visits, 512]
   - Output: Days [batch, visits, 1]

**Why both tasks?**
- **Stage**: Current disease severity
- **Time**: Urgency of intervention (patient will progress in 30 days vs. 300 days)

**Multi-task learning benefits:**
- Shared LSTM learns better representations
- Time prediction adds temporal reasoning
- Complements classification with regression

---

## Forward Pass: Step-by-Step

### Method 1: `encode_visit()` - Single Visit Encoding

```python
def encode_visit(self, visit_codes, visit_mask):
    """
    Encode a single visit into a fixed-size embedding.
    
    Args:
        visit_codes: [batch, max_codes_per_visit]
        visit_mask: [batch, max_codes_per_visit] - 1 for real codes, 0 for padding
    
    Returns:
        visit_embedding: [batch, visit_embed_dim]
    """
```

**Step 1: Embed codes**

```python
code_embeds = self.code_embeddings(visit_codes)  # [batch, codes, code_dim]
```

**Example:**
```python
# Batch of 32 patients, each visit has up to 20 codes (padded)
visit_codes = torch.randint(0, 30000, (32, 20))  # [32, 20]
code_embeds = code_embeddings(visit_codes)  # [32, 20, 128]

# Each code is now a 128-dim vector
```

**Step 2: Apply Transformer encoder**

```python
# Mask padding tokens
attn_mask = ~visit_mask.bool()  # True = ignore
visit_repr = self.visit_encoder(
    code_embeds,
    src_key_padding_mask=attn_mask
)  # [batch, codes, code_dim]
```

**What happens:**
- Transformer allows codes to attend to each other
- Padding tokens are masked (ignored)
- Each code's representation is now **contextualized** by other codes in the visit

**Example attention patterns:**
```
Visit codes: [E11.9 (diabetes), 2160-0 (creatinine), 860975 (metformin), <PAD>, <PAD>]

Transformer attention:
  E11.9 attends to: 2160-0 (kidney function), 860975 (diabetes drug)
  2160-0 attends to: E11.9 (disease causing elevated creatinine)
  860975 attends to: E11.9 (treating this condition)
  <PAD> ignored
```

**Step 3: Pool to single visit embedding**

```python
# Pool to single visit embedding (mean over non-padding)
visit_mask_expanded = visit_mask.unsqueeze(-1)  # [batch, codes, 1]
masked_repr = visit_repr * visit_mask_expanded
visit_embed = masked_repr.sum(dim=1) / visit_mask.sum(dim=1, keepdim=True)
```

**What this does:**
- Computes **masked average** of code representations
- Only averages over real codes (padding ignored)
- Result: Single vector representing entire visit

**Shape transformations:**
```python
visit_repr:     [32, 20, 128]
visit_mask:     [32, 20]
visit_mask_expanded: [32, 20, 1]

masked_repr:    [32, 20, 128]  # Zero out padding
visit_embed:    [32, 128]      # Sum over codes, divide by count
```

**Example:**
```python
# Visit with 3 real codes + 17 padding
visit_repr = [embed_1, embed_2, embed_3, pad, pad, ..., pad]  # [20, 128]
visit_mask = [1, 1, 1, 0, 0, ..., 0]  # [20]

# Masked sum
masked_sum = embed_1 + embed_2 + embed_3  # [128]

# Average
visit_embed = masked_sum / 3  # [128]
```

**Step 4: Project to visit space**

```python
return self.visit_projection(visit_embed)  # [batch, visit_dim]
```

**Final shape:** [32, 256] (from 128 to 256 dimensions)

---

### Method 2: `forward()` - Full Sequence Processing

```python
def forward(self, patient_visits, time_features, visit_mask):
    """
    Predict disease progression from visit sequence.
    
    Args:
        patient_visits: [batch, num_visits, max_codes_per_visit]
        time_features: [batch, num_visits, 2] - (days_since_first, days_since_prev)
        visit_mask: [batch, num_visits, max_codes_per_visit]
    
    Returns:
        stage_logits: [batch, num_visits, num_stages]
        time_to_progression: [batch, num_visits, 1]
    """
```

**Input Example:**
```python
batch_size = 32 patients
num_visits = 10 visits per patient
max_codes_per_visit = 20 codes per visit

patient_visits:  [32, 10, 20]  # Code indices
time_features:   [32, 10, 2]   # (days from first visit, days from previous visit)
visit_mask:      [32, 10, 20]  # 1 = real code, 0 = padding
```

**Step 1: Encode each visit**

```python
batch_size, num_visits, max_codes = patient_visits.shape

# Encode each visit
visit_embeds = []
for i in range(num_visits):
    visit_embed = self.encode_visit(
        patient_visits[:, i, :],    # [batch, max_codes]
        visit_mask[:, i, :]         # [batch, max_codes]
    )  # [batch, visit_dim]
    visit_embeds.append(visit_embed)

visit_embeds = torch.stack(visit_embeds, dim=1)  # [batch, visits, visit_dim]
```

**What happens:**
- Loop over each visit position (0 to num_visits-1)
- For all patients simultaneously, encode visit i
- Stack results into sequence

**Shape transformations:**
```python
patient_visits[:, 0, :]:  [32, 20]    # Visit 0 for all patients
encode_visit():           [32, 256]   # Visit 0 embeddings

After loop:
visit_embeds:             [32, 10, 256]  # All visits encoded
```

**Step 2: Add time features**

```python
# Concatenate time features
visit_embeds_with_time = torch.cat([
    visit_embeds,     # [batch, visits, 256]
    time_features     # [batch, visits, 2]
], dim=-1)  # [batch, visits, 258]
```

**Time features:**
```python
time_features[patient_i, visit_j] = [
    days_since_first_visit,   # Absolute time
    days_since_previous_visit # Relative time (visit spacing)
]
```

**Example:**
```python
Visit 0: time_features = [0, 0]       # First visit
Visit 1: time_features = [90, 90]     # 90 days after first, 90 since visit 0
Visit 2: time_features = [180, 90]    # 180 days after first, 90 since visit 1
Visit 3: time_features = [365, 185]   # 365 days after first, 185 since visit 2
```

**Why both time features?**
- `days_since_first`: Captures disease duration
- `days_since_previous`: Captures visit frequency (rapid progression → frequent visits)

**Step 3: Model visit sequence with LSTM**

```python
# Model visit sequence
lstm_out, _ = self.visit_lstm(visit_embeds_with_time)  # [batch, visits, hidden]
lstm_out = self.dropout(lstm_out)
```

**LSTM processing:**
```python
# Input shape: [32, 10, 258]
# LSTM processes sequence:
#   h0 = LSTM(visit_0, h_prev=0, c_prev=0)
#   h1 = LSTM(visit_1, h_prev=h0, c_prev=c0)
#   h2 = LSTM(visit_2, h_prev=h1, c_prev=c1)
#   ...
#   h9 = LSTM(visit_9, h_prev=h8, c_prev=c8)

# Output: [32, 10, 512]
# Each hidden state h_t encodes "patient state after visit t"
```

**What each hidden state represents:**
- `h0`: Patient state after visit 0 (baseline)
- `h1`: Patient state after visit 1 (early disease)
- `h2`: Patient state after visit 2 (progression detected?)
- ...
- `h9`: Current patient state (latest information)

**Step 4: Predict stage and time**

```python
# Predict stage at each visit
stage_logits = self.stage_classifier(lstm_out)  # [batch, visits, num_stages]

# Predict time to next stage
time_pred = self.time_to_progression(lstm_out)  # [batch, visits, 1]
time_pred = torch.relu(time_pred)  # Ensure positive
```

**Stage prediction:**
```python
# For each visit, predict which stage the patient is in
stage_logits:  [32, 10, 5]  # 5 CKD stages

# Example for patient 0, visit 3:
stage_logits[0, 3] = [0.1, 0.2, 0.65, 0.04, 0.01]
# Predicted stage: 3 (index 2) with 65% confidence
```

**Time prediction:**
```python
# For each visit, predict days until progression to next stage
time_pred:  [32, 10, 1]

# Example for patient 0, visit 3:
time_pred[0, 3] = [120.5]  # Predicted to progress in ~120 days
```

**Why ReLU on time?**
- Time predictions must be positive (can't progress in -50 days)
- ReLU(x) = max(0, x) ensures non-negative

---

## Complete Data Flow Example

Let's trace a single patient through the entire model:

### Input: Patient Timeline

```python
Patient P001 has 3 visits:

Visit 0 (Day 0):
  Codes: [E11.9, 2160-0, 860975]  # Diabetes, creatinine, metformin
  Time: (0 days from first, 0 days from previous)

Visit 1 (Day 90):
  Codes: [E11.9, 2160-0, 860975, I10]  # + Hypertension
  Time: (90 days from first, 90 days from previous)

Visit 2 (Day 180):
  Codes: [E11.22, 2160-0, N18.3, 4548-4]  # Diabetes w/ CKD, CKD stage 3
  Time: (180 days from first, 90 days from previous)
```

### Forward Pass

**Step 1: Encode Visit 0**

```python
visit_codes = [E11.9, 2160-0, 860975]  # Indices: [42, 523, 1200]

# Embed codes
code_embeds = code_embeddings([42, 523, 1200])
# Shape: [3, 128]
# [[0.23, -0.45, ...], [0.19, -0.41, ...], [0.21, -0.43, ...]]

# Transformer encoding (codes attend to each other)
visit_repr = visit_encoder(code_embeds)
# Shape: [3, 128] (contextualized)

# Pool to single visit embedding
visit_embed_0 = mean(visit_repr)
# Shape: [128]

# Project to visit space
visit_embed_0 = visit_projection(visit_embed_0)
# Shape: [256]
```

**Step 2: Encode Visits 1 and 2 (same process)**

```python
visit_embed_1 = encode_visit([E11.9, 2160-0, 860975, I10])  # [256]
visit_embed_2 = encode_visit([E11.22, 2160-0, N18.3, 4548-4])  # [256]

# Stack into sequence
visit_sequence = stack([visit_embed_0, visit_embed_1, visit_embed_2])
# Shape: [3, 256]
```

**Step 3: Add time features**

```python
time_features = [
    [0, 0],      # Visit 0
    [90, 90],    # Visit 1
    [180, 90]    # Visit 2
]
# Shape: [3, 2]

visit_sequence_with_time = concat(visit_sequence, time_features)
# Shape: [3, 258]
```

**Step 4: LSTM processing**

```python
# Initial state
h_0 = zeros(512), c_0 = zeros(512)

# Visit 0
h_1, c_1 = LSTM(visit_0_with_time, h_0, c_0)
# h_1: [512] - Patient state after first visit

# Visit 1
h_2, c_2 = LSTM(visit_1_with_time, h_1, c_1)
# h_2: [512] - Patient state after second visit (hypertension developed)

# Visit 2
h_3, c_3 = LSTM(visit_2_with_time, h_2, c_2)
# h_3: [512] - Current patient state (CKD diagnosed)

lstm_out = [h_1, h_2, h_3]
# Shape: [3, 512]
```

**Step 5: Predictions**

```python
# Stage predictions at each visit
stage_logits = stage_classifier(lstm_out)
# Shape: [3, 5]

stage_logits = [
    [0.8, 0.15, 0.04, 0.01, 0.0],   # Visit 0: Stage 1 (healthy)
    [0.3, 0.5, 0.15, 0.04, 0.01],   # Visit 1: Stage 2 (mild)
    [0.05, 0.15, 0.65, 0.13, 0.02]  # Visit 2: Stage 3 (CKD detected)
]

# Time predictions at each visit
time_pred = time_to_progression(lstm_out)
# Shape: [3, 1]

time_pred = [
    [240.5],  # Visit 0: Predicted to progress in ~240 days
    [120.3],  # Visit 1: Faster progression (now 120 days)
    [60.8]    # Visit 2: Accelerating (60 days to next stage)
]
```

---

## Shape Reference Table

| Component | Input Shape | Output Shape | Example |
|-----------|-------------|--------------|---------|
| **Code Embeddings** | [batch, codes] | [batch, codes, 128] | [32, 20] → [32, 20, 128] |
| **Transformer Encoder** | [batch, codes, 128] | [batch, codes, 128] | [32, 20, 128] → [32, 20, 128] |
| **Visit Pooling** | [batch, codes, 128] | [batch, 128] | [32, 20, 128] → [32, 128] |
| **Visit Projection** | [batch, 128] | [batch, 256] | [32, 128] → [32, 256] |
| **Visit Encoding (full)** | [batch, visits, codes] | [batch, visits, 256] | [32, 10, 20] → [32, 10, 256] |
| **Add Time Features** | [batch, visits, 256] + [batch, visits, 2] | [batch, visits, 258] | Concatenate |
| **LSTM** | [batch, visits, 258] | [batch, visits, 512] | [32, 10, 258] → [32, 10, 512] |
| **Stage Classifier** | [batch, visits, 512] | [batch, visits, 5] | [32, 10, 512] → [32, 10, 5] |
| **Time Predictor** | [batch, visits, 512] | [batch, visits, 1] | [32, 10, 512] → [32, 10, 1] |

---

## Why This Architecture Works

### 1. Hierarchical Representation

**Problem:** EHR data is inherently hierarchical (codes → visits → patient trajectory)

**Solution:** Two-level hierarchy matches clinical reality
- Level 1: Aggregate codes within visit (set encoding)
- Level 2: Model visit sequence (temporal modeling)

**Alternative (naive):** Flatten all codes into one sequence
```python
# Bad: Treats all codes as flat sequence
all_codes = [visit0_code0, visit0_code1, ..., visit1_code0, visit1_code1, ...]
# Loses visit boundaries, ignores clinical structure
```

---

### 2. Pre-trained Embeddings

**Problem:** Medical codes are sparse (thousands of codes, limited data per code)

**Solution:** Use pre-trained embeddings from millions of patients
- Captures semantic relationships
- Better generalization
- Faster convergence

**Alternative (naive):** Random initialization
```python
# Bad: Each code starts random, needs lots of data to learn
embeddings = nn.Embedding(30000, 128)  # Random init
```

---

### 3. Temporal Modeling with LSTM

**Problem:** Disease progression unfolds over time with long-term dependencies

**Solution:** LSTM captures:
- Long-term patterns (CKD develops over years)
- Variable visit spacing (irregular follow-ups)
- Sequential dependencies (stage 3 must follow stage 2)

**Alternative (naive):** CNN or simple aggregation
```python
# Bad: CNN only captures local patterns, not long-term progression
conv = nn.Conv1d(...)  # Limited receptive field
```

---

### 4. Multi-task Learning

**Problem:** Stage classification alone may miss temporal dynamics

**Solution:** Joint prediction of stage + time-to-progression
- Stage: Current severity
- Time: Urgency/velocity of disease
- Shared representations benefit both tasks

---

### 5. Attention within Visits

**Problem:** Not all codes equally important (diagnosis > medication refill)

**Solution:** Transformer encoder learns to weight important codes
- Attends to diagnostic codes more
- Captures code interactions (diabetes + elevated creatinine → CKD risk)

---

## Training the Model

### Loss Functions

```python
# Stage classification loss (cross-entropy)
stage_loss = nn.CrossEntropyLoss()(stage_logits, true_stages)

# Time prediction loss (MSE)
time_loss = nn.MSELoss()(time_pred, true_time_to_progression)

# Combined loss
total_loss = stage_loss + lambda_time * time_loss
```

### Training Loop

```python
model = VisitGroupedProgressionModel(
    pretrained_code_embeddings=cehrbert_embeddings,
    code_embed_dim=128,
    visit_embed_dim=256,
    hidden_dim=512,
    num_stages=5,
    dropout=0.1
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    for batch in dataloader:
        patient_visits, time_features, visit_mask, stages, times = batch
        
        # Forward pass
        stage_logits, time_pred = model(patient_visits, time_features, visit_mask)
        
        # Compute losses
        stage_loss = nn.CrossEntropyLoss()(
            stage_logits.view(-1, 5),
            stages.view(-1)
        )
        time_loss = nn.MSELoss()(time_pred.squeeze(), times)
        
        total_loss = stage_loss + 0.5 * time_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

---

## Inference: Making Predictions

### Scenario 1: Predict Current Stage

```python
model.eval()
with torch.no_grad():
    # Patient with 5 visits so far
    stage_logits, time_pred = model(patient_visits, time_features, visit_mask)
    
    # Get current stage (latest visit)
    current_stage_logits = stage_logits[:, -1, :]  # [batch, 5]
    current_stage = current_stage_logits.argmax(dim=-1)  # [batch]
    
    print(f"Predicted stage: {current_stage.item() + 1}")  # +1 for 1-indexed
```

### Scenario 2: Predict Time to Next Stage

```python
# Days until progression
days_to_progression = time_pred[:, -1, 0]  # [batch]
print(f"Estimated progression in {days_to_progression.item():.0f} days")
```

### Scenario 3: Analyze Progression Trajectory

```python
# Get stage probabilities over all visits
all_stage_probs = torch.softmax(stage_logits, dim=-1)  # [batch, visits, 5]

# Plot progression for patient 0
import matplotlib.pyplot as plt

for stage in range(5):
    plt.plot(all_stage_probs[0, :, stage].cpu(), label=f'Stage {stage+1}')

plt.xlabel('Visit Number')
plt.ylabel('Stage Probability')
plt.legend()
plt.title('Disease Progression Trajectory')
plt.show()
```

---

## Key Takeaways

1. **Hierarchical design** matches clinical reality (codes → visits → trajectory)
2. **Pre-trained embeddings** provide strong initialization (don't train from scratch)
3. **Transformer within visits** captures code interactions via attention
4. **LSTM across visits** models temporal dependencies and irregular spacing
5. **Multi-task learning** (stage + time) improves representation quality
6. **Causal modeling** (unidirectional LSTM) enables prospective prediction

This architecture is well-suited for disease progression modeling because it respects the structure of clinical data, leverages pre-trained knowledge, and explicitly models temporal dynamics.

---

## Extensions and Improvements

### 1. Hierarchical Attention

Add attention over visits (which past visits most relevant?):
```python
visit_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
attended_visits, _ = visit_attention(lstm_out, lstm_out, lstm_out)
```

### 2. Incorporate Demographics

Add patient features (age, gender) to each visit:
```python
demographic_features = patient_demographics.unsqueeze(1).expand(-1, num_visits, -1)
visit_embeds_with_all = torch.cat([visit_embeds, time_features, demographic_features], dim=-1)
```

### 3. Survival Analysis

Instead of time-to-progression, predict survival curves:
```python
from pycox.models import CoxPH
survival_model = CoxPH(lstm_out, time_to_event, event_indicator)
```

### 4. Interpretability

Add attention visualization to explain predictions:
```python
# Extract attention weights from Transformer
attention_weights = visit_encoder.layers[0].self_attn.attn_weights
# Visualize which codes contributed most to prediction
```

---

## Comparison with Alternative Architectures

| Architecture | Pros | Cons | Best Use Case |
|--------------|------|------|---------------|
| **Flat sequence (no visits)** | Simple | Loses structure | Short sequences |
| **CNN** | Fast, parallel | Limited long-term memory | Local patterns |
| **Transformer (full)** | Best for context | Quadratic complexity | Medium sequences |
| **Visit-grouped + LSTM** | Hierarchical, efficient | More complex | Long-term progression |
| **GRU instead of LSTM** | Fewer parameters | Less expressive | Smaller datasets |

**This architecture** is optimal for disease progression because it balances structure, efficiency, and long-term temporal modeling.
