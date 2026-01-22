# LSTM Baseline Model: Architecture Walkthrough

**Date:** January 20, 2026  
**Source:** `src/ehrsequencing/models/lstm_baseline.py`  
**Purpose:** Understanding the LSTM baseline model for visit-grouped EHR sequences

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [VisitEncoder Deep Dive](#visitencoder-deep-dive)
3. [LSTMBaseline Model](#lstmbaseline-model)
4. [Forward Pass Walkthrough](#forward-pass-walkthrough)
5. [Design Decisions](#design-decisions)

---

## Architecture Overview

The LSTM baseline model follows a hierarchical architecture:

```
Patient EHR Data:
┌─────────────────────────────────────────────────────────┐
│ Visit 1        Visit 2        Visit 3                   │
│ [Code A,       [Code B,       [Code C,                  │
│  Code B,        Code D,        Code E,                  │
│  Code C]        Code E]        Code F]                  │
└─────────────────────────────────────────────────────────┘
        ↓ Embedding Layer
┌─────────────────────────────────────────────────────────┐
│ [[emb_A,       [[emb_B,       [[emb_C,                  │
│   emb_B,         emb_D,         emb_E,                  │
│   emb_C],        emb_E],        emb_F],                 │
└─────────────────────────────────────────────────────────┘
        ↓ VisitEncoder (aggregate codes within visit)
┌─────────────────────────────────────────────────────────┐
│ visit_vec_1    visit_vec_2    visit_vec_3              │
│ [v₁ ∈ ℝᵈ]      [v₂ ∈ ℝᵈ]      [v₃ ∈ ℝᵈ]                │
└─────────────────────────────────────────────────────────┘
        ↓ LSTM (model temporal dependencies)
┌─────────────────────────────────────────────────────────┐
│ hidden_1       hidden_2       hidden_3                  │
│ [h₁ ∈ ℝʰ]      [h₂ ∈ ℝʰ]      [h₃ ∈ ℝʰ]                │
└─────────────────────────────────────────────────────────┘
        ↓ Prediction Head (final hidden → output)
┌─────────────────────────────────────────────────────────┐
│ Prediction: [0.85] (e.g., 85% disease risk)            │
└─────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
1. **Visit-level modeling**: Respect clinical structure (visits are natural units)
2. **Flexible aggregation**: Multiple ways to combine codes within visits
3. **Masked operations**: Handle variable-length sequences correctly
4. **Task-agnostic**: Supports classification, regression, sequence tasks

---

## VisitEncoder Deep Dive

The `VisitEncoder` solves the problem: **"How do we combine multiple medical codes within a single visit into one vector?"**

### Problem Statement

**Input:** A visit with variable number of codes
```python
Visit 1: [E11.9, LOINC:2160-0, RX:860975]  # 3 codes
Visit 2: [E11.9, LOINC:2160-0, RX:860975, I10, LOINC:4548-4]  # 5 codes
Visit 3: [N18.3]  # 1 code
```

After embedding:
```python
Visit 1: [[emb_1], [emb_2], [emb_3]]  # Shape: [3, 128]
Visit 2: [[emb_1], [emb_2], [emb_3], [emb_4], [emb_5]]  # Shape: [5, 128]
Visit 3: [[emb_1]]  # Shape: [1, 128]
```

**Challenge:** LSTM expects fixed-size input per timestep
```python
# Need to convert variable [num_codes, embed_dim] → fixed [embed_dim]
```

**Solution:** VisitEncoder aggregates codes into fixed-size visit vector

---

### Aggregation Strategies

The VisitEncoder supports 4 aggregation methods:

| Method | Description | Use Case | Pros | Cons |
|--------|-------------|----------|------|------|
| **mean** | Average embeddings | Default, general-purpose | Simple, stable | Treats all codes equally |
| **sum** | Sum embeddings | Preserve magnitude | Sensitive to code count | May explode with many codes |
| **max** | Max pooling | Capture strongest signal | Robust to noise | Loses fine-grained info |
| **attention** | Learned weights | Importance weighting | Most expressive | More parameters, may overfit |

---

### 1. Attention Aggregation with Linear Maps

#### The Code

```python
if aggregation == 'attention':
    self.attention = nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim // 2),  # Compress
        nn.Tanh(),                                      # Nonlinearity
        nn.Linear(embedding_dim // 2, 1)                # Score
    )
```

#### Why This Architecture?

**Goal:** Learn to assign importance weights to each code in a visit.

**Architecture Breakdown:**

```
Input: Code embedding [embedding_dim]
       ↓
Layer 1: Linear(embedding_dim → embedding_dim // 2)
       ↓
       Compress to smaller space
       (e.g., 128 → 64)
       ↓
Tanh: Nonlinear activation
       ↓
       Introduce expressiveness
       ↓
Layer 2: Linear(embedding_dim // 2 → 1)
       ↓
       Map to single score
       ↓
Output: Attention score [1]
```

---

#### Why Two Layers (Not One)?

**Option 1: Single Linear Layer (Naive)**
```python
# Naive approach
self.attention = nn.Linear(embedding_dim, 1)
```

**Problem:** This is just a dot product with a learned weight vector
```python
score = w · embedding  # Linear function only

# Limited expressivity:
# Can only learn linear combinations
# E.g., "give high score if dimension 5 is large"
# Cannot learn: "give high score if dimensions 5 AND 10 are both large"
```

---

**Option 2: Two Layers with Bottleneck (Actual Implementation)**
```python
self.attention = nn.Sequential(
    nn.Linear(embedding_dim, embedding_dim // 2),  # Bottleneck
    nn.Tanh(),
    nn.Linear(embedding_dim // 2, 1)
)
```

**Benefits:**

1. **Nonlinear Expressiveness**
   ```python
   # Can learn complex patterns:
   # "High score if (dim_5 is high AND dim_10 is low) OR (dim_3 is medium)"
   # Nonlinearity (Tanh) enables this
   ```

2. **Dimensionality Reduction (Bottleneck)**
   ```python
   # embedding_dim = 128
   # Bottleneck = 64
   
   # Forces network to learn compact representation
   # Fewer parameters than full embedding_dim → 1
   
   # Parameters:
   # Single layer: 128 * 1 = 128 params
   # Two layers: 128 * 64 + 64 * 1 = 8,256 params
   # (More expressive despite more params - the bottleneck regularizes)
   ```

3. **Prevents Overfitting**
   ```python
   # Bottleneck acts as information bottleneck
   # Must compress to 64 dims before scoring
   # Forces learning of generalizable patterns
   ```

---

#### Mathematical Intuition

**Attention mechanism computes:**

```python
# For each code i in visit:
score_i = attention_network(embedding_i)

# Convert scores to weights (sum to 1)
weights = softmax([score_1, score_2, ..., score_n])

# Weighted average
visit_vector = Σ (weight_i * embedding_i)
```

**Two-layer network enables:**
```python
score = W₂ · tanh(W₁ · embedding + b₁) + b₂

# This can learn:
# - "Diagnosis codes (high in dim 0-50) are important" (linear component)
# - "But only if accompanied by lab results (dim 51-80)" (nonlinear interaction)
# - "Medication refills (high in dim 81-100) are less important" (weighting)
```

---

#### Concrete Example

**Scenario:** Visit with 3 codes

```python
Visit codes:
  Code A: E11.9 (diabetes diagnosis)
  Code B: 2160-0 (creatinine lab)
  Code C: 860975 (metformin medication)

Embeddings (simplified to 4D):
  emb_A = [0.8, 0.2, -0.1, 0.5]  # Diagnosis pattern
  emb_B = [0.1, 0.9, 0.3, -0.2]  # Lab pattern
  emb_C = [0.3, -0.1, 0.7, 0.4]  # Medication pattern
```

**Attention computation:**

```python
# Layer 1: Compress 4D → 2D
W₁ = [[0.5, 0.3],   # Learned weights
      [0.2, 0.8],
      [-0.4, 0.6],
      [0.1, -0.2]]

hidden_A = tanh(W₁ᵀ · emb_A) = tanh([0.47, 0.11]) = [0.44, 0.11]
hidden_B = tanh(W₁ᵀ · emb_B) = tanh([0.44, 0.88]) = [0.41, 0.71]
hidden_C = tanh(W₁ᵀ · emb_C) = tanh([-0.11, 0.43]) = [-0.11, 0.41]

# Layer 2: Map 2D → 1D (score)
W₂ = [1.2, 0.8]

score_A = W₂ · hidden_A = 1.2*0.44 + 0.8*0.11 = 0.616
score_B = W₂ · hidden_B = 1.2*0.41 + 0.8*0.71 = 1.060  # Highest!
score_C = W₂ · hidden_C = 1.2*(-0.11) + 0.8*0.41 = 0.196

# Softmax to get weights
weights = softmax([0.616, 1.060, 0.196])
        = [0.28, 0.53, 0.19]

# Weighted average
visit_vector = 0.28*emb_A + 0.53*emb_B + 0.19*emb_C
```

**Interpretation:** 
- Lab result (Code B) gets highest weight (0.53)
- Diagnosis (Code A) is moderately important (0.28)
- Medication refill (Code C) is less important (0.19)
- The network **learned** these weights from data!

---

#### Why Bottleneck Dimension is `embedding_dim // 2`?

**Design choice:** Half the embedding dimension

```python
embedding_dim = 128
bottleneck = 128 // 2 = 64
```

**Rationale:**

1. **Balance between expressiveness and efficiency**
   ```python
   # Too large (e.g., same as embedding_dim):
   bottleneck = 128  # 128*128 + 128*1 = 16,512 params
   # Risk: Overfitting, slow training
   
   # Too small (e.g., very compressed):
   bottleneck = 8   # 128*8 + 8*1 = 1,032 params
   # Risk: Underfitting, can't learn complex patterns
   
   # Just right (half):
   bottleneck = 64  # 128*64 + 64*1 = 8,256 params
   # Good compromise
   ```

2. **Information Bottleneck Theory**
   ```python
   # Forcing compression through bottleneck
   # Makes network learn most important features
   # Improves generalization
   ```

3. **Empirical Success**
   ```python
   # Half-size bottlenecks are common in:
   # - Attention mechanisms (Transformer)
   # - Autoencoders
   # - Recommendation systems
   ```

**Alternative choices:**
```python
# More aggressive compression (if overfitting)
bottleneck = embedding_dim // 4  # 128 → 32

# More expressive (if underfitting)
bottleneck = embedding_dim * 3 // 4  # 128 → 96
```

---

#### Forward Pass with Attention

```python
def forward(self, visit_embeddings, visit_mask):
    # visit_embeddings: [batch, codes, embed_dim]
    # visit_mask: [batch, codes, 1]
    
    # Compute attention scores for each code
    attention_scores = self.attention(visit_embeddings)
    # Shape: [batch, codes, 1]
    
    # Example for one patient:
    # visit_embeddings = [[emb_A], [emb_B], [emb_C], [PAD]]
    # attention_scores = [[0.616], [1.060], [0.196], [-2.3]]
    
    # Mask padding codes (set to -inf so softmax → 0)
    attention_scores = attention_scores.masked_fill(
        visit_mask == 0, float('-inf')
    )
    # After masking: [[0.616], [1.060], [0.196], [-inf]]
    
    # Softmax over codes dimension
    attention_weights = torch.softmax(attention_scores, dim=1)
    # Shape: [batch, codes, 1]
    # After softmax: [[0.28], [0.53], [0.19], [0.0]]
    # Note: -inf → 0 after softmax (padding ignored)
    
    # Weighted sum
    visit_vector = (visit_embeddings * attention_weights).sum(dim=1)
    # Shape: [batch, embed_dim]
    # = 0.28*emb_A + 0.53*emb_B + 0.19*emb_C + 0.0*PAD
    
    return visit_vector
```

**Key insight:** Masking ensures padding tokens get zero weight, so they don't affect the visit representation.

---

### 2. Visit Mask Initialization

#### The Code

```python
if visit_mask is None:
    visit_mask = torch.ones(
        visit_embeddings.shape[:-1],  # Key line!
        device=visit_embeddings.device
    )
```

#### What Does `shape[:-1]` Mean?

**Shape slicing in PyTorch:**

```python
tensor.shape       # Full shape tuple
tensor.shape[:-1]  # All dimensions except the last
tensor.shape[-1]   # Only the last dimension

# Example:
visit_embeddings.shape = [32, 20, 128]
                         └─┬─┘ └┬┘ └┬┘
                          batch codes embed_dim

visit_embeddings.shape[:-1] = [32, 20]  # Exclude last dimension
visit_embeddings.shape[-1] = 128        # Only last dimension
```

---

#### Why Initialize This Way?

**Problem:** User might not provide a mask for visits with no padding.

```python
# Scenario: Small dataset, all visits have 5 codes, no padding
visit_embeddings = torch.randn(32, 5, 128)  # [batch, codes, embed]
visit_mask = None  # User didn't provide mask
```

**Solution:** Create a default mask of all 1s (all codes are real).

```python
visit_mask = torch.ones([32, 5], device=visit_embeddings.device)
# Shape: [batch, codes]
# All 1s → all codes are treated as real (no padding)
```

---

#### Why Not `torch.ones(visit_embeddings.shape)`?

**Wrong approach:**
```python
# BAD: Creates mask with same shape as embeddings
visit_mask = torch.ones([32, 5, 128], device=...)
# Shape: [batch, codes, embed_dim]  ← Wrong!

# Mask should have shape [batch, codes], not [batch, codes, embed_dim]
# We want one mask value per code, not per embedding dimension
```

**Correct approach:**
```python
# GOOD: Creates mask with shape [batch, codes]
visit_mask = torch.ones([32, 5], device=...)
# Shape: [batch, codes]  ← Correct!

# Later expanded to [batch, codes, 1] for broadcasting
```

---

#### Device Placement

```python
visit_mask = torch.ones(..., device=visit_embeddings.device)
                                     └────────────┬────────────┘
                                     Ensure same device (CPU/GPU)
```

**Why important?**

```python
# If embeddings on GPU:
visit_embeddings.device  # cuda:0

# But mask on CPU:
visit_mask.device  # cpu

# Operations fail:
result = visit_embeddings * visit_mask  # RuntimeError: devices don't match!

# Solution: Create mask on same device
visit_mask = torch.ones(..., device=visit_embeddings.device)
```

---

#### The Full Mask Flow

```python
# Input
visit_embeddings: [32, 20, 128]  # batch, codes, embed_dim
visit_mask: None

# Step 1: Initialize mask
if visit_mask is None:
    visit_mask = torch.ones([32, 20], device=cuda)
# Shape: [32, 20]  (matches [batch, codes])

# Step 2: Expand mask for broadcasting
visit_mask = visit_mask.unsqueeze(-1)
# Shape: [32, 20, 1]
# Broadcasting: [32, 20, 1] * [32, 20, 128] → [32, 20, 128]

# Step 3: Masked aggregation
masked_embeddings = visit_embeddings * visit_mask
# Shape: [32, 20, 128]
# Each embedding multiplied by 1 (all codes treated as real)
```

---

### 3. Mean Aggregation

#### The Code

```python
if self.aggregation == 'mean':
    # Masked mean
    masked_embeddings = visit_embeddings * visit_mask
    sum_embeddings = masked_embeddings.sum(dim=1)
    count = visit_mask.sum(dim=1).clamp(min=1)
    visit_vector = sum_embeddings / count
```

#### Why Not Just Use `.mean()`?

**Naive approach (WRONG):**
```python
# This is WRONG with padding!
visit_vector = visit_embeddings.mean(dim=1)
```

**Problem with naive mean:**

```python
# Visit with 3 real codes + 2 padding
visit_embeddings = [
    [emb_1],  # Real code: [0.5, -0.3, 0.8]
    [emb_2],  # Real code: [0.3, 0.6, -0.2]
    [emb_3],  # Real code: [0.7, 0.1, 0.4]
    [PAD],    # Padding: [0.0, 0.0, 0.0]
    [PAD]     # Padding: [0.0, 0.0, 0.0]
]

# Naive mean (WRONG)
naive_mean = (emb_1 + emb_2 + emb_3 + PAD + PAD) / 5
           = (sum of 3 real embeddings) / 5
           # Dividing by 5 (including padding) dilutes the signal!

# Correct masked mean
masked_mean = (emb_1 + emb_2 + emb_3) / 3
            = (sum of 3 real embeddings) / 3
            # Dividing by 3 (only real codes) preserves signal strength
```

---

#### Step-by-Step Walkthrough

**Example Setup:**

```python
batch_size = 2
max_codes = 4
embed_dim = 3

# Patient 1: 3 real codes + 1 padding
# Patient 2: 2 real codes + 2 padding

visit_embeddings = torch.tensor([
    # Patient 1
    [[0.5, -0.3, 0.8],   # Code A
     [0.3, 0.6, -0.2],   # Code B
     [0.7, 0.1, 0.4],    # Code C
     [0.0, 0.0, 0.0]],   # Padding
    
    # Patient 2
    [[0.9, 0.2, 0.5],    # Code D
     [0.1, -0.4, 0.3],   # Code E
     [0.0, 0.0, 0.0],    # Padding
     [0.0, 0.0, 0.0]]    # Padding
])
# Shape: [2, 4, 3]

visit_mask = torch.tensor([
    # Patient 1: 3 real codes
    [[1.0],
     [1.0],
     [1.0],
     [0.0]],  # Padding
    
    # Patient 2: 2 real codes
    [[1.0],
     [1.0],
     [0.0],   # Padding
     [0.0]]   # Padding
])
# Shape: [2, 4, 1]
```

---

**Step 1: Zero out padding embeddings**

```python
masked_embeddings = visit_embeddings * visit_mask
# Shape: [2, 4, 3]

# Result:
# Patient 1:
[[0.5, -0.3, 0.8],   # 1.0 * [0.5, -0.3, 0.8]
 [0.3, 0.6, -0.2],   # 1.0 * [0.3, 0.6, -0.2]
 [0.7, 0.1, 0.4],    # 1.0 * [0.7, 0.1, 0.4]
 [0.0, 0.0, 0.0]]    # 0.0 * [...] = [0, 0, 0] ← Padding zeroed out

# Patient 2:
[[0.9, 0.2, 0.5],    # 1.0 * [0.9, 0.2, 0.5]
 [0.1, -0.4, 0.3],   # 1.0 * [0.1, -0.4, 0.3]
 [0.0, 0.0, 0.0],    # 0.0 * [...] = [0, 0, 0] ← Padding zeroed out
 [0.0, 0.0, 0.0]]    # 0.0 * [...] = [0, 0, 0] ← Padding zeroed out
```

**Why multiply?** Broadcasting zeros out padding dimensions automatically.

---

**Step 2: Sum over codes dimension**

```python
sum_embeddings = masked_embeddings.sum(dim=1)
# Sum over dim=1 (codes dimension)
# Shape: [2, 3]  (batch, embed_dim)

# Patient 1: Sum of 3 real codes
sum_1 = [0.5, -0.3, 0.8] + [0.3, 0.6, -0.2] + [0.7, 0.1, 0.4] + [0, 0, 0]
      = [1.5, 0.4, 1.0]

# Patient 2: Sum of 2 real codes
sum_2 = [0.9, 0.2, 0.5] + [0.1, -0.4, 0.3] + [0, 0, 0] + [0, 0, 0]
      = [1.0, -0.2, 0.8]

sum_embeddings = [
    [1.5, 0.4, 1.0],   # Patient 1
    [1.0, -0.2, 0.8]   # Patient 2
]
```

---

**Step 3: Count real codes (for averaging)**

```python
count = visit_mask.sum(dim=1).clamp(min=1)
# Sum over dim=1 (codes dimension)
# Shape: [2, 1]

# Patient 1: 1 + 1 + 1 + 0 = 3
# Patient 2: 1 + 1 + 0 + 0 = 2

count = [
    [3.0],  # Patient 1 has 3 real codes
    [2.0]   # Patient 2 has 2 real codes
]
```

**Why `.clamp(min=1)`?**

```python
# Edge case: Visit with all padding (empty visit)
visit_mask_empty = [[0.0], [0.0], [0.0], [0.0]]
count_empty = visit_mask_empty.sum(dim=1)  # 0.0

# Without clamp:
visit_vector = sum_embeddings / 0.0  # Division by zero! NaN!

# With clamp:
count_empty = count_empty.clamp(min=1)  # 1.0
visit_vector = sum_embeddings / 1.0  # Safe! (though still zeros)
```

**This prevents NaN** in the rare case of completely empty visits.

---

**Step 4: Divide to get mean**

```python
visit_vector = sum_embeddings / count
# Broadcasting: [2, 3] / [2, 1] → [2, 3]

# Patient 1:
visit_vector_1 = [1.5, 0.4, 1.0] / 3.0
               = [0.5, 0.133, 0.333]

# Patient 2:
visit_vector_2 = [1.0, -0.2, 0.8] / 2.0
               = [0.5, -0.1, 0.4]

# Final result
visit_vector = [
    [0.5, 0.133, 0.333],   # Patient 1: mean of 3 codes
    [0.5, -0.1, 0.4]        # Patient 2: mean of 2 codes
]
# Shape: [2, 3]  (batch, embed_dim)
```

---

#### Comparison: Naive vs. Masked Mean

**Scenario:** Patient with 3 codes + 1 padding

```python
embeddings = [
    [0.5, -0.3, 0.8],
    [0.3, 0.6, -0.2],
    [0.7, 0.1, 0.4],
    [0.0, 0.0, 0.0]  # Padding
]

# Naive mean (WRONG)
naive = embeddings.mean(dim=0)
      = sum / 4  # Divides by 4 (includes padding)
      = [1.5/4, 0.4/4, 1.0/4]
      = [0.375, 0.1, 0.25]
      # Signal diluted by 25% due to padding!

# Masked mean (CORRECT)
masked = masked_sum / 3  # Divides by 3 (only real codes)
       = [1.5/3, 0.4/3, 1.0/3]
       = [0.5, 0.133, 0.333]
       # True average of real codes
```

**Impact:** With 20% padding, naive mean is 20% weaker than correct mean. This can significantly hurt model performance!

---

#### Broadcasting Mechanics

**Key to understanding the code:**

```python
visit_embeddings: [batch, codes, embed_dim]
visit_mask: [batch, codes, 1]

# Multiplication (broadcasting)
masked_embeddings = visit_embeddings * visit_mask
# [batch, codes, embed_dim] * [batch, codes, 1]
# → [batch, codes, embed_dim]

# How broadcasting works:
# For each (batch, code) position:
#   embedding[i, j, :] * mask[i, j, 0]
#   All embed_dim elements multiplied by same mask value
```

**Example:**
```python
embedding = [[0.5, -0.3, 0.8]]  # [1, 1, 3]
mask = [[1.0]]                   # [1, 1, 1]

result = embedding * mask
# mask[0,0,0]=1.0 broadcasts to all 3 embedding dimensions
result = [[0.5*1.0, -0.3*1.0, 0.8*1.0]]
       = [[0.5, -0.3, 0.8]]  # All dimensions scaled by same value
```

---

## LSTMBaseline Model

### Architecture Components

```python
class LSTMBaseline(nn.Module):
    def __init__(self, ...):
        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. Visit encoder
        self.visit_encoder = VisitEncoder(embedding_dim, aggregation='mean')
        
        # 3. LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            bidirectional=False,
            batch_first=True
        )
        
        # 4. Prediction head
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()  # For binary classification
```

---

### Component Roles

#### 1. Embedding Layer

```python
self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
```

**Purpose:** Map discrete codes to continuous vectors

**Example:**
```python
vocab_size = 10000  # 10K unique medical codes
embedding_dim = 128

# Code indices
codes = torch.tensor([42, 523, 1200, 0])  # Last is padding
# After embedding
embeddings = embedding(codes)
# Shape: [4, 128]
# embeddings[0] = learned vector for code 42
# embeddings[3] = zero vector (padding_idx=0)
```

**Why `padding_idx=0`?**
```python
# Padding tokens (index 0) get zero embeddings
# Ensures padding doesn't contribute to visit representation
```

---

#### 2. Visit Encoder

```python
self.visit_encoder = VisitEncoder(
    embedding_dim=embedding_dim,
    aggregation='mean',  # or 'attention'
    dropout=dropout
)
```

**Purpose:** Aggregate codes within each visit

**Transform:**
```python
# Input: Variable codes per visit
visit_codes: [batch, codes, embed_dim]

# Output: Fixed visit vector
visit_vector: [batch, embed_dim]
```

---

#### 3. LSTM

```python
self.lstm = nn.LSTM(
    input_size=embedding_dim,      # 128
    hidden_size=hidden_dim,         # 256
    num_layers=2,                   # Stack 2 layers
    dropout=dropout,                # 0.3 between layers
    bidirectional=False,            # Causal (for prediction)
    batch_first=True                # [batch, seq, features]
)
```

**Purpose:** Model temporal dependencies across visits

**Input/Output:**
```python
# Input: Sequence of visit vectors
input: [batch, num_visits, embedding_dim]
       [32, 10, 128]

# Output: Hidden states at each visit
output: [batch, num_visits, hidden_dim]
        [32, 10, 256]

# Final states
h_n: [num_layers, batch, hidden_dim]
     [2, 32, 256]
```

**Why `bidirectional=False`?**
```python
# For prospective prediction, only use past information
# At visit t, can only see visits 0 to t (not future visits)
# Bidirectional would "cheat" by seeing future visits
```

---

#### 4. Prediction Head

```python
self.fc = nn.Linear(hidden_dim, output_dim)
self.activation = nn.Sigmoid()  # For binary classification
```

**Purpose:** Map final hidden state to prediction

**Example:**
```python
# Binary classification (disease yes/no)
output_dim = 1
final_hidden: [batch, hidden_dim] = [32, 256]
logits = fc(final_hidden)  # [32, 1]
predictions = sigmoid(logits)  # [32, 1] in range [0, 1]
```

**Task-specific activations:**
```python
# Binary classification
activation = nn.Sigmoid()  # Output in [0, 1]

# Multi-class classification (5 disease stages)
activation = nn.Softmax(dim=-1)  # Output sums to 1

# Regression (risk score)
activation = nn.Identity()  # Raw output
```

---

## Forward Pass Walkthrough

Let's trace a complete forward pass with concrete data.

### Example Data

```python
# 2 patients, 3 visits each, up to 4 codes per visit
batch_size = 2
num_visits = 3
max_codes = 4

visit_codes = torch.tensor([
    # Patient 1
    [
        [42, 523, 1200, 0],      # Visit 1: 3 codes + padding
        [42, 523, 1200, 1500],   # Visit 2: 4 codes
        [100, 200, 0, 0]         # Visit 3: 2 codes + padding
    ],
    # Patient 2
    [
        [50, 60, 0, 0],          # Visit 1: 2 codes + padding
        [50, 60, 70, 80],        # Visit 2: 4 codes
        [90, 0, 0, 0]            # Visit 3: 1 code + padding
    ]
])
# Shape: [2, 3, 4]

visit_mask = torch.tensor([
    # Patient 1
    [[1, 1, 1, 0],   # Visit 1: 3 real codes
     [1, 1, 1, 1],   # Visit 2: 4 real codes
     [1, 1, 0, 0]],  # Visit 3: 2 real codes
    # Patient 2
    [[1, 1, 0, 0],   # Visit 1: 2 real codes
     [1, 1, 1, 1],   # Visit 2: 4 real codes
     [1, 0, 0, 0]]   # Visit 3: 1 real code
])
# Shape: [2, 3, 4]
```

---

### Step 1: Embed Codes

```python
# Input
visit_codes: [2, 3, 4]

# Embedding layer
code_embeddings = self.embedding(visit_codes)
# Shape: [2, 3, 4, 128]
#        [batch, visits, codes, embed_dim]

# Each code now has a 128-dim vector
# Example for code 42:
# embeddings[0, 0, 0] = learned_vector_for_code_42
```

---

### Step 2: Flatten for Visit Encoding

```python
# Need to process all visits at once (batched operation)
# Reshape: [batch, visits, codes, embed] → [batch*visits, codes, embed]

code_embeddings_flat = code_embeddings.view(
    batch_size * num_visits,  # 2 * 3 = 6
    max_codes,                 # 4
    self.embedding_dim         # 128
)
# Shape: [6, 4, 128]

# Now treating each visit independently:
# [
#   Patient 1 Visit 1,  # Index 0
#   Patient 1 Visit 2,  # Index 1
#   Patient 1 Visit 3,  # Index 2
#   Patient 2 Visit 1,  # Index 3
#   Patient 2 Visit 2,  # Index 4
#   Patient 2 Visit 3   # Index 5
# ]

# Similarly flatten mask
visit_mask_flat = visit_mask.view(6, 4)
```

---

### Step 3: Encode Visits

```python
visit_vectors = self.visit_encoder(code_embeddings_flat, visit_mask_flat)
# Shape: [6, 128]

# Each visit is now a single 128-dim vector
# Example for Patient 1 Visit 1 (3 codes):
# visit_vectors[0] = mean([emb_42, emb_523, emb_1200])

# Reshape back to [batch, visits, embed_dim]
visit_vectors = visit_vectors.view(2, 3, 128)
# Shape: [2, 3, 128]
```

---

### Step 4: LSTM Processing

```python
# Input: Sequence of visit vectors
# Shape: [2, 3, 128]
#        [batch, visits, embed_dim]

lstm_output, (hidden, cell) = self.lstm(visit_vectors)

# lstm_output: [2, 3, 256]
#              [batch, visits, hidden_dim]
# Contains hidden states for all 3 visits:
# lstm_output[0, 0] = hidden state after visit 1 (patient 1)
# lstm_output[0, 1] = hidden state after visit 2 (patient 1)
# lstm_output[0, 2] = hidden state after visit 3 (patient 1)

# hidden: [2, 2, 256]
#         [num_layers, batch, hidden_dim]
# hidden[0] = final hidden of layer 1
# hidden[1] = final hidden of layer 2

# cell: [2, 2, 256]
#       [num_layers, batch, hidden_dim]
```

---

### Step 5: Extract Final Hidden State

```python
# For prediction, use last layer's final hidden
if self.bidirectional:
    # Bidirectional: concatenate forward and backward
    final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
    # Shape: [batch, 2*hidden_dim]
else:
    # Unidirectional: just last layer
    final_hidden = hidden[-1]
    # Shape: [2, 256]
    #        [batch, hidden_dim]

# final_hidden[0] = Patient 1's final state (after 3 visits)
# final_hidden[1] = Patient 2's final state (after 3 visits)
```

---

### Step 6: Prediction

```python
# Apply dropout
final_hidden = self.dropout(final_hidden)
# Shape: [2, 256]

# Linear layer
logits = self.fc(final_hidden)
# Shape: [2, 1]  (for binary classification)
# logits[0] = raw score for patient 1
# logits[1] = raw score for patient 2

# Activation
predictions = self.activation(logits)
# Shape: [2, 1]
# predictions[0] = P(disease | patient 1) ∈ [0, 1]
# predictions[1] = P(disease | patient 2) ∈ [0, 1]

# Example output:
# predictions = [[0.85],   # Patient 1: 85% disease probability
#                [0.23]]   # Patient 2: 23% disease probability
```

---

### Complete Flow Diagram

```
Input: visit_codes [2, 3, 4]
       ↓
Embedding: [2, 3, 4, 128]
       ↓
Flatten: [6, 4, 128]
       ↓
VisitEncoder (mean aggregation):
  - Mask padding codes
  - Compute mean of real codes
  - Output: [6, 128]
       ↓
Reshape: [2, 3, 128]
       ↓
LSTM:
  - Process visit sequence
  - Output: [2, 3, 256]
  - Final hidden: [2, 256]
       ↓
Prediction head:
  - Linear: [2, 1]
  - Sigmoid: [2, 1]
       ↓
Predictions: [[0.85], [0.23]]
```

---

## Design Decisions

### 1. Why Visit-Level Modeling?

**Alternative: Flatten all codes into one sequence**
```python
# Bad approach
all_codes = [visit1_code1, visit1_code2, ..., visit2_code1, visit2_code2, ...]
# Shape: [batch, total_codes]
# Feed directly to LSTM
```

**Problems:**
1. Loses visit structure (clinical reality)
2. Variable total length (hard to batch)
3. No clear boundary between visits
4. Can't predict per-visit outcomes

**Visit-level approach (better):**
```python
# Good approach
visits = [visit1_vector, visit2_vector, visit3_vector]
# Shape: [batch, num_visits, embed_dim]
# Each visit is a meaningful unit
```

**Benefits:**
1. Respects clinical structure
2. Fixed-length visit sequences (easier to batch)
3. Can predict at each visit
4. Interpretable (visit 5 contributed X to prediction)

---

### 2. Why Multiple Aggregation Methods?

Different clinical scenarios benefit from different aggregations:

**Mean:** General-purpose
```python
# Use when: All codes equally important
# Example: Routine checkup with standard tests
```

**Max:** Capture strongest signal
```python
# Use when: One critical code dominates
# Example: Emergency visit with acute event
```

**Attention:** Learn importance
```python
# Use when: Code importance varies and should be learned
# Example: Complex multi-morbid patients
```

---

### 3. Why Unidirectional LSTM?

**Bidirectional** (sees future):
```python
# At visit 3, LSTM sees visits 1-5
# Good for: Retrospective analysis
# Bad for: Real-time prediction (cheating!)
```

**Unidirectional** (causal):
```python
# At visit 3, LSTM sees only visits 1-3
# Good for: Real-time prediction, deployment
# Matches clinical reality: can't see future
```

**When to use bidirectional:**
- Post-hoc analysis (all visits known)
- Retrospective phenotyping
- Research studies

**When to use unidirectional:**
- Real-time risk prediction
- Clinical decision support
- Prospective studies

---

### 4. Why 2 LSTM Layers?

**Single layer:**
```python
num_layers = 1
# Pros: Faster, fewer parameters
# Cons: Limited expressiveness
```

**Two layers:**
```python
num_layers = 2
# Pros: Hierarchical representations
#   - Layer 1: Basic temporal patterns
#   - Layer 2: Higher-level abstractions
# Cons: Slower, more parameters
```

**Empirical findings:**
- 2 layers: Good balance for EHR data
- 3+ layers: Marginal gains, risk overfitting
- 1 layer: Often sufficient for small datasets

---

### 5. Why Packing Padded Sequences?

**Without packing:**
```python
# Patient 1: 8 visits
# Patient 2: 12 visits
# Pad to 12 visits, LSTM processes all 12 for both patients
# Wastes computation on 4 padded visits for patient 1
```

**With packing:**
```python
packed = pack_padded_sequence(visit_vectors, lengths=[8, 12], ...)
# LSTM only processes real visits
# Patient 1: 8 steps
# Patient 2: 12 steps
# Faster and more correct!
```

**Benefits:**
1. **Speed:** Skip padding computations
2. **Correctness:** Hidden states don't see padding
3. **Memory:** More efficient

---

## Summary

### Key Takeaways

1. **VisitEncoder aggregates codes within visits**
   - Mean aggregation: Masked average (handle padding correctly)
   - Attention aggregation: Learned importance weights (two-layer network for expressiveness)
   - Mask initialization: Default to all 1s if not provided

2. **LSTMBaseline models visit sequences**
   - Embedding → VisitEncoder → LSTM → Prediction
   - Respects clinical structure (visits as units)
   - Handles variable lengths (packing)

3. **Masked operations are crucial**
   - Padding must be explicitly handled
   - Broadcasting mechanics enable efficient masking
   - Always mask before aggregation

4. **Design is flexible**
   - Multiple aggregation strategies
   - Task-agnostic (classification, regression)
   - Bidirectional optional

### Common Pitfalls to Avoid

1. ❌ Using naive `.mean()` without masking
   - ✅ Always use masked mean with padding

2. ❌ Creating mask with wrong shape
   - ✅ Mask shape: [batch, codes], not [batch, codes, embed_dim]

3. ❌ Device mismatch (CPU vs GPU)
   - ✅ Always create tensors on same device

4. ❌ Single-layer attention (limited expressiveness)
   - ✅ Use two-layer with bottleneck for nonlinearity

5. ❌ Not packing variable-length sequences
   - ✅ Use pack_padded_sequence for efficiency

---

## Code Reference Summary

| Component | Purpose | Key Method |
|-----------|---------|------------|
| **VisitEncoder** | Aggregate codes → visit vector | Mean, attention, max pooling |
| **Attention** | Learn code importance | Two-layer MLP with bottleneck |
| **Masking** | Handle padding | Multiply by mask, masked_fill |
| **LSTM** | Temporal modeling | Process visit sequence |
| **Packing** | Efficiency | pack_padded_sequence |

This architecture provides a strong baseline for visit-grouped EHR modeling, balancing simplicity with effectiveness.
