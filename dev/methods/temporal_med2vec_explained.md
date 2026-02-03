# Temporal Med2Vec: Understanding Temporally-Aware Medical Code Embeddings

**Date:** January 20, 2026  
**Source:** `docs/methods/modern-code-embeddings.md`  
**Purpose:** Private development notes on TemporalMed2Vec implementation

---

## Overview

**TemporalMed2Vec** is an extension of the Med2Vec architecture that incorporates **temporal awareness** into medical code embeddings. It learns vector representations of medical codes (ICD, LOINC, RXNORM, etc.) from EHR sequences while accounting for time gaps between events.

**Key Innovation:** Unlike traditional Word2Vec/Med2Vec that treat context windows uniformly, TemporalMed2Vec **downweights** codes that are temporally distant using exponential decay.

---

## Core Concepts

### 1. Context Codes (Your Understanding: ✓ CORRECT)

**Definition:** Context codes are medical codes that appear **within a sliding window** around a target code in a patient's temporal sequence.

**Example:**

```
Patient Timeline:
  t=0   days: Code A
  t=10  days: Code B
  t=15  days: Code C (TARGET)
  t=20  days: Code D
  t=30  days: Code E

With window_size=2:
  Target: Code C
  Context: [Code B, Code D] (within ±2 positions)
```

**Key Points:**
- Context is bidirectional: looks both backward and forward in time
- Window size controls how many neighboring events to consider
- Mimics Word2Vec's context window but on temporal event sequences

---

### 2. Time Deltas and Temporal Weighting (Your Understanding: ✓ CORRECT)

**Definition:** Time delta (Δt) is the **absolute time difference** (in days) between a target code and each context code. It's used to **attenuate** (reduce) the influence of temporally distant codes.

**Mechanism: Exponential Decay**

```
weight = exp(-λ * Δt)

where:
  λ (lambda) = time_decay parameter (e.g., 0.01)
  Δt = time difference in days
```

**Example:**

```
Target: Code C at day 15
Context codes with time deltas:
  - Code B: Δt = |15 - 10| = 5 days  → weight = exp(-0.01 * 5)  = 0.951
  - Code D: Δt = |20 - 15| = 5 days  → weight = exp(-0.01 * 5)  = 0.951
  - Code E: Δt = |30 - 15| = 15 days → weight = exp(-0.01 * 15) = 0.861
```

**Intuition:**
- Codes close in time (small Δt) → weight ≈ 1 → **strong influence**
- Codes far in time (large Δt) → weight < 1 → **reduced influence**
- This captures the clinical reality that recent events are more predictive

---

## Architecture Deep Dive

### Model Components

```python
class TemporalMed2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, time_decay=0.1):
        super().__init__()
        # Two embedding matrices (like Word2Vec)
        self.code_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.time_decay = time_decay  # λ parameter
```

**Why Two Embedding Matrices?**
- `code_embeddings`: Represents codes as **target** predictions
- `context_embeddings`: Represents codes as **context** for prediction
- Separating these improves expressivity (standard in skip-gram models)

---

### Forward Pass Breakdown

```python
def forward(self, target_code, context_codes, time_deltas):
    # Step 1: Embed target and context codes
    target_embed = self.code_embeddings(target_code)      # [batch, embed_dim]
    context_embed = self.context_embeddings(context_codes) # [batch, context, embed_dim]
    
    # Step 2: Compute similarity scores (dot product)
    scores = torch.einsum('be,bce->bc', target_embed, context_embed)
    # einsum notation: b=batch, e=embed_dim, c=context_size
    # Result: [batch, context] - similarity of target to each context code
    
    # Step 3: Apply temporal decay
    temporal_weights = torch.exp(-self.time_decay * time_deltas)
    # Shape: [batch, context]
    # Closer codes → higher weight, distant codes → lower weight
    
    # Step 4: Compute weighted loss
    log_probs = torch.log_softmax(scores, dim=-1)  # Convert to probabilities
    weighted_loss = -(log_probs * temporal_weights).sum() / temporal_weights.sum()
    # Negative log-likelihood weighted by temporal importance
    
    return weighted_loss
```

**Step-by-Step Example:**

```
Batch size = 1, Context size = 3, Embed dim = 4

Target: Code 42 at day 100
Context: [(Code 15, day 95), (Code 23, day 102), (Code 8, day 110)]

Time deltas: [5, 2, 10] days
Temporal weights (λ=0.1): [exp(-0.5)=0.606, exp(-0.2)=0.819, exp(-1.0)=0.368]

Target embedding:     [0.5, -0.3, 0.2, 0.8]
Context embeddings:
  Code 15: [0.4, -0.2, 0.1, 0.7] → dot product = 0.73 → prob ≈ 0.35
  Code 23: [0.6, -0.4, 0.3, 0.9] → dot product = 1.05 → prob ≈ 0.50
  Code 8:  [0.1, 0.1, -0.1, 0.3] → dot product = 0.18 → prob ≈ 0.15

Weighted loss = -(log(0.35)*0.606 + log(0.50)*0.819 + log(0.15)*0.368) / (0.606+0.819+0.368)
              = Emphasizes Code 23 (recent, high prob) more than Code 8 (distant, low prob)
```

---

## Training Procedure

### Data Preparation: Creating Training Triplets

```python
def create_temporal_training_pairs(patient_sequences, window_size=5):
    """
    Converts patient timelines into (target, context, time_delta) triplets.
    
    Input: patient_sequences = [
        [(timestamp_1, code_1), (timestamp_2, code_2), ...],  # Patient 1
        [(timestamp_1, code_1), (timestamp_2, code_2), ...],  # Patient 2
        ...
    ]
    
    Output: triplets = [
        (target_code, [context_code_1, ...], [delta_1, ...]),
        ...
    ]
    """
    triplets = []
    
    for sequence in patient_sequences:
        # For each code in the patient timeline
        for i, (target_time, target_code) in enumerate(sequence):
            context_codes = []
            time_deltas = []
            
            # Define context window: [i-window_size, i+window_size]
            # Excludes the target itself (i != j)
            for j in range(max(0, i - window_size), 
                          min(len(sequence), i + window_size + 1)):
                if i == j:
                    continue  # Skip the target code
                
                context_time, context_code = sequence[j]
                time_delta = abs((target_time - context_time).days)
                
                context_codes.append(context_code)
                time_deltas.append(time_delta)
            
            if context_codes:  # Only add if context exists
                triplets.append((target_code, context_codes, time_deltas))
    
    return triplets
```

**Concrete Example:**

```
Patient P001 Timeline:
  2020-01-01 | Code A
  2020-01-10 | Code B
  2020-01-15 | Code C
  2020-01-20 | Code D
  2020-02-01 | Code E

With window_size = 2:

Triplet 1 (target = Code A, i=0):
  Context: [Code B (i=1), Code C (i=2)]
  Deltas:  [9 days, 14 days]

Triplet 2 (target = Code B, i=1):
  Context: [Code A (i=0), Code C (i=2), Code D (i=3)]
  Deltas:  [9 days, 5 days, 10 days]

Triplet 3 (target = Code C, i=2):
  Context: [Code A (i=0), Code B (i=1), Code D (i=3), Code E (i=4)]
  Deltas:  [14 days, 5 days, 5 days, 17 days]

... and so on
```

---

### Training Loop

```python
# Initialize model
model = TemporalMed2Vec(
    vocab_size=10000,      # Number of unique medical codes
    embed_dim=128,         # Embedding dimension
    time_decay=0.01        # λ parameter (tune via validation)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(50):
    for target, context, deltas in dataloader:
        # Forward pass: compute loss
        loss = model(target, context, deltas)
        
        # Backward pass: update embeddings
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**After Training:**
- `model.code_embeddings.weight` contains learned embeddings
- Codes with similar temporal co-occurrence patterns have similar vectors
- Can use for downstream tasks: prediction, clustering, visualization

---

## Key Hyperparameters

### 1. `time_decay` (λ)

**Effect:**
- **Small λ (e.g., 0.001)**: Slow decay → distant codes still influential
- **Large λ (e.g., 0.1)**: Fast decay → only recent codes matter

**Tuning Strategy:**
- Start with λ = 0.01 (half-life ≈ 70 days)
- For acute conditions: larger λ (faster decay)
- For chronic conditions: smaller λ (longer memory)

**Half-life calculation:**
```
t_half = ln(2) / λ

λ = 0.01  → t_half = 69 days
λ = 0.1   → t_half = 7 days
λ = 0.001 → t_half = 693 days
```

### 2. `window_size`

**Effect:**
- Small window (e.g., 3): Local context only
- Large window (e.g., 10): Capture long-range dependencies

**Recommendation:**
- ICU/acute: window_size = 3-5
- Outpatient/chronic: window_size = 10-20

### 3. `embed_dim`

**Common choices:** 64, 128, 256
- Larger dimensions: More expressivity but slower training
- Typically 128 is a good default

---

## Comparison with Standard Med2Vec

| Feature | Standard Med2Vec | Temporal Med2Vec |
|---------|------------------|------------------|
| Context window | Fixed size, uniform weights | Fixed size, time-weighted |
| Temporal info | Ignored | Exponential decay based on Δt |
| Loss | Uniform negative log-likelihood | Weighted negative log-likelihood |
| Use case | General co-occurrence | Time-sensitive sequences |

**When to use Temporal Med2Vec:**
- Disease progression modeling
- Time-critical events (e.g., post-surgery complications)
- Chronic condition tracking with irregularly spaced visits

**When standard Med2Vec suffices:**
- Only co-occurrence matters (not timing)
- All events within window are equally important
- Smaller datasets where temporal weighting adds noise

---

## Practical Considerations

### 1. Handling Missing Time Information

If some events lack precise timestamps:
```python
# Option 1: Use visit order as proxy
time_delta = abs(i - j)  # Position difference instead of days

# Option 2: Assign default time gaps
time_delta = 30 * abs(i - j)  # Assume 30 days between visits
```

### 2. Very Long Time Gaps

For events years apart, exponential decay → near-zero weights:
```python
# Clip time deltas to prevent vanishing weights
time_deltas = torch.clamp(time_deltas, max=365)  # Cap at 1 year
```

### 3. Batch Padding

Context sizes vary across patients. Need padding:
```python
from torch.nn.utils.rnn import pad_sequence

# Pad context_codes and time_deltas to same length
context_codes_padded = pad_sequence(context_codes, batch_first=True)
time_deltas_padded = pad_sequence(time_deltas, batch_first=True)
```

---

## Extension Ideas

### 1. Visit-Level Embeddings

Instead of individual codes, embed entire visits:
```python
visit_embed = temporal_pooling(code_embeddings[visit_codes], visit_time_deltas)
```

### 2. Bidirectional Temporal Weighting

Different decay for past vs. future:
```python
if time_delta < 0:  # Past event
    weight = exp(-lambda_past * abs(time_delta))
else:  # Future event
    weight = exp(-lambda_future * abs(time_delta))
```

### 3. Hierarchical Codes

Incorporate ICD-10 hierarchy:
```python
# Add parent code embeddings
parent_code = get_parent_icd10(target_code)
loss += hierarchy_loss(embed[target_code], embed[parent_code])
```

---

## Summary

**Your Understanding is 100% Correct:**

1. **Context codes** = codes within a sliding window around the target
2. **Time deltas** = used to attenuate influence of temporally distant codes via exponential decay

**Key Takeaway:** TemporalMed2Vec extends skip-gram embeddings with temporal awareness, making learned representations sensitive to the **timing** of medical events, not just their co-occurrence. This is critical for modeling disease progression, treatment responses, and other time-dependent clinical phenomena.

**Next Steps:**
- Experiment with different λ values for your clinical domain
- Compare learned embeddings with standard Med2Vec on downstream tasks
- Visualize embeddings (t-SNE) to see temporal clustering

---

## References

- Original Med2Vec: Choi et al. (2016) - "Multi-layer Representation Learning for Medical Concepts"
- Skip-gram model: Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space"
- Temporal embeddings in healthcare: Multiple recent works (2020-2024) incorporating time into sequence models
