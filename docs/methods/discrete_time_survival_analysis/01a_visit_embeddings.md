# Visit Embeddings: Two Conceptually Different Representations

**Last Updated:** 2026-02-03  
**Topics:** Visit ID embeddings, aggregated visit embeddings, scatter_add mechanics

---

## Overview

This document clarifies a crucial conceptual distinction that often causes confusion: **Visit ID embeddings** (input-side signal) vs **Aggregated visit embeddings** (output-side representation).

These live at completely different levels of abstraction and serve fundamentally different purposes in the model pipeline.

**Key insight:** Visit ID embedding tells the model "this token is from visit 3," while aggregated visit embedding represents "visit 3 contains these specific conditions with their learned relationships."

---

## Table of Contents

1. [Visit ID Embedding (Input-Side Signal)](#1-visit-id-embedding-input-side-signal)
2. [Aggregated Visit Embedding (Output-Side Representation)](#2-aggregated-visit-embedding-output-side-representation)
3. [Direct Comparison](#3-direct-comparison)
4. [How scatter_add Works](#4-how-scatter_add-works)
5. [Why This Distinction Matters](#5-why-this-distinction-matters)
6. [Optimization Implications](#6-optimization-implications)

---

## 1. Visit ID Embedding (Input-Side Signal)

### What It Is

A **learnable lookup table** that maps visit index to an embedding vector, added to the token representation before the transformer.

### Mathematical Definition

**Embedding matrices:**

```
E^code   ∈ ℝ^(|V| × d)        # Code embeddings
E^age    ∈ ℝ^(A × d)          # Age embeddings
E^visit  ∈ ℝ^(V_max × d)      # Visit ID embeddings
E^pos    ∈ ℝ^(L_max × d)      # Positional embeddings
```

Where:
- |V| = vocabulary size
- d = embedding dimension
- A = number of age bins
- V_max = maximum visit index
- L_max = maximum sequence length

**For token i in the sequence:**

```
x_i = E^code[c_i] + E^age[a_i] + E^visit[v_i] + E^pos[p_i]
```

Where:
- c_i = code ID ∈ {0, 1, ..., |V|-1}
- a_i = age bin index
- v_i = visit ID ∈ {0, 1, ..., V_max-1}
- p_i = position index
- x_i ∈ ℝ^d = final token embedding

### What It Represents

**Interpretation:**

> "This code occurred in visit 3 rather than visit 1."

The visit ID embedding injects visit index as **contextual prior information**.

**It does NOT:**
- ❌ Summarize the visit
- ❌ Represent the visit's content
- ❌ Know what codes are in the visit

**It DOES:**
- ✅ Add a learned bias vector to each token
- ✅ Provide temporal grouping information
- ✅ Help the model distinguish visit boundaries

### Properties

| Property | Value |
|----------|-------|
| **Level** | Token-level |
| **Type** | Input feature |
| **When created** | Before transformer |
| **Learned from** | All training tasks (MLM, survival, etc.) |
| **Same for all tokens in visit?** | Yes (lookup by visit index) |
| **Shape** | (V_max × d) |

### Implementation

```python
# From embeddings.py
class BEHRTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ...
        self.visit_embedding = nn.Embedding(
            config.max_visits, 
            config.embedding_dim
        )
    
    def forward(self, codes, ages, visit_ids):
        # Lookup embeddings
        code_emb = self.code_embedding(codes)        # [B, L, d]
        age_emb = self.age_embedding(ages)          # [B, L, d]
        visit_emb = self.visit_embedding(visit_ids) # [B, L, d]
        pos_emb = self.positional_embedding(...)    # [B, L, d]
        
        # Sum all components
        embeddings = code_emb + age_emb + visit_emb + pos_emb
        
        # Normalize and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings  # [B, L, d]
```

---

## 2. Aggregated Visit Embedding (Output-Side Representation)

### What It Is

A **computed representation** that aggregates contextualized token embeddings from the transformer to create a visit-level vector.

### Mathematical Definition

**After transformer encoding:**

```
H ∈ ℝ^(B × L × d)
```

Where:
- B = batch size
- L = sequence length (number of tokens)
- d = embedding dimension
- H_{b,i} ∈ ℝ^d = contextual embedding of token i in batch b

**These embeddings now encode:**
- Code meaning (from pre-training)
- Temporal context (from attention)
- Co-morbid structure (from cross-attention)
- Long-range dependencies (from full history)

**Visit-level aggregation:**

For patient b with T_b visits, where visit t contains tokens I_{b,t}:

```
V_{b,t} = (1 / |I_{b,t}|) * Σ_{i ∈ I_{b,t}} H_{b,i}
```

This is **mean pooling per visit**.

Where:
- I_{b,t} = {i : visit_ids[b,i] = t and attention_mask[b,i] = 1}
- V_{b,t} ∈ ℝ^d = visit representation
- |I_{b,t}| = number of codes in visit t

### What It Represents

**Interpretation:**

> "Visit 3 contains hypertension, CKD, elevated creatinine, and anemia — and attention has already contextualized them with the patient's history."

This is a **semantic representation** of the visit.

**It DOES:**
- ✅ Represent the visit's medical content
- ✅ Incorporate learned relationships between codes
- ✅ Capture context from patient history
- ✅ Depend on actual codes present

**It does NOT:**
- ❌ Simply look up visit index
- ❌ Use the same vector for different visits
- ❌ Exist before transformer computation

### Properties

| Property | Value |
|----------|-------|
| **Level** | Visit-level |
| **Type** | Output representation |
| **When created** | After transformer |
| **Learned from** | Contextual modeling + task-specific loss |
| **Same for all visits?** | No (depends on content) |
| **Shape** | (B × T_max × d) |

### Implementation

```python
# From behrt_survival.py (simplified)
def aggregate_to_visits(self, hidden_states, visit_ids, attention_mask):
    """
    Aggregate token embeddings to visit embeddings.
    
    Args:
        hidden_states: [B, L, d] - contextual token embeddings
        visit_ids: [B, L] - visit index for each token
        attention_mask: [B, L] - 1 for real tokens, 0 for padding
    
    Returns:
        visit_embeddings: [B, T_max, d]
    """
    B, L, d = hidden_states.shape
    T_max = visit_ids.max() + 1
    
    # Initialize output
    visit_sums = torch.zeros(B, T_max, d, device=hidden_states.device)
    visit_counts = torch.zeros(B, T_max, device=hidden_states.device)
    
    # Mask out padding
    valid_mask = attention_mask.bool()
    
    # Scatter-add: group tokens by visit
    visit_sums.scatter_add_(
        dim=1,
        index=visit_ids.unsqueeze(-1).expand(-1, -1, d),
        src=hidden_states * valid_mask.unsqueeze(-1)
    )
    
    # Count tokens per visit
    visit_counts.scatter_add_(
        dim=1,
        index=visit_ids,
        src=valid_mask.float()
    )
    
    # Mean pooling: divide by count
    visit_embeddings = visit_sums / (visit_counts.unsqueeze(-1) + 1e-8)
    
    return visit_embeddings  # [B, T_max, d]
```

---

## 3. Direct Comparison

### Side-by-Side

| Aspect | Visit ID Embedding | Aggregated Visit Embedding |
|--------|-------------------|---------------------------|
| **Conceptual level** | Input signal | Output representation |
| **When computed** | Before transformer | After transformer |
| **How computed** | Learned lookup | Computed from outputs |
| **Dependencies** | Visit index only | Actual codes + context |
| **Uniqueness** | Same vector for all tokens in visit | Different for each visit based on content |
| **Shape** | (V_max × d) - parameter matrix | (B × T × d) - activation tensor |
| **Trainable** | Yes (embedding matrix) | No (computed from trainable H) |
| **Gradients** | Direct from loss | Through transformer from loss |

### Analogy

Think of them as:

**Visit ID embedding:**
> "Street number" — tells you the address, not what's inside

**Aggregated visit embedding:**
> "Summary of what happened inside the building" — actual content

### Information Flow

```
Input:
    visit_ids → Visit ID Embedding → Added to token representation

Processing:
    Token embeddings → Transformer → Contextual embeddings H

Output:
    H + visit_ids → scatter_add → Aggregated visit embeddings V
```

---

## 4. How scatter_add Works

### The Problem

We want to compute:

```
V_{b,t} = Σ_{i ∈ I_{b,t}} H_{b,i}
```

without Python loops (vectorized for GPU efficiency).

### Shapes Overview

**Inputs:**
```
H           ∈ ℝ^(B × L × d)    # Contextual embeddings
visit_ids   ∈ ℤ^(B × L)        # Visit index for each token
```

**Output:**
```
V ∈ ℝ^(B × T_max × d)          # Visit embeddings
```

### Mathematical Operation

`scatter_add` performs a **vectorized group-by-sum**.

**Conceptually:**

```python
for b in range(B):
    for i in range(L):
        t = visit_ids[b, i]
        output[b, t] += H[b, i]
```

**Mathematically:**

```
V_{b,t} = Σ_{i=1}^L 𝟙(visit_ids[b,i] = t) · H_{b,i}
```

Where:
- 𝟙(·) is the indicator function
- Returns 1 if condition true, 0 otherwise

### PyTorch Implementation

```python
# Initialize output
output = torch.zeros(B, T_max, d)

# Expand visit_ids to match hidden_states dimensions
index = visit_ids.unsqueeze(-1).expand(-1, -1, d)  # [B, L, d]

# Group-by-sum via scatter_add
output.scatter_add_(
    dim=1,           # Scatter along visit dimension
    index=index,     # Which visit each token belongs to
    src=H           # Source values to add
)
```

**What happens:**
- For each position (b, i) in H
- Look up visit index: t = index[b, i, :]
- Add H[b, i] to output[b, t]

### Mean Pooling

**After summing, we need to divide by visit size:**

```python
# Count tokens per visit
visit_counts = torch.zeros(B, T_max)
visit_counts.scatter_add_(
    dim=1,
    index=visit_ids,
    src=torch.ones_like(visit_ids, dtype=torch.float)
)

# Mean pooling
V = output / (visit_counts.unsqueeze(-1) + eps)
```

**Final formula:**

```
V_{b,t} = (Σ_{i=1}^L 𝟙(visit_ids[b,i] = t) · H_{b,i}) / count_{b,t}
```

Where:

```
count_{b,t} = Σ_{i=1}^L 𝟙(visit_ids[b,i] = t)
```

### Masking for Padding

**Critical:** Must exclude padding tokens!

```python
# Create valid mask
valid_mask = attention_mask.bool()  # [B, L]

# Mask hidden states before aggregation
H_masked = H * valid_mask.unsqueeze(-1)  # [B, L, d]

# Now scatter-add with masked values
output.scatter_add_(dim=1, index=index, src=H_masked)

# Count only valid tokens
visit_counts.scatter_add_(
    dim=1,
    index=visit_ids,
    src=valid_mask.float()
)
```

**With masking, the formula becomes:**

```
V_{b,t} = (Σ_{i=1}^L 𝟙(visit_ids[b,i]=t) · 𝟙(mask[b,i]=1) · H_{b,i}) 
          / (Σ_{i=1}^L 𝟙(visit_ids[b,i]=t) · 𝟙(mask[b,i]=1))
```

Both conditions must be satisfied!

---

## 5. Why This Distinction Matters

### Conceptual Clarity

**Visit ID embedding says:**
> "You are in visit #3."

**Aggregated visit embedding says:**
> "Visit #3 contains these specific conditions with their learned relationships."

One is **positional metadata**.  
The other is **semantic representation**.

One is injected **before learning**.  
The other is **learned by contextual modeling**.

### Different Roles in Model

**Visit ID embedding:**
- Influences representation before contextualization
- Provides inductive bias about temporal structure
- Same information for all patients at visit 3

**Aggregated visit embedding:**
- Result of contextual reasoning
- Contains patient-specific medical information
- Different for each patient even at same visit index

### Example

**Patient A, Visit 2:**
- Codes: Diabetes (250.00), Hypertension (401.9)
- Visit ID embedding: E^visit[2] (same for all visit 2s)
- Aggregated embedding: Mean of contextual embeddings of diabetes + hypertension codes

**Patient B, Visit 2:**
- Codes: Pneumonia (486), Fever (780.6), Cough (786.2)
- Visit ID embedding: E^visit[2] (same as Patient A!)
- Aggregated embedding: Mean of contextual embeddings of pneumonia + fever + cough (completely different!)

### Optimization Perspective

They play **totally different roles in optimization**:

**Visit ID embeddings:**
- Get gradients through ALL training tasks
- Learn temporal position encoding
- Shared across all patients

**Aggregated visit embeddings:**
- Get gradients through survival loss only
- Learn task-specific representations
- Patient-specific

---

## 6. Optimization Implications

### Gradient Flow

**For Visit ID embeddings:**

```
Loss → Head → Transformer → Token embeddings → Visit ID embedding matrix
```

Gradients flow through token-level losses (MLM, survival, etc.).

**For Aggregated visit embeddings:**

```
Loss → Aggregation → Transformer outputs
```

Gradients flow through survival loss only.

### Why scatter_add is Differentiable

**Summation is linear**, so gradients flow cleanly:

```
∂L/∂H_{b,i} = ∂L/∂V_{b,t} · ∂V_{b,t}/∂H_{b,i}
```

For mean pooling:

```
∂V_{b,t}/∂H_{b,i} = 1/count_{b,t}  if i ∈ I_{b,t}
                  = 0                otherwise
```

**This is clean and stable** — no vanishing gradients, no numerical issues.

### Backpropagation Example

**Forward pass:**
```
Token embeddings x → Transformer → H → scatter_add → V → Hazard MLP → Loss
```

**Backward pass:**
```
∂L/∂loss = 1
∂L/∂hazard = ∂L/∂loss · ∂loss/∂hazard
∂L/∂V = ∂L/∂hazard · ∂hazard/∂V
∂L/∂H = ∂L/∂V · ∂V/∂H  (via scatter_add)
∂L/∂x = ∂L/∂H · ∂H/∂x  (via transformer)
```

All operations are differentiable with stable gradients.

---

## 7. Complete Pipeline

### Information Flow

```
Input Data:
    codes: [101, 250, 401] (diabetes, hypertension at visit 1)
    visit_ids: [0, 0, 0]
    
↓ Embedding Layer

Token Embeddings:
    x_0 = E^code[101] + E^age[45] + E^visit[0] + E^pos[0]
    x_1 = E^code[250] + E^age[45] + E^visit[0] + E^pos[1]
    x_2 = E^code[401] + E^age[45] + E^visit[0] + E^pos[2]
    
↓ Transformer

Contextual Embeddings:
    H_0 = f_transformer(x_0, x_1, x_2)  # Attended to all codes
    H_1 = f_transformer(x_0, x_1, x_2)  # Attended to all codes
    H_2 = f_transformer(x_0, x_1, x_2)  # Attended to all codes
    
↓ scatter_add Aggregation

Visit Embedding:
    V_0 = (H_0 + H_1 + H_2) / 3  # Mean of visit 0
    
↓ Hazard Head

Hazard:
    h_0 = sigmoid(MLP(V_0))  # Risk at visit 0
```

### Key Observations

1. **Visit ID embedding (E^visit[0])** is added to all three tokens BEFORE transformer
2. **Aggregated visit embedding (V_0)** is computed AFTER transformer from all three contextual outputs
3. They serve completely different purposes in different stages of the pipeline

---

## 8. Common Misconceptions

### ❌ Misconception 1: "Visit embedding = aggregated embedding"

**Wrong:** They are fundamentally different objects at different pipeline stages.

**Correct:** Visit ID embedding is input metadata; aggregated embedding is output representation.

### ❌ Misconception 2: "Aggregated embedding is just a sum of visit ID embeddings"

**Wrong:** Aggregated embedding comes from transformer outputs, not input embeddings.

**Correct:** Aggregated embedding = mean of contextualized token embeddings.

### ❌ Misconception 3: "scatter_add is just groupby"

**Partially correct:** It's vectorized groupby-sum, but requires careful masking and mean normalization.

### ❌ Misconception 4: "Visit ID embeddings are redundant with positional embeddings"

**Wrong:** They serve different purposes.

**Correct:**
- Positional embedding: "token 5 in sequence"
- Visit ID embedding: "this token is from visit 2"

Multiple tokens can share the same visit ID but have different positions.

---

## 9. Advanced Topics

### Alternative Aggregation Methods

**Current:** Mean pooling

**Alternatives:**

1. **Attention-based pooling:**
```python
# Learn attention weights for aggregation
α = softmax(W_attn @ H)  # [B, L]
V_t = Σ α_i · H_i
```

2. **Max pooling:**
```python
V_t = max_{i ∈ I_{b,t}} H_{b,i}
```

3. **CLS token per visit:**
```python
# Add special [CLS] token at start of each visit
V_t = H_{cls_token_for_visit_t}
```

**Trade-offs:**
- Mean pooling: Simple, stable, democratic
- Attention pooling: More flexible, learnable, but more parameters
- Max pooling: Focuses on strongest signal, but loses information
- CLS token: Clean separation, but requires architectural change

### Visit Size Variation

**Issue:** Visits have different numbers of codes (n_{b,t} varies)

**Impact on mean pooling:**
- Large visits get "diluted" (many codes averaged)
- Small visits have "concentrated" signal (few codes averaged)

**Potential solution:** Weighted aggregation by visit size:
```python
V_t = (Σ H_i) / sqrt(|I_{b,t}|)  # Normalize by sqrt of count
```

---

## Summary

### Key Takeaways

1. **Two different objects:**
   - Visit ID embedding = input lookup table
   - Aggregated visit embedding = output representation

2. **Different stages:**
   - Visit ID: Before transformer
   - Aggregated: After transformer

3. **Different purposes:**
   - Visit ID: Temporal position signal
   - Aggregated: Semantic visit content

4. **scatter_add mechanics:**
   - Vectorized group-by-sum operation
   - Requires careful padding mask handling
   - Differentiable with stable gradients

5. **Optimization:**
   - Visit ID embeddings learned from all tasks
   - Aggregated embeddings computed from contextual modeling
   - Both critical for survival analysis

---

## References

**Code:**
- `src/ehrsequencing/models/embeddings.py` - Visit ID embeddings
- `src/ehrsequencing/models/behrt_survival.py` - Visit aggregation via scatter_add

**Related Documentation:**
- `01_behrt_model_overview.md` - Complete model pipeline
- `01b_ehr_tokens_tensors.md` - Data flattening and masking
- `dev/models/pretrain_finetune/05_embedding_summation_and_quality_analysis.md` - Why sum embeddings

---

**Last Updated:** 2026-02-03  
**Status:** Complete and ready for use
