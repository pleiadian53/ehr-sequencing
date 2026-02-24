# From Hierarchical EHR Data to Flat Token Sequences

**Last Updated:** 2026-02-03  
**Topics:** Data flattening, visit boundaries, attention masking, padding discipline

---

## Overview

This document explains how hierarchical EHR data (Patient → Visits → Codes) is transformed into the flat tensor format required by transformers, and why careful handling of visit boundaries and padding is critical for correct model behavior.

**Key insight:** We flatten the tree structure into a sequence but preserve visit boundaries through `visit_ids` and attention masking. This enables efficient transformer computation while maintaining the hierarchical semantics.

---

## Table of Contents

1. [The True Structure of EHR Data](#1-the-true-structure-of-ehr-data)
2. [Flattening: From Tree to Sequence](#2-flattening-from-tree-to-sequence)
3. [The Dual Role of visit_ids](#3-the-dual-role-of-visit_ids)
4. [Padding: The Silent Killer](#4-padding-the-silent-killer)
5. [Hierarchical vs Flattened Trade-offs](#5-hierarchical-vs-flattened-trade-offs)
6. [What the Model Actually Sees](#6-what-the-model-actually-sees)
7. [Optimization Implications](#7-optimization-implications)

---

## 1. The True Structure of EHR Data

### Natural Hierarchy

EHR data is inherently hierarchical:

```
Patient
├── Visit 1
│   ├── Code A
│   ├── Code B
│   └── Code C
├── Visit 2
│   ├── Code D
│   └── Code E
└── Visit 3
    ├── Code F
    ├── Code G
    ├── Code H
    └── Code I
```

### Mathematical Representation

For patient b:

```
Number of visits: T_b
Visit t contains: n_{b,t} codes
Codes in visit t: {c_{b,t,1}, c_{b,t,2}, ..., c_{b,t,n_{b,t}}}
```

This is a **ragged 3-level tree**:
- Level 1: Patient
- Level 2: Visits (variable count)
- Level 3: Codes per visit (variable count)

### The Problem

Deep learning frameworks (PyTorch, TensorFlow) expect **fixed-size tensors**:

```
Transformer input ∈ ℝ^(B × L)
```

Where:
- B = batch size (fixed)
- L = sequence length (fixed)

**Ragged trees don't fit into fixed tensors** → We must flatten.

---

## 2. Flattening: From Tree to Sequence

### The Flattening Operation

**Before flattening (hierarchical):**

```
Visit 1: [A, B, C]
Visit 2: [D, E]
Visit 3: [F, G, H, I]
```

**After flattening (sequential):**

```
Sequence: [A, B, C, D, E, F, G, H, I]
```

### Mathematical Definition

The flattened sequence for patient b is:

```
sequence_b = [c_{b,1,1}, ..., c_{b,1,n_{b,1}},  # Visit 1
              c_{b,2,1}, ..., c_{b,2,n_{b,2}},  # Visit 2
              ...,
              c_{b,T_b,1}, ..., c_{b,T_b,n_{b,T_b}}]  # Visit T_b
```

Total sequence length:

```
L_b = Σ_{t=1}^{T_b} n_{b,t}
```

### What We've Lost

By flattening, we **destroyed the tree structure**:
- ❌ Don't know where visits start/end
- ❌ Don't know which codes belong together
- ❌ Don't know visit ordering explicitly

### Preserving Structure

To recover visit boundaries, we keep a **parallel tensor**:

```
visit_ids ∈ ℤ^L
```

**Example:**

```
Codes:      [A, B, C, D, E, F, G, H, I]
visit_ids:  [0, 0, 0, 1, 1, 2, 2, 2, 2]
```

This vector is the **lifeline** that preserves visit structure in the flat representation.

---

## 3. The Dual Role of visit_ids

The same tensor `visit_ids` plays **two conceptually different roles** in the pipeline. Understanding this duality is critical.

### Role 1: Input Feature for BEHRT Embeddings

**Purpose:** Inject temporal grouping information into token representations.

**How it's used:**

```python
# In BEHRT embedding layer
visit_emb = self.visit_embedding(visit_ids)  # Lookup in embedding table
x = code_emb + age_emb + visit_emb + pos_emb
```

**What it does:**
- Looks up visit index in learnable embedding matrix E^visit ∈ ℝ^(V_max × d)
- Same embedding vector for all tokens in visit t
- Says: "This token belongs to visit t" (structural prior)

**Key point:** This is a **feature** that gets embedded and processed by the model.

**Semantics:**
> "Temporal grouping metadata injected into token representation"

### Role 2: Aggregation Boundary in Survival Model

**Purpose:** Define which tokens to group when computing visit-level representations.

**How it's used:**

```python
# In survival model
V_{b,t} = mean of H_{b,i} where visit_ids[b,i] == t
```

**What it does:**
- Acts as grouping index for scatter-add operation
- Determines which tokens belong to same visit for aggregation
- Says: "Group these tokens together to form visit representation"

**Key point:** This is an **indexing operation** for aggregation, not an embedding lookup.

**Semantics:**
> "Boundary marker that defines aggregation groups"

### Comparison

| Aspect | Role 1: Input Feature | Role 2: Aggregation Index |
|--------|----------------------|--------------------------|
| **Stage** | Before transformer | After transformer |
| **Operation** | Embedding lookup | Grouping/indexing |
| **Purpose** | Inject structural prior | Define aggregation boundaries |
| **Output** | Embedding vector | Visit representation |
| **Gradients** | Through embedding matrix | Through aggregated embeddings |

### Why This Duality is Elegant

**Same numbers, different semantic roles:**

```
visit_ids = [0, 0, 1, 1, 1, 2, 2]

# Role 1: "Token 0 is from visit 0" (lookup)
# Role 2: "Tokens 0,1 belong to visit 0" (grouping)
```

The same information serves two purposes efficiently!

### Why This Duality is Dangerous

**If misunderstood:**
- May think visit embedding = aggregated embedding (WRONG!)
- May forget to mask padding in aggregation (BUGS!)
- May confuse input signals with output representations (CONFUSION!)

**See also:** `01a_visit_embeddings.md` for detailed explanation of this distinction.

---

## 4. Padding: The Silent Killer

### The Batching Problem

Sequences in a batch have **different lengths**:

```
Patient 1: 10 tokens
Patient 2: 25 tokens
Patient 3: 15 tokens
```

Tensors require **fixed dimensions** → Must pad to maximum length:

```
L_max = max(L_1, L_2, ..., L_B) = 25
```

### Padded Representation

```python
# Patient 1 (10 real tokens, 15 padding)
codes:      [c1, c2, ..., c10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
visit_ids:  [ 0,  0, ...,   2, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?]
            └─── real tokens ──┘ └────────── padding ──────────────────┘
```

### The Danger

**If padding is not handled correctly:**

1. **Transformer contamination:**
   - Transformer attends to padding tokens
   - Padding "information" leaks into representations
   - Contextual embeddings become corrupted

2. **Aggregation errors:**
   - Padding tokens get included in visit sums
   - Visit representations become meaningless
   - Hazard predictions become unstable

3. **Loss computation:**
   - Loss computed on padding positions
   - Gradients from padding corrupt learning
   - Model learns to predict padding (useless!)

**Result:** Model quietly fails without obvious errors.

---

### Solution 1: Attention Mask

**Create a mask indicating real vs padding tokens:**

```python
attention_mask ∈ {0, 1}^(B × L)

Where:
    1 = real token
    0 = padding token
```

**Example:**

```python
codes:          [c1, c2, c3, c4,  0,  0,  0,  0]
attention_mask: [ 1,  1,  1,  1,  0,  0,  0,  0]
```

**Usage in transformer:**

```python
# PyTorch transformer expects boolean mask
# True = positions to IGNORE
src_key_padding_mask = ~attention_mask.bool()

output = self.transformer(
    embeddings,
    src_key_padding_mask=src_key_padding_mask
)
```

**Effect:**

```
Attention(i, j) = 0  if token j is padding
```

Padding tokens are **completely excluded** from attention computation.

---

### Solution 2: Aggregation Mask Discipline

**When aggregating to visits, must filter padding:**

**Correct aggregation:**

```
V_{b,t} = (Σ_{i=1}^L 𝟙(visit_ids[b,i]=t) · 𝟙(mask[b,i]=1) · H_{b,i})
          / (Σ_{i=1}^L 𝟙(visit_ids[b,i]=t) · 𝟙(mask[b,i]=1))
```

Both conditions must be enforced:
1. Token belongs to visit t
2. Token is not padding

**Implementation:**

```python
def aggregate_to_visits(hidden_states, visit_ids, attention_mask):
    """
    Aggregate token embeddings to visit embeddings.
    
    Correctly handles padding by masking before aggregation.
    """
    B, L, d = hidden_states.shape
    T_max = visit_ids.max() + 1
    
    # Initialize outputs
    visit_sums = torch.zeros(B, T_max, d, device=hidden_states.device)
    visit_counts = torch.zeros(B, T_max, device=hidden_states.device)
    
    # CRITICAL: Mask out padding
    valid_mask = attention_mask.bool()  # [B, L]
    
    # Only aggregate valid (non-padding) tokens
    masked_hidden = hidden_states * valid_mask.unsqueeze(-1)
    
    # Scatter-add to group by visit
    visit_sums.scatter_add_(
        dim=1,
        index=visit_ids.unsqueeze(-1).expand(-1, -1, d),
        src=masked_hidden
    )
    
    # Count valid tokens per visit
    visit_counts.scatter_add_(
        dim=1,
        index=visit_ids,
        src=valid_mask.float()
    )
    
    # Mean pooling (safe division)
    visit_embeddings = visit_sums / (visit_counts.unsqueeze(-1) + 1e-8)
    
    return visit_embeddings
```

**Key points:**
- ✅ Mask hidden states BEFORE scatter-add
- ✅ Count only valid tokens for mean
- ✅ Safe division with epsilon to avoid NaN

---

### Solution 3: Loss Masking

**For token-level losses (MLM), ignore padding positions:**

```python
# CrossEntropy automatically ignores index=-100
loss = nn.CrossEntropyLoss(ignore_index=-100)

# Set padding positions to -100
labels[attention_mask == 0] = -100

# Compute loss (padding ignored)
mlm_loss = loss(logits.view(-1, vocab_size), labels.view(-1))
```

**For sequence/visit-level losses, mask is handled in aggregation step.**

---

### Common Padding Bugs

**Bug 1: Forgetting attention mask**
```python
# WRONG: No masking
output = model(codes, ages, visit_ids)  # ❌

# CORRECT: With attention mask
mask = (codes != 0).long()
output = model(codes, ages, visit_ids, mask)  # ✅
```

**Bug 2: Padding values in visit_ids**
```python
# WRONG: visit_ids has undefined values for padding
codes:      [1, 2, 3, 0, 0, 0]
visit_ids:  [0, 0, 1, ?, ?, ?]  # ❌ What should ? be?

# CORRECT: Set padding visit_ids to 0 (will be masked anyway)
visit_ids:  [0, 0, 1, 0, 0, 0]  # ✅ Consistent, masked out later
```

**Bug 3: Including padding in visit counts**
```python
# WRONG: Count all tokens
count = (visit_ids == t).sum()  # ❌ Includes padding!

# CORRECT: Count only valid tokens
count = ((visit_ids == t) & (mask == 1)).sum()  # ✅
```

---

## 5. Hierarchical vs Flattened Trade-offs

### Design Decision

**Chosen approach:** Flatten everything + use visit_ids to preserve structure

**Alternative:** Hierarchical transformer (visit encoder → patient encoder)

### Flattening Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Single transformer** | Simpler architecture, fewer components |
| **Full cross-visit attention** | Tokens can attend across visit boundaries |
| **Simpler batching** | Standard tensor operations |
| **BEHRT compatibility** | Can use pre-trained BEHRT directly |
| **Computational efficiency** | One forward pass instead of nested |

### Flattening Disadvantages

| Disadvantage | Explanation |
|--------------|-------------|
| **Mask discipline** | Must carefully handle padding everywhere |
| **Visit boundaries implicit** | Structure only preserved through visit_ids |
| **Sequence length limits** | Max sequence length = max tokens (not max visits) |
| **Memory for long histories** | O(total_tokens) vs O(visits) * O(codes_per_visit) |

### Hierarchical Alternative

**Architecture:**
```
Visit Encoder: codes → visit representation
Patient Encoder: visit sequence → patient representation
```

**Pros:**
- ✅ Explicit visit boundaries
- ✅ Can use visit-level features naturally
- ✅ Scales better with many visits

**Cons:**
- ❌ More complex architecture
- ❌ No pre-trained models available
- ❌ Limited cross-visit attention
- ❌ Slower training (nested loops)

### Why Flattening is Pragmatic

For BEHRT-based models, flattening is the **pragmatic choice**:
1. Leverage BEHRT pre-training
2. Simpler implementation
3. Full attention across history
4. Better code reuse

**Trade-off:** Requires rigorous masking discipline (but worth it!).

---

## 6. What the Model Actually Sees

### The Transformer's View

After flattening, **the transformer sees one long sequence**:

```
[c_{b,1,1}, c_{b,1,2}, ..., c_{b,T,n_T}]
```

**It does NOT inherently know:**
- ❌ Where visits start or end
- ❌ How many codes in each visit
- ❌ Visit temporal spacing
- ❌ Visit boundaries

### How Structure is Recovered

**The model learns visit structure through embeddings:**

1. **Visit embeddings** (E^visit[visit_ids])
   - Same vector for tokens in same visit
   - Different vectors for different visits
   - Learned grouping signal

2. **Positional embeddings** (E^pos[position])
   - Different for each token position
   - Provides fine-grained ordering

3. **Age embeddings** (E^age[age_bin])
   - Changes across visits (patient ages)
   - Temporal progression signal

**Together, these embeddings encode:**
- Visit boundaries (via visit embeddings)
- Token ordering (via positional embeddings)
- Temporal progression (via age embeddings)

### Geometric Intuition

**Without embeddings:**
```
All tokens look the same → Model can't distinguish visits
```

**With embeddings:**
```
Token embedding = code + age + visit + position

Tokens in same visit:
    - Share visit embedding (grouping)
    - Have different position embeddings (ordering)
    
Tokens in different visits:
    - Have different visit embeddings (separation)
    - Have different age embeddings (time progression)
```

**Result:** Inductive bias about hierarchical structure is **encoded in the embedding space**.

---

## 7. Optimization Implications

### If Structure is Not Preserved

**Without proper visit encoding:**

```
Model treats:
    [A, B, C] from visit 1
    
The same as:
    [A], [B], [C] from visits 1, 2, 3
```

**Consequences:**
- Model confuses codes within visit vs across visits
- Survival aggregation smears information incorrectly
- Hazard estimates become unstable and meaningless

### Why the Pipeline Works

**The pipeline preserves structure through:**

1. **Flattening**
   - Converts tree to sequence

2. **Visit embeddings**
   - Inject visit grouping as learned prior

3. **Attention masking**
   - Prevents padding contamination

4. **Correct aggregation**
   - Recovers visit-level representations using visit_ids

**Together, these restore hierarchical structure in differentiable form.**

### Gradient Flow

**Structure preservation is critical for gradients:**

```
Loss → Hazard → Visit embedding → scatter_add → Token embeddings → Transformer
```

If visit boundaries are wrong:
- Gradients flow to wrong tokens
- Model learns spurious patterns
- Optimization diverges or gets stuck

**With correct structure:**
- Gradients flow to correct token groups
- Model learns visit-level patterns
- Optimization converges smoothly

---

## 8. Mental Model

### The Forest Analogy

**Think of EHR data as:**

```
Raw EHR = Forest of trees
    Each patient = Tree
    Each visit = Branch
    Each code = Leaf

Flattening = Cut forest into long vine
    Lay all leaves in sequence
    Lose branch structure

visit_ids = Colored tape marking branches
    Each color = Original branch
    Preserves grouping information

scatter_add = Regrouping leaves into branches
    Group by tape color
    Reconstruct branch-level information
```

**It works — but only if you keep the colored tape intact!**

### The Street Analogy

**From `01a_visit_embeddings.md`:**

```
Visit ID embedding = Street number
    "This is building #3"
    
Aggregated visit embedding = What happened inside
    "Meeting, presentation, coffee, discussion"
```

Flattening destroys buildings, but visit_ids preserve addresses so we can rebuild.

---

## 9. Implementation Best Practices

### Checklist for Correct Implementation

- [ ] **Always create attention mask:**
  ```python
  attention_mask = (codes != pad_token_id).long()
  ```

- [ ] **Pass mask to transformer:**
  ```python
  output = model(codes, ages, visit_ids, attention_mask)
  ```

- [ ] **Mask before aggregation:**
  ```python
  masked_hidden = hidden * attention_mask.unsqueeze(-1)
  ```

- [ ] **Count only valid tokens:**
  ```python
  counts = attention_mask.float().scatter_add(...)
  ```

- [ ] **Safe division:**
  ```python
  mean = sum / (count + 1e-8)  # Avoid division by zero
  ```

- [ ] **Validate visit_ids:**
  ```python
  assert (visit_ids[attention_mask==0] == 0).all()  # Padding has visit_id=0
  ```

### Testing Mask Correctness

**Test 1: Padding should not affect outputs**

```python
# Create sequence with and without padding
seq1 = [1, 2, 3]
seq2 = [1, 2, 3, 0, 0, 0]  # Same but padded

output1 = model(seq1)
output2 = model(seq2)

# Outputs should be identical for non-padding positions
assert torch.allclose(output1, output2[:len(seq1)])
```

**Test 2: Aggregation should ignore padding**

```python
# Visit with padding
codes = torch.tensor([[1, 2, 3, 0, 0]])
visit_ids = torch.tensor([[0, 0, 1, 0, 0]])
mask = torch.tensor([[1, 1, 1, 0, 0]])

# Aggregate
visit_emb = aggregate_to_visits(hidden, visit_ids, mask)

# Visit 0 should only include first 2 tokens
# Visit 1 should only include token 2
# Padding tokens (3, 4) should not affect results
```

---

## 10. Advanced Topics

### Long Sequence Handling

**Problem:** Patients with many visits → very long sequences

**Solutions:**

1. **Truncation:**
   ```python
   # Keep most recent L_max tokens
   codes = codes[:, -L_max:]
   visit_ids = visit_ids[:, -L_max:]
   ```

2. **Visit-level sampling:**
   ```python
   # Sample K most important visits
   selected_visits = select_important_visits(visits, K)
   ```

3. **Hierarchical modeling:**
   ```python
   # Use visit encoder → patient encoder
   # (But requires new architecture)
   ```

### Efficient Batching

**Problem:** Variable-length sequences waste computation on padding

**Solution:** Pack sequences to minimize padding

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Sort by length (descending)
lengths = (attention_mask.sum(dim=1)).cpu()
sorted_lengths, indices = lengths.sort(descending=True)

# Pack (removes padding)
packed = pack_padded_sequence(embeddings[indices], sorted_lengths, batch_first=True)

# Process
packed_output = model(packed)

# Unpack
output, _ = pad_packed_sequence(packed_output, batch_first=True)

# Restore original order
output = output[indices.argsort()]
```

---

## Summary

### Key Takeaways

1. **Hierarchical → Flat transformation:**
   - EHR data is naturally hierarchical (patient → visits → codes)
   - Transformers require flat sequences
   - Flattening loses structure unless carefully preserved

2. **visit_ids dual role:**
   - Input feature: Embedding lookup for temporal grouping
   - Aggregation index: Boundary marker for grouping tokens

3. **Padding is dangerous:**
   - Must mask in attention
   - Must mask in aggregation
   - Must mask in loss
   - Silent bugs if missed!

4. **Structure preservation:**
   - Visit embeddings encode grouping
   - Attention masking prevents contamination
   - scatter_add recovers visit-level representations

5. **Design trade-off:**
   - Flattening is pragmatic for BEHRT
   - Requires rigorous masking discipline
   - Enables pre-training and full attention

### Mental Checklist

**Every time you process EHR sequences:**

1. ✅ Create attention mask
2. ✅ Pass mask to transformer
3. ✅ Mask before aggregation
4. ✅ Count only valid tokens
5. ✅ Test padding correctness

**If you follow this discipline, flattening works beautifully!**

---

## References

**Code:**
- `src/ehrsequencing/models/embeddings.py` - Embedding layer with visit IDs
- `src/ehrsequencing/models/behrt.py` - Transformer with attention masking
- `src/ehrsequencing/models/behrt_survival.py` - Visit aggregation implementation

**Related Documentation:**
- `01_behrt_model_overview.md` - Complete model pipeline
- `01a_visit_embeddings.md` - Visit ID vs aggregated embeddings
- `dev/models/pretrain_finetune/01_behrt_model_design.md` - BEHRT architecture details

---

**Last Updated:** 2026-02-03  
**Status:** Complete and ready for use
