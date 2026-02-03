# TemporalMed2Vec Q&A: Deep Dive into Implementation Details

**Date:** January 20, 2026  
**Purpose:** Follow-up questions and practical considerations for TemporalMed2Vec implementation

---

## Question 1: Vocab Size in Medical Code Domain

### What is `vocab_size`?

In the context of medical code embeddings, `vocab_size` represents the **total number of unique medical codes** in your vocabulary that need embeddings.

```python
model = TemporalMed2Vec(
    vocab_size=10000,  # Number of unique codes to embed
    embed_dim=128,
    time_decay=0.01
)
```

### Practical Considerations

#### 1. Standard Medical Code System Sizes

Different medical coding systems have vastly different vocabulary sizes:

| Code System | Approximate Size | Use Case |
|-------------|-----------------|----------|
| **ICD-10-CM** | ~70,000 codes | Diagnosis codes (US) |
| **ICD-9-CM** | ~14,000 codes | Legacy diagnosis codes |
| **LOINC** | ~90,000 codes | Lab tests and observations |
| **SNOMED CT** | ~350,000 concepts | Clinical terminology |
| **RxNorm** | ~200,000 concepts | Medications |
| **CPT** | ~10,000 codes | Procedures |
| **Custom/Hospital** | Variable | Institution-specific codes |

**Reality Check:** In practice, you rarely use the full code system.

---

#### 2. Real-World Dataset Sizes

Most EHR datasets contain a **subset** of these codes:

```python
# Example: Analyzing your dataset
import pandas as pd
from collections import Counter

# Load patient sequences
patient_data = load_ehr_sequences()  # List of [(timestamp, code), ...]

# Count unique codes
all_codes = []
for patient in patient_data:
    all_codes.extend([code for timestamp, code in patient])

code_counts = Counter(all_codes)
n_unique_codes = len(code_counts)

print(f"Unique codes in dataset: {n_unique_codes}")
print(f"Most common codes: {code_counts.most_common(10)}")
```

**Typical Results:**
- Small hospital (1 year): 5,000-15,000 unique codes
- Large hospital (5 years): 20,000-50,000 unique codes
- Multi-site EHR (10 years): 50,000-100,000 unique codes

**Your vocab_size should match your dataset:**
```python
vocab_size = len(code_counts)  # Exact number of unique codes
```

---

#### 3. Memory Considerations

Each embedding matrix requires significant memory:

```python
# Memory calculation
vocab_size = 50000
embed_dim = 128

# Two embedding matrices: code_embeddings + context_embeddings
memory_per_matrix = vocab_size * embed_dim * 4  # 4 bytes per float32
total_memory = 2 * memory_per_matrix

print(f"Memory per matrix: {memory_per_matrix / 1e6:.1f} MB")
print(f"Total embedding memory: {total_memory / 1e6:.1f} MB")
```

**Example Results:**
- vocab_size=10,000, embed_dim=128: **~10 MB** total
- vocab_size=50,000, embed_dim=128: **~51 MB** total
- vocab_size=100,000, embed_dim=256: **~205 MB** total
- vocab_size=350,000 (full SNOMED), embed_dim=128: **~358 MB** total

**GPU Considerations:**
- Modern GPUs (16GB+): Can handle vocab_size up to 500,000
- Older GPUs (4-8GB): May struggle with vocab_size > 100,000
- Consider batch size impact: embeddings + batch activations

---

#### 4. Handling Rare Codes

**Problem:** Medical datasets follow a **long-tail distribution**:
- Top 1,000 codes: 80% of occurrences
- Bottom 30,000 codes: 1% of occurrences (many appear only once)

**Why This Matters:**
- Rare codes have insufficient training signal
- Embedding quality is poor for codes seen < 10 times
- Memory wasted on rarely-used embeddings

**Solution 1: Frequency Thresholding**

```python
def filter_rare_codes(code_counts, min_frequency=10):
    """
    Keep only codes that appear at least min_frequency times.
    """
    frequent_codes = {
        code: count 
        for code, count in code_counts.items() 
        if count >= min_frequency
    }
    
    print(f"Original vocab size: {len(code_counts)}")
    print(f"Filtered vocab size: {len(frequent_codes)}")
    
    # Create vocabulary mapping
    code_to_idx = {code: idx for idx, code in enumerate(frequent_codes.keys())}
    code_to_idx['<UNK>'] = len(code_to_idx)  # Unknown token for rare codes
    
    return code_to_idx

# Example
code_counts = Counter(all_codes)
code_to_idx = filter_rare_codes(code_counts, min_frequency=10)
vocab_size = len(code_to_idx)  # Reduced size

model = TemporalMed2Vec(vocab_size=vocab_size, embed_dim=128, time_decay=0.01)
```

**Typical Reduction:**
- Original: 50,000 unique codes
- After filtering (min_freq=10): 8,000 codes (**84% reduction**)
- Covers: 95%+ of all code occurrences

**Solution 2: Hierarchical Grouping**

Group rare specific codes to their parent category:

```python
def group_icd10_codes(code, level='3digit'):
    """
    ICD-10 Example: E11.65 → E11 (3-digit) or E11.6 (4-digit)
    """
    if level == '3digit':
        return code[:3]  # E11.65 → E11
    elif level == '4digit':
        return code[:5]  # E11.65 → E11.6
    return code

# Before grouping: E11.00, E11.01, E11.02, ... (100+ specific diabetes codes)
# After grouping: E11 (single diabetes category)
```

---

#### 5. Multi-System Vocabularies

**Challenge:** Real EHR data contains **mixed code systems**:

```python
Patient Timeline:
  2020-01-15 | ICD10:E11.9
  2020-01-15 | LOINC:4548-4
  2020-01-15 | RXNORM:860975
```

**Approach 1: Unified Vocabulary (Recommended)**

```python
# Create unified code space with prefixes
all_codes_with_prefix = []

for patient in patient_data:
    for timestamp, code in patient:
        # Add system prefix
        if code.startswith('E') or code.startswith('I'):
            unified_code = f"ICD10:{code}"
        elif is_loinc(code):
            unified_code = f"LOINC:{code}"
        elif is_rxnorm(code):
            unified_code = f"RXNORM:{code}"
        
        all_codes_with_prefix.append(unified_code)

code_to_idx = {code: idx for idx, code in enumerate(set(all_codes_with_prefix))}
vocab_size = len(code_to_idx)
```

**Result:** vocab_size = unique ICD codes + unique LOINC codes + unique RxNorm codes

**Approach 2: Separate Embeddings per System**

```python
class MultiSystemMed2Vec(nn.Module):
    def __init__(self, vocab_sizes_dict, embed_dim=128):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            'icd10': nn.Embedding(vocab_sizes_dict['icd10'], embed_dim),
            'loinc': nn.Embedding(vocab_sizes_dict['loinc'], embed_dim),
            'rxnorm': nn.Embedding(vocab_sizes_dict['rxnorm'], embed_dim),
        })
    
    def get_embedding(self, code, system):
        return self.embeddings[system](code)

# Usage
vocab_sizes = {'icd10': 15000, 'loinc': 5000, 'rxnorm': 8000}
model = MultiSystemMed2Vec(vocab_sizes)
```

---

#### 6. Dynamic Vocabularies

**Problem:** New codes added over time (e.g., COVID-19 codes in 2020)

**Solution: Out-of-Vocabulary (OOV) Handling**

```python
class RobustTemporalMed2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, time_decay=0.1):
        super().__init__()
        # Add +1 for <UNK> token
        self.code_embeddings = nn.Embedding(vocab_size + 1, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size + 1, embed_dim)
        self.time_decay = time_decay
        self.unk_idx = vocab_size  # Last index reserved for unknown
    
    def forward(self, target_code, context_codes, time_deltas):
        # Clamp out-of-vocab codes to <UNK>
        target_code = torch.where(target_code >= self.unk_idx, 
                                  self.unk_idx, 
                                  target_code)
        context_codes = torch.where(context_codes >= self.unk_idx, 
                                    self.unk_idx, 
                                    context_codes)
        
        # Rest of forward pass...
```

---

### Practical Recommendations

#### Starting Point

```python
# Step 1: Analyze your dataset
code_counts = analyze_code_frequency(your_data)
print(f"Total unique codes: {len(code_counts)}")

# Step 2: Filter rare codes
code_to_idx = filter_rare_codes(code_counts, min_frequency=10)
vocab_size = len(code_to_idx)
print(f"Filtered vocab size: {vocab_size}")

# Step 3: Choose embed_dim based on vocab_size
if vocab_size < 1000:
    embed_dim = 64
elif vocab_size < 10000:
    embed_dim = 128
else:
    embed_dim = 256

# Step 4: Initialize model
model = TemporalMed2Vec(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    time_decay=0.01
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

#### Rule of Thumb

- **Small dataset** (< 10k patients): vocab_size ≈ 5,000-10,000
- **Medium dataset** (10k-100k patients): vocab_size ≈ 10,000-30,000
- **Large dataset** (> 100k patients): vocab_size ≈ 30,000-100,000

**Filter aggressively:** Better to have high-quality embeddings for common codes than poor embeddings for all codes.

---

## Question 2: Mastering Einsum Expressions

### What is `einsum`?

`torch.einsum` (Einstein summation) is a **compact notation** for expressing tensor operations using index notation.

```python
scores = torch.einsum('be,bce->bc', target_embed, context_embed)
```

### Basic Einsum Syntax

**Format:** `einsum('input_indices,input_indices->output_indices', tensor1, tensor2)`

**Rules:**
1. Each letter represents a **dimension**
2. Repeated letters = **sum over that dimension** (reduction)
3. Letters in output = **keep that dimension**
4. Letters not in output = **summed out**

---

### Tutorial: From Simple to Complex

#### Level 1: Vector Operations

**Example 1: Dot Product**

```python
# Traditional
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])
result = (a * b).sum()  # 1*4 + 2*5 + 3*6 = 32

# Einsum
result = torch.einsum('i,i->', a, b)
#                     'i,i->'
#                      │ │  │
#                      │ │  └─ output: scalar (no indices)
#                      │ └──── b has dimension i
#                      └─────── a has dimension i
#                     Repeated 'i' → sum over i
```

**Example 2: Element-wise Product (No Sum)**

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

result = torch.einsum('i,i->i', a, b)  # [4., 10., 18.]
#                     'i,i->i'
#                          └─ Keep dimension i in output (no sum)
```

---

#### Level 2: Matrix Operations

**Example 3: Matrix-Vector Multiplication**

```python
# Matrix: [3, 4], Vector: [4]
A = torch.randn(3, 4)
v = torch.randn(4)

# Traditional
result = A @ v  # Shape: [3]

# Einsum
result = torch.einsum('ij,j->i', A, v)
#                     'ij,j->i'
#                      ││ │  │
#                      ││ │  └─ output has dimension i (3)
#                      ││ └──── v has dimension j (4)
#                      │└─────── A has dimensions i,j (3,4)
#                      └──────── j appears in both → sum over j
```

**Step-by-step:**
```
A[i,j] * v[j] → sum over j → result[i]

result[0] = A[0,0]*v[0] + A[0,1]*v[1] + A[0,2]*v[2] + A[0,3]*v[3]
result[1] = A[1,0]*v[0] + A[1,1]*v[1] + A[1,2]*v[2] + A[1,3]*v[3]
result[2] = A[2,0]*v[0] + A[2,1]*v[1] + A[2,2]*v[2] + A[2,3]*v[3]
```

**Example 4: Matrix Multiplication**

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)

# Traditional
result = A @ B  # Shape: [3, 5]

# Einsum
result = torch.einsum('ik,kj->ij', A, B)
#                     'ik,kj->ij'
#                      ││ ││  ││
#                      ││ ││  └┴─ output: [i, j] = [3, 5]
#                      ││ └┴──── B: [k, j] = [4, 5]
#                      └┴──────── A: [i, k] = [3, 4]
#                     k repeated → sum over k (dimension 4)
```

---

#### Level 3: Batch Operations

**Example 5: Batched Dot Products**

```python
# Batch of vectors: [batch_size, vector_dim]
a = torch.randn(32, 128)  # 32 vectors of dim 128
b = torch.randn(32, 128)

# Compute dot product for each pair (a[i] · b[i])
result = torch.einsum('bi,bi->b', a, b)  # Shape: [32]
#                     'bi,bi->b'
#                      ││ ││  │
#                      ││ ││  └─ output: [b] = [32] (one score per batch)
#                      ││ └┴──── b: [batch, i]
#                      └┴──────── a: [batch, i]
#                     i repeated but NOT in output → sum over i
#                     b in output → keep batch dimension
```

**Example 6: Batch Matrix-Vector**

```python
# Batch of matrices and vectors
A = torch.randn(32, 3, 4)  # 32 matrices of shape [3, 4]
v = torch.randn(32, 4)     # 32 vectors of shape [4]

# Compute A[i] @ v[i] for each batch element
result = torch.einsum('bij,bj->bi', A, v)  # Shape: [32, 3]
#                     'bij,bj->bi'
#                      │││ ││  ││
#                      │││ ││  └┴─ output: [batch, i] = [32, 3]
#                      │││ └┴──── v: [batch, j] = [32, 4]
#                      └┴┴──────── A: [batch, i, j] = [32, 3, 4]
#                     j repeated → sum over j
```

---

#### Level 4: Medical Code Example

**The TemporalMed2Vec Line Explained**

```python
target_embed = torch.randn(32, 128)      # [batch_size, embed_dim]
context_embed = torch.randn(32, 10, 128) # [batch_size, context_size, embed_dim]

scores = torch.einsum('be,bce->bc', target_embed, context_embed)
#                     'be,bce->bc'
#                      ││ │││  ││
#                      ││ │││  └┴─ output: [batch, context] = [32, 10]
#                      ││ ││└───── context_embed: [batch, context, embed]
#                      └┴─└└───── target_embed: [batch, embed]
#                     e repeated → sum over embedding dimension
#                     b,c kept → output has batch and context dimensions
```

**What it computes:**
```python
scores[batch_i, context_j] = Σ target_embed[batch_i, k] * context_embed[batch_i, context_j, k]
                              k=0 to embed_dim

# This is the dot product between:
# - Target embedding for patient i
# - Context embedding j for patient i
```

**Equivalent Traditional Code:**

```python
# Method 1: Explicit loop (slow)
scores = torch.zeros(32, 10)
for b in range(32):
    for c in range(10):
        scores[b, c] = (target_embed[b] * context_embed[b, c]).sum()

# Method 2: Broadcasting (fast but verbose)
scores = (target_embed.unsqueeze(1) * context_embed).sum(dim=-1)
#         [32, 1, 128]            *  [32, 10, 128] → [32, 10, 128]
#                                                     sum(dim=-1) → [32, 10]

# Method 3: Batched matrix multiply
scores = torch.bmm(target_embed.unsqueeze(1), context_embed.transpose(1, 2)).squeeze(1)
#                  [32, 1, 128]              @ [32, 128, 10] → [32, 1, 10] → [32, 10]

# Method 4: Einsum (cleanest)
scores = torch.einsum('be,bce->bc', target_embed, context_embed)
```

**Why einsum wins:** Clearest intent, compiler-optimized, no shape gymnastics.

---

### Practice Exercises

#### Exercise 1: Translate to Einsum

Given these operations, write the einsum equivalent:

```python
# a) Sum all elements of a matrix
A = torch.randn(3, 4)
result = A.sum()
# Answer: torch.einsum('ij->', A)

# b) Sum along rows (result shape: [4])
result = A.sum(dim=0)
# Answer: torch.einsum('ij->j', A)

# c) Sum along columns (result shape: [3])
result = A.sum(dim=1)
# Answer: torch.einsum('ij->i', A)
```

#### Exercise 2: Batch Attention Scores

```python
# Query: [batch=32, seq_len=20, d_model=64]
# Key: [batch=32, seq_len=20, d_model=64]
# Want: Attention scores [batch=32, seq_len=20, seq_len=20]

query = torch.randn(32, 20, 64)
key = torch.randn(32, 20, 64)

# Compute: scores[b, i, j] = query[b, i, :] · key[b, j, :]
scores = torch.einsum('bid,bjd->bij', query, key)
#                     'bid,bjd->bij'
#                      │││ │││  │││
#                      │││ │││  └┴┴─ output: [b, i, j]
#                      │││ └┴┴───── key: [batch, j, d_model]
#                      └┴┴──────── query: [batch, i, d_model]
#                     d repeated → dot product over d_model
```

#### Exercise 3: Medical Code Triplet Similarity

```python
# Anchor codes: [batch=64, embed=128]
# Positive codes: [batch=64, embed=128]
# Negative codes: [batch=64, num_negatives=10, embed=128]

anchor = torch.randn(64, 128)
positive = torch.randn(64, 128)
negative = torch.randn(64, 10, 128)

# Compute positive similarity: [64]
pos_sim = torch.einsum('be,be->b', anchor, positive)

# Compute negative similarities: [64, 10]
neg_sim = torch.einsum('be,bne->bn', anchor, negative)
#                      'be,bne->bn'
#                       ││ │││  ││
#                       ││ │││  └┴─ output: [batch, num_neg]
#                       ││ └┴┴───── negative: [batch, num_neg, embed]
#                       └┴──────── anchor: [batch, embed]
```

---

### Einsum Cheat Sheet

| Operation | Traditional | Einsum |
|-----------|-------------|--------|
| Dot product | `(a*b).sum()` | `'i,i->'` |
| Matrix-vector | `A @ v` | `'ij,j->i'` |
| Matrix-matrix | `A @ B` | `'ik,kj->ij'` |
| Outer product | `a.unsqueeze(1) * b.unsqueeze(0)` | `'i,j->ij'` |
| Batch dot product | `(a*b).sum(-1)` | `'bi,bi->b'` |
| Batch matmul | `torch.bmm(A, B)` | `'bik,bkj->bij'` |
| Transpose | `A.T` | `'ij->ji'` |
| Diagonal | `torch.diag(A)` | `'ii->i'` |
| Trace | `torch.trace(A)` | `'ii->'` |

---

## Question 3: Do Models Always Return Loss?

### Short Answer: **No**

Models typically return **different outputs** depending on the context (training vs. inference).

---

### Training Mode: Return Loss

During training, it's common (but not required) for models to compute and return loss:

```python
class TemporalMed2Vec(nn.Module):
    def forward(self, target_code, context_codes, time_deltas):
        # ... compute embeddings and scores ...
        loss = -(log_probs * temporal_weights).sum() / temporal_weights.sum()
        return loss

# Training
for epoch in range(50):
    for target, context, deltas in dataloader:
        loss = model(target, context, deltas)  # Returns loss directly
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Why this design?**
- Convenient: Loss computation inside model keeps related code together
- Encapsulation: Loss details (temporal weighting) hidden from training loop
- Common in research code and tutorials

**Downside:**
- Couples model architecture with loss function
- Harder to reuse model with different losses
- Inference requires different code path

---

### Inference Mode: Return Embeddings/Predictions

For inference (getting embeddings, predictions), models return **useful outputs**, not loss:

```python
class TemporalMed2Vec(nn.Module):
    def forward(self, target_code, context_codes, time_deltas, return_loss=True):
        target_embed = self.code_embeddings(target_code)
        context_embed = self.context_embeddings(context_codes)
        scores = torch.einsum('be,bce->bc', target_embed, context_embed)
        temporal_weights = torch.exp(-self.time_decay * time_deltas)
        
        if return_loss:
            log_probs = torch.log_softmax(scores, dim=-1)
            loss = -(log_probs * temporal_weights).sum() / temporal_weights.sum()
            return loss
        else:
            return scores  # or return target_embed for embeddings

# Inference: Get embeddings
model.eval()
with torch.no_grad():
    embeddings = model.code_embeddings.weight  # All code embeddings
    
    # Or get specific code embedding
    code_idx = torch.tensor([42])
    specific_embed = model.code_embeddings(code_idx)
```

---

### Best Practice: Separate Loss Function

**Production Pattern:** Keep model and loss separate

```python
class TemporalMed2Vec(nn.Module):
    """Model only computes representations, no loss."""
    
    def __init__(self, vocab_size, embed_dim=128, time_decay=0.1):
        super().__init__()
        self.code_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.time_decay = time_decay
    
    def forward(self, target_code, context_codes, time_deltas):
        """
        Returns:
            scores: [batch, context] similarity scores
            temporal_weights: [batch, context] time-based weights
        """
        target_embed = self.code_embeddings(target_code)
        context_embed = self.context_embeddings(context_codes)
        scores = torch.einsum('be,bce->bc', target_embed, context_embed)
        temporal_weights = torch.exp(-self.time_decay * time_deltas)
        
        return scores, temporal_weights
    
    def get_embeddings(self, code_indices):
        """Get embeddings for specific codes."""
        return self.code_embeddings(code_indices)


class TemporalLoss(nn.Module):
    """Separate loss function."""
    
    def forward(self, scores, temporal_weights):
        log_probs = torch.log_softmax(scores, dim=-1)
        loss = -(log_probs * temporal_weights).sum() / temporal_weights.sum()
        return loss


# Training with separation
model = TemporalMed2Vec(vocab_size=10000, embed_dim=128, time_decay=0.01)
loss_fn = TemporalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    for target, context, deltas in dataloader:
        # Model computes representations
        scores, weights = model(target, context, deltas)
        
        # Loss function computes loss
        loss = loss_fn(scores, weights)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Inference: No need to worry about loss
model.eval()
with torch.no_grad():
    # Get embeddings
    code_indices = torch.tensor([0, 42, 100])
    embeddings = model.get_embeddings(code_indices)
    
    # Compute similarity between two codes
    code_a_embed = model.get_embeddings(torch.tensor([42]))
    code_b_embed = model.get_embeddings(torch.tensor([100]))
    similarity = torch.cosine_similarity(code_a_embed, code_b_embed, dim=-1)
```

---

### Common Patterns in PyTorch

#### Pattern 1: Training-Only Loss (Original Code)

```python
class Model(nn.Module):
    def forward(self, x, y):
        predictions = self.network(x)
        loss = criterion(predictions, y)
        return loss  # Only useful for training
```

**Pro:** Simple training loop  
**Con:** Can't use model for inference without modifying forward()

---

#### Pattern 2: Conditional Loss

```python
class Model(nn.Module):
    def forward(self, x, y=None):
        predictions = self.network(x)
        
        if y is not None:  # Training mode
            loss = criterion(predictions, y)
            return loss, predictions
        else:  # Inference mode
            return predictions
```

**Pro:** Single forward() for training and inference  
**Con:** Awkward API, conditional logic in forward()

---

#### Pattern 3: Separate Methods (Recommended)

```python
class Model(nn.Module):
    def forward(self, x):
        """Always returns predictions."""
        return self.network(x)
    
    def compute_loss(self, predictions, targets):
        """Separate method for loss."""
        return criterion(predictions, targets)

# Training
predictions = model(x)
loss = model.compute_loss(predictions, y)

# Inference
predictions = model(x)
```

**Pro:** Clear separation, flexible  
**Con:** Slightly more verbose

---

#### Pattern 4: External Loss (Best Practice)

```python
class Model(nn.Module):
    def forward(self, x):
        return self.network(x)

# Loss function is separate module/function
criterion = nn.CrossEntropyLoss()

# Training
predictions = model(x)
loss = criterion(predictions, y)

# Inference
predictions = model(x)
```

**Pro:** Maximum flexibility, reusable components  
**Con:** None (this is standard PyTorch style)

---

### Real-World Examples

#### Hugging Face Transformers

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Training: Pass labels, get loss
outputs = model(input_ids, labels=labels)
loss = outputs.loss  # Loss is computed inside

# Inference: No labels, get logits
outputs = model(input_ids)
predictions = outputs.logits  # No loss computed
```

---

#### PyTorch Vision Models

```python
import torchvision.models as models

resnet = models.resnet50(pretrained=True)

# Always returns features, never loss
features = resnet(images)

# Loss computed externally
criterion = nn.CrossEntropyLoss()
loss = criterion(features, labels)
```

---

### When to Return Loss vs. Outputs

| Use Case | Return | Reasoning |
|----------|--------|-----------|
| Research/prototyping | Loss | Convenience, rapid iteration |
| Production models | Outputs | Flexibility, reusability |
| Library/framework | Outputs | Users define their own losses |
| Simple task | Loss | Keep code minimal |
| Multiple tasks | Outputs | Different losses for different tasks |
| Pre-training + fine-tuning | Outputs | Different losses per stage |

---

### Refactoring TemporalMed2Vec for Production

Here's how I'd refactor the original code:

```python
class TemporalMed2Vec(nn.Module):
    """
    Temporal medical code embeddings with clean separation of concerns.
    """
    def __init__(self, vocab_size, embed_dim=128, time_decay=0.1):
        super().__init__()
        self.code_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.time_decay = time_decay
    
    def forward(self, target_code, context_codes, time_deltas):
        """
        Compute similarity scores between target and context codes.
        
        Returns:
            dict with keys:
                - 'scores': [batch, context] similarity scores
                - 'temporal_weights': [batch, context] time-based weights
                - 'target_embeddings': [batch, embed_dim] target embeddings
        """
        target_embed = self.code_embeddings(target_code)
        context_embed = self.context_embeddings(context_codes)
        scores = torch.einsum('be,bce->bc', target_embed, context_embed)
        temporal_weights = torch.exp(-self.time_decay * time_deltas)
        
        return {
            'scores': scores,
            'temporal_weights': temporal_weights,
            'target_embeddings': target_embed
        }
    
    def get_code_embedding(self, code_idx):
        """Get embedding for specific code(s)."""
        return self.code_embeddings(code_idx)
    
    def compute_similarity(self, code_a, code_b):
        """Compute cosine similarity between two codes."""
        embed_a = self.code_embeddings(code_a)
        embed_b = self.code_embeddings(code_b)
        return torch.cosine_similarity(embed_a, embed_b, dim=-1)


def temporal_skip_gram_loss(scores, temporal_weights):
    """
    Compute temporally-weighted negative log-likelihood loss.
    
    Args:
        scores: [batch, context] similarity scores
        temporal_weights: [batch, context] time-based weights
    
    Returns:
        loss: scalar
    """
    log_probs = torch.log_softmax(scores, dim=-1)
    weighted_loss = -(log_probs * temporal_weights).sum() / temporal_weights.sum()
    return weighted_loss


# Training
model = TemporalMed2Vec(vocab_size=10000, embed_dim=128, time_decay=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    for target, context, deltas in dataloader:
        # Forward pass
        outputs = model(target, context, deltas)
        
        # Compute loss
        loss = temporal_skip_gram_loss(outputs['scores'], 
                                       outputs['temporal_weights'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Inference: Multiple use cases
model.eval()
with torch.no_grad():
    # Use case 1: Get embedding for diabetes code
    diabetes_code = torch.tensor([42])
    diabetes_embed = model.get_code_embedding(diabetes_code)
    
    # Use case 2: Find similar codes
    similarity = model.compute_similarity(
        torch.tensor([42]),  # Diabetes
        torch.tensor([100])  # Metformin
    )
    
    # Use case 3: Predict context given target
    outputs = model(target, context, deltas)
    predictions = outputs['scores'].argmax(dim=-1)
```

---

### Summary

1. **Vocab Size:** Depends on your dataset's unique codes (typically 5k-50k after filtering). Filter rare codes and handle mixed code systems carefully.

2. **Einsum:** Compact notation for tensor operations. Pattern: repeated indices = sum, indices in output = keep. The TemporalMed2Vec line computes batch dot products between target and context embeddings.

3. **Model Loss:** Not required. Best practice is to separate model (returns representations) from loss function (computes training objective). The original code returns loss for simplicity, but production code should separate concerns.

---

## Next Steps

1. **Experiment with vocab_size filtering:** Plot embedding quality vs. minimum frequency threshold
2. **Practice einsum:** Try rewriting common operations in your codebase
3. **Refactor training code:** Separate model outputs from loss computation

Feel free to ask follow-up questions on any of these topics!
