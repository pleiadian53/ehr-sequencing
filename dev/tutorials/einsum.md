## Mastering Einsum Expressions

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

| Operation         | Traditional                       | Einsum           |
| ----------------- | --------------------------------- | ---------------- |
| Dot product       | `(a*b).sum()`                     | `'i,i->'`        |
| Matrix-vector     | `A @ v`                           | `'ij,j->i'`      |
| Matrix-matrix     | `A @ B`                           | `'ik,kj->ij'`    |
| Outer product     | `a.unsqueeze(1) * b.unsqueeze(0)` | `'i,j->ij'`      |
| Batch dot product | `(a*b).sum(-1)`                   | `'bi,bi->b'`     |
| Batch matmul      | `torch.bmm(A, B)`                 | `'bik,bkj->bij'` |
| Transpose         | `A.T`                             | `'ij->ji'`       |
| Diagonal          | `torch.diag(A)`                   | `'ii->i'`        |
| Trace             | `torch.trace(A)`                  | `'ii->'`         |

---

## 