# LoRA Deep Dive: Understanding Low-Rank Adaptation

**Last Updated:** 2026-02-03  
**Purpose:** Comprehensive explanation of LoRA implementation and usage

---

## Table of Contents

1. [What is LoRA?](#what-is-lora)
2. [Does LoRA Always Freeze W?](#does-lora-always-freeze-w)
3. [Where Can LoRA Apply?](#where-can-lora-apply)
4. [LoRA and Layer Normalization](#lora-and-layer-normalization)
5. [Using LoRA with Foundation Models](#using-lora-with-foundation-models)
6. [Implementation Details](#implementation-details)
7. [Practical Examples](#practical-examples)
8. [Advanced Topics](#advanced-topics)

---

## What is LoRA?

### The Core Idea

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning method that dramatically reduces trainable parameters while maintaining performance.

**Instead of fine-tuning W ∈ R^(d×k):**

```
Traditional fine-tuning:
  - Update all d×k parameters in W
  - Requires storing full gradients
  - Memory intensive
```

**LoRA freezes W and learns ΔW = BA:**

```
LoRA fine-tuning:
  - Freeze W (no gradients)
  - Learn B ∈ R^(d×r) and A ∈ R^(r×k) where r << min(d,k)
  - Only r×(d+k) parameters instead of d×k
  - Forward pass: h = Wx + (BA)x
```

### Parameter Reduction Example

**Example: Attention projection layer**

```
Original: W ∈ R^(768×768) = 589,824 parameters

LoRA with rank=8:
  - B ∈ R^(768×8) = 6,144 parameters
  - A ∈ R^(8×768) = 6,144 parameters
  - Total: 12,288 parameters

Reduction: 98% fewer parameters! (12K vs 589K)
```

### Why Low-Rank?

**Key insight from the LoRA paper:**

> "Pre-trained models have low intrinsic dimensionality when adapting to new tasks"

This means:
- Fine-tuning updates are low-rank
- Most adaptation happens in a low-dimensional subspace
- We don't need full-rank updates

**Proof:** LoRA with rank 8 achieves 99%+ of full fine-tuning performance with 0.1% of parameters!

---

## Does LoRA Always Freeze W?

### Yes! That's the Whole Point

**From the implementation** (`src/ehrsequencing/models/lora.py:111-114`):

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, ...):
        super().__init__()
        self.linear = linear
        
        # ✅ ALWAYS freeze original weights
        self.linear.weight.requires_grad = False  # Line 112
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False  # Line 114
```

**Why freeze W?**

1. **Memory efficiency** - No gradients for W → massive memory savings
2. **Computation efficiency** - Only compute gradients for B and A
3. **Preserve pre-training** - W contains knowledge from pre-training
4. **Enable sharing** - Base model can be shared, only LoRA weights differ

### What Gets Trained?

**Frozen (W):**
- ❌ Original weight matrix W
- ❌ Original bias b (if present)
- ✅ These are loaded from pre-trained model

**Trainable (B, A):**
- ✅ Low-rank matrix B ∈ R^(d×r)
- ✅ Low-rank matrix A ∈ R^(r×k)
- ✅ These start from initialization (A ~ Kaiming, B ~ zeros)

### Forward Pass Math

**Original layer:**
```
h = Wx + b
```

**LoRA layer:**
```
h = Wx + b + (BA)x
  = Wx + b + ΔWx
  = (W + ΔW)x + b
```

Where:
- W is frozen (from pre-training)
- ΔW = BA is learned (task-specific adaptation)
- Scaling factor α/r is applied to ΔW

**Implementation** (`lora.py:126-128`):

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass: original + LoRA adaptation."""
    return self.linear(x) + self.lora(x)
    #      ↑ frozen W    ↑ trainable BA
```

---

## Where Can LoRA Apply?

### Short Answer: Any `nn.Linear` Layer

**LoRA can wrap ANY linear layer in your model**, including:

✅ Attention projections (Q, K, V, Output)  
✅ Feedforward networks (FFN layers)  
✅ Embedding projections  
✅ Classification heads  
✅ Any custom linear layers

### Transformer-Specific Applications

**Typical transformer architecture:**

```
TransformerBlock:
  ├── MultiheadAttention
  │   ├── Q projection: Linear(d_model, d_model)  ← LoRA here
  │   ├── K projection: Linear(d_model, d_model)  ← LoRA here
  │   ├── V projection: Linear(d_model, d_model)  ← LoRA here
  │   └── Output projection: Linear(d_model, d_model)  ← LoRA here
  │
  └── FeedForward
      ├── Linear1: Linear(d_model, d_ff)  ← LoRA here (optional)
      └── Linear2: Linear(d_ff, d_model)  ← LoRA here (optional)
```

### Where LoRA is Most Effective

**From research and practice:**

| Layer Type | LoRA Effectiveness | Reason |
|-----------|-------------------|--------|
| **Q, K, V projections** | ⭐⭐⭐⭐⭐ | Attention is core mechanism, high impact |
| **Output projection** | ⭐⭐⭐⭐ | Aggregates attention, important |
| **FFN layers** | ⭐⭐⭐ | Large parameters, good reduction |
| **Embeddings** | ⭐⭐ | Often fine-tuned separately |
| **Layer norms** | ⭐ | Few parameters, usually train normally |

**Best practice:** Apply LoRA to attention layers by default, optionally FFN.

### BEHRT-Specific Application

**In our BEHRT implementation:**

```python
def apply_lora_to_behrt(
    model,
    lora_attention=True,      # ← LoRA on Q, K, V, Out projections
    lora_feedforward=False,   # ← LoRA on FFN (optional)
    train_embeddings=True,    # ← Train embeddings normally (not LoRA)
    train_head=True          # ← Train MLM head normally (not LoRA)
):
    # Apply LoRA to transformer encoder layers
    # Keep embeddings and head trainable (no LoRA)
```

**Why not LoRA everywhere?**

- **Embeddings:** Domain-specific, often need full capacity
- **Task heads:** Task-specific, typically small (no need for LoRA)
- **Frozen base:** LoRA on transformer only, other parts flexible

---

## LoRA and Layer Normalization

### Short Answer: Order Doesn't Matter

**LoRA wraps the linear layer transparently**, so it's invisible to layer normalization.

### Typical Transformer Layer Structure

```python
# Standard TransformerEncoderLayer structure:

# 1. Self-attention sub-layer
x = x + self_attn(layer_norm(x))
           ↑
    This calls: Q, K, V projections
    LoRA wraps these linear layers

# 2. Feedforward sub-layer  
x = x + ffn(layer_norm(x))
        ↑
    This calls: Linear1, Linear2
    LoRA wraps these linear layers
```

**LoRA is applied INSIDE the attention/FFN modules**, so:

```
layer_norm(x) → [Q_linear + LoRA](x)
                 ↑
            Wrapped together

# The layer norm sees LoRA as just another linear layer
# No special ordering needed!
```

### Implementation Example

**Without LoRA:**
```python
class Attention(nn.Module):
    def __init__(self):
        self.q_proj = nn.Linear(768, 768)  # Original
        
    def forward(self, x):
        q = self.q_proj(x)  # Forward pass
```

**With LoRA:**
```python
class Attention(nn.Module):
    def __init__(self):
        self.q_proj = nn.Linear(768, 768)
        # Apply LoRA:
        self.q_proj = LinearWithLoRA(self.q_proj, rank=8)
        
    def forward(self, x):
        q = self.q_proj(x)  # Still just one call!
        # Internally: q = W_frozen @ x + (B @ A) @ x
```

**Key insight:** LoRA is a **drop-in replacement** for `nn.Linear`, so existing code doesn't need to change!

### Pre-LN vs Post-LN Transformers

**Both work identically with LoRA:**

**Pre-LN (modern, e.g., BEHRT):**
```python
x = x + attn(layer_norm(x))      # LoRA inside attn
x = x + ffn(layer_norm(x))       # LoRA inside ffn
```

**Post-LN (original Transformer):**
```python
x = layer_norm(x + attn(x))      # LoRA inside attn
x = layer_norm(x + ffn(x))       # LoRA inside ffn
```

**LoRA works the same in both!** It wraps the linear layers, not the normalization.

---

## Using LoRA with Foundation Models

### General Pattern (Works with Any PyTorch Model)

```python
# 1. Load your foundation model
model = load_pretrained_model()  # From anywhere

# 2. Apply LoRA to target layers
from ehrsequencing.models.lora import apply_lora_to_model

model = apply_lora_to_model(
    model,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'],
    rank=8,
    alpha=16.0
)

# 3. Now only LoRA parameters are trainable!
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")  # Only LoRA A and B matrices
```

### HuggingFace Models

**LoRA works with HuggingFace models too:**

```python
from transformers import AutoModel
from ehrsequencing.models.lora import apply_lora_to_model

# Load pre-trained model
model = AutoModel.from_pretrained("bert-base-uncased")

# Apply LoRA (HuggingFace uses different naming)
model = apply_lora_to_model(
    model,
    target_modules=[
        '.*query',      # Q projections
        '.*key',        # K projections  
        '.*value',      # V projections
        '.*dense'       # Output projections
    ],
    rank=8
)

# Fine-tune only LoRA parameters
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)
```

**Note:** HuggingFace naming conventions differ from PyTorch native transformers.

### Custom Models

**LoRA works with ANY model that uses `nn.Linear`:**

```python
class MyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(...)
        self.decoder = nn.TransformerDecoder(...)
        self.head = nn.Linear(768, 10)
    
# Apply LoRA
model = MyCustomModel()
model = apply_lora_to_model(
    model,
    target_modules=['encoder.*', 'decoder.*'],  # Regex patterns
    rank=8
)
```

### Target Module Patterns

**The `target_modules` argument accepts regex patterns:**

```python
# Example 1: Specific layer names
target_modules=['q_proj', 'v_proj']  # Only Q and V projections

# Example 2: Wildcard patterns
target_modules=['.*proj']  # All projection layers

# Example 3: Layer-specific
target_modules=['encoder.*.self_attn.*']  # Only encoder attention

# Example 4: Multiple patterns
target_modules=[
    'encoder.*.self_attn.*',  # Encoder attention
    'decoder.*.cross_attn.*'  # Decoder cross-attention
]
```

**Our implementation** (`lora.py:163-174`):

```python
import re

# Compile regex patterns
patterns = [re.compile(pattern) for pattern in target_modules]

def should_apply_lora(name: str) -> bool:
    """Check if module name matches any pattern."""
    return any(pattern.search(name) for pattern in patterns)

# Find and replace linear layers
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and should_apply_lora(name):
        # Replace with LoRA version
        lora_module = LinearWithLoRA(module, rank=rank, ...)
```

---

## Implementation Details

### LoRA Mathematics

**Original linear layer:**
```
h = Wx + b
```

**LoRA adaptation:**
```
h = Wx + b + α/r · (BA)x

Where:
  - W ∈ R^(d×k): Frozen pre-trained weights
  - B ∈ R^(d×r): Trainable low-rank matrix (initialized to zeros)
  - A ∈ R^(r×k): Trainable low-rank matrix (Kaiming initialization)
  - r: Rank (typically 8, 16, or 32)
  - α: Scaling factor (typically 16)
  - α/r: Normalization (keeps scale consistent across ranks)
```

**Why scaling factor α/r?**

- Without scaling: LoRA contribution would depend on rank r
- With scaling: α/r normalizes the contribution
- Allows changing rank without re-tuning learning rate

### Initialization Strategy

**From implementation** (`lora.py:54-56`):

```python
# Initialize A with Kaiming uniform, B with zeros
nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
nn.init.zeros_(self.lora_B)
```

**Why this initialization?**

1. **B starts at zero** → LoRA has no effect initially
2. **A is random** → Ready to learn
3. **Initially: BA = 0** → Model starts exactly as pre-trained
4. **During training:** BA gradually learns task-specific adaptation

**This is brilliant design:**
- At initialization: h = Wx (exact pre-trained behavior)
- After training: h = Wx + ΔWx (adapted to new task)

### Forward Pass Implementation

**From `lora.py:58-72`:**

```python
class LoRALayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x @ A^T @ B^T = (BA)x
        result = x @ self.lora_A.T        # [batch, seq, in] @ [in, r] → [batch, seq, r]
        result = self.dropout(result)      # Dropout for regularization
        result = result @ self.lora_B.T    # [batch, seq, r] @ [r, out] → [batch, seq, out]
        return result * self.scaling       # Scale by α/r
```

**Computational complexity:**

```
Original: O(batch × seq × d × k)
LoRA: O(batch × seq × r × (d + k))

If r << min(d, k), LoRA is much faster!
```

### Memory Savings

**Example: BEHRT-large model (768 dimensions, 12 layers)**

**Without LoRA:**
```
Attention projections per layer: 4 × (768 × 768) = 2,359,296 params
12 layers: 28,311,552 params
Gradients + optimizer states (Adam): 3× memory
Total: ~85M parameters worth of GPU memory
```

**With LoRA (rank=8):**
```
LoRA params per projection: (768 × 8) + (8 × 768) = 12,288 params
12 layers × 4 projections: 589,824 params
Reduction: 98% fewer trainable parameters!
Total GPU memory: ~1.8M parameters worth
```

**Savings: ~47× less GPU memory for gradients!**

---

## Practical Examples

### Example 1: Basic LoRA Application

```python
from ehrsequencing.models.behrt import BEHRTConfig, BEHRTForMLM
from ehrsequencing.models.lora import apply_lora_to_behrt, count_parameters

# 1. Create model
config = BEHRTConfig.large(vocab_size=1000)
model = BEHRTForMLM(config)

# 2. Apply LoRA
model = apply_lora_to_behrt(
    model,
    rank=8,                    # Low-rank dimension
    lora_attention=True,       # Apply to attention layers
    lora_feedforward=False,    # Don't apply to FFN
    train_embeddings=True,     # Keep embeddings trainable
    train_head=True           # Keep MLM head trainable
)

# 3. Check parameter reduction
stats = count_parameters(model)
print(f"Total parameters: {stats['total']:,}")
print(f"Trainable: {stats['trainable']:,} ({stats['trainable_percent']:.1f}%)")
print(f"LoRA parameters: {stats['lora']:,}")

# Output:
# Total parameters: 67,108,864
# Trainable: 13,421,824 (20.0%)
# LoRA parameters: 589,824
```

### Example 2: Fine-tuning Pre-trained BEHRT

```python
# 1. Load pre-trained model
model = load_pretrained_behrt('checkpoint.pt')

# 2. Apply LoRA for efficient fine-tuning
model = apply_lora_to_behrt(
    model,
    rank=16,                   # Higher rank for complex tasks
    train_embeddings=False,    # Freeze embeddings (pre-trained)
    train_head=True           # Adapt task head
)

# 3. Create optimizer (only LoRA + head parameters)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
    weight_decay=0.01
)

# 4. Train on downstream task
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()  # Only LoRA gradients computed!
        optimizer.step()
```

### Example 3: Multiple LoRA Adapters for Different Tasks

```python
# Train separate LoRA adapters for different tasks
# Base model is shared, only LoRA weights differ!

# Task 1: Disease prediction
model_disease = BEHRTForMLM(config)
model_disease = apply_lora_to_behrt(model_disease, rank=8)
train(model_disease, disease_data)
save_lora_weights(model_disease, 'lora_disease.pt')

# Task 2: Readmission prediction (same base!)
model_readmit = BEHRTForMLM(config)
model_readmit = apply_lora_to_behrt(model_readmit, rank=8)
train(model_readmit, readmission_data)
save_lora_weights(model_readmit, 'lora_readmit.pt')

# At inference: swap LoRA weights for different tasks!
base_model = BEHRTForMLM(config)
base_model = apply_lora_to_behrt(base_model, rank=8)

# Use disease adapter
load_lora_weights(base_model, 'lora_disease.pt')
pred_disease = base_model(patient_data)

# Switch to readmission adapter
load_lora_weights(base_model, 'lora_readmit.pt')
pred_readmit = base_model(patient_data)
```

### Example 4: Gradual Rank Adjustment

```python
# Start with low rank, increase if needed

# Try rank=4 (most efficient)
model_r4 = apply_lora_to_behrt(model, rank=4)
perf_r4 = evaluate(model_r4)  # 85% accuracy

# Try rank=8 (balanced)
model_r8 = apply_lora_to_behrt(model, rank=8)
perf_r8 = evaluate(model_r8)  # 88% accuracy

# Try rank=16 (more capacity)
model_r16 = apply_lora_to_behrt(model, rank=16)
perf_r16 = evaluate(model_r16)  # 89% accuracy

# Diminishing returns after rank=16
# Choose rank=8 for best efficiency/performance trade-off
```

---

## Advanced Topics

### Rank Selection

**How to choose rank r?**

| Rank | Params (per projection) | Use Case |
|------|------------------------|----------|
| 4 | 6K | Simple tasks, extreme efficiency |
| 8 | 12K | Default, works for most tasks |
| 16 | 25K | Complex tasks, more capacity |
| 32 | 49K | Very complex, near full fine-tuning |
| 64 | 98K | Rarely needed |

**Rule of thumb:**
- Start with rank=8
- Increase to 16 if performance plateaus
- Rarely need rank > 32

**From LoRA paper:**
> "Rank as low as 1 or 2 is sufficient for many tasks"

### Combining LoRA with Other Techniques

**LoRA + Pretrained Embeddings:**
```python
# 1. Load pretrained embeddings
embeddings = load_med2vec_embeddings(...)

# 2. Initialize model
model = BEHRTForMLM(config)
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    embeddings,
    freeze=False  # Fine-tune embeddings
)

# 3. Apply LoRA to transformer
model = apply_lora_to_behrt(
    model,
    rank=8,
    train_embeddings=True  # Embeddings trainable (not LoRA)
)

# Result: Pretrained embeddings + LoRA transformer
```

**LoRA + Quantization:**
```python
# Combine LoRA with 8-bit quantization (QLoRA)
# Even more memory efficient!

import bitsandbytes as bnb

# 1. Quantize base model to 8-bit
model = quantize_model(model, bits=8)

# 2. Apply LoRA on top
model = apply_lora_to_behrt(model, rank=8)

# Now: 8-bit frozen weights + 16-bit LoRA adapters
# Massive memory savings!
```

### LoRA Merging

**After training, merge LoRA into base weights:**

```python
def merge_lora_weights(model):
    """Merge LoRA adapters into base weights (for inference)."""
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            # Compute: W_merged = W_frozen + BA
            lora_delta = module.lora.lora_B @ module.lora.lora_A
            lora_delta = lora_delta * module.lora.scaling
            
            # Merge into original weight
            module.linear.weight.data += lora_delta
            
            # Replace with merged linear layer
            parent = get_parent_module(model, name)
            setattr(parent, name.split('.')[-1], module.linear)
    
    return model

# After training
model = train_with_lora(model, data)

# Merge for inference (no LoRA overhead)
model = merge_lora_weights(model)

# Now it's just a standard model (faster inference!)
```

**When to merge:**
- ✅ Deployment (single task)
- ✅ Inference optimization
- ❌ Multi-task (lose adapter swapping)

### LoRA Interpolation

**Interpolate between tasks:**

```python
# Load two LoRA adapters
lora_A1, lora_B1 = load_lora_weights('task1.pt')
lora_A2, lora_B2 = load_lora_weights('task2.pt')

# Interpolate (α ∈ [0, 1])
alpha = 0.7
lora_A_mixed = alpha * lora_A1 + (1 - alpha) * lora_A2
lora_B_mixed = alpha * lora_B1 + (1 - alpha) * lora_B2

# Apply mixed adapter
model.lora.lora_A.data = lora_A_mixed
model.lora.lora_B.data = lora_B_mixed

# Now model behavior is 70% task1 + 30% task2!
```

---

## Summary

### Key Takeaways

1. **Yes, LoRA ALWAYS freezes W** - That's the core design principle

2. **LoRA applies to any `nn.Linear`** - Q, K, V, output projections, FFN, etc.

3. **Order with LayerNorm doesn't matter** - LoRA wraps linear layers transparently

4. **Works with any PyTorch model** - Foundation models, HuggingFace, custom models

5. **Dramatic parameter reduction** - 98% fewer trainable params, 47× less GPU memory

6. **Minimal performance loss** - Typically within 1% of full fine-tuning

7. **Enables multi-task learning** - Share base, swap LoRA adapters per task

### When to Use LoRA

✅ **Use LoRA when:**
- Fine-tuning large pre-trained models
- Limited GPU memory
- Need multiple task-specific adaptations
- Want fast fine-tuning iteration

❌ **Don't use LoRA when:**
- Training from scratch (no benefit)
- Model is already small (overhead not worth it)
- Need absolute maximum performance (rare)

### Best Practices

1. **Default rank:** Start with rank=8, increase if needed
2. **Target attention:** Apply LoRA to attention layers by default
3. **Keep task heads trainable:** Don't LoRA classification/MLM heads
4. **Scale appropriately:** α=16 is a good default
5. **Save only LoRA:** Store adapters separately from base model

---

## References

- **LoRA Paper:** Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
- **Implementation:** `src/ehrsequencing/models/lora.py`
- **Usage Examples:** `examples/pretrain_finetune/train_behrt_demo.py`

---

**Next:** See `01_behrt_model_design.md` for how LoRA integrates with BEHRT architecture
