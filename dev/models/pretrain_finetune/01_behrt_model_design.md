# BEHRT Model Design and Architecture

**Last Updated:** 2026-02-02  
**Author:** Private development notes

## Overview

This document provides a comprehensive walkthrough of the BEHRT (BERT for Electronic Health Records) model implementation in the EHR-sequencing project, focusing on model design, pretrained embeddings, LoRA adaptation, and the pretrain/finetune workflow.

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Embedding Design](#embedding-design)
3. [Pretrained Embeddings](#pretrained-embeddings)
4. [LoRA (Low-Rank Adaptation)](#lora-low-rank-adaptation)
5. [Training Workflow](#training-workflow)
6. [Key Implementation Details](#key-implementation-details)

---

## Model Architecture

### Core BEHRT Model

**Location:** `src/ehrsequencing/models/behrt.py`

The BEHRT model is implemented as a transformer-based architecture with three main components:

```
BEHRT = Embeddings → Transformer Encoder → Output Layer
```

### Model Variants

The implementation provides **three pre-configured model sizes**:

| Model Size | Embedding Dim | Hidden Dim | Layers | Heads | Max Seq | Use Case |
|-----------|---------------|------------|--------|-------|---------|----------|
| **Small** | 64 | 128 | 2 | 4 | 50 | Local dev (M1 16GB) |
| **Medium** | 128 | 256 | 4 | 8 | 100 | Small GPU/Workstation |
| **Large** | 256 | 512 | 6 | 8 | 200 | Cloud GPU (A40/A100) |

**Configuration Example:**

```python
from ehrsequencing.models.behrt import BEHRTConfig

# Create a large model configuration
config = BEHRTConfig.large(vocab_size=1000)
# BEHRTConfig(
#     vocab_size=1000,
#     embedding_dim=256,
#     hidden_dim=512,
#     num_layers=6,
#     num_heads=8,
#     max_position=200,
#     dropout=0.1
# )
```

### Key Architectural Components

#### 1. BEHRT Base Model (`BEHRT`)

The core model that produces contextualized sequence representations:

```python
class BEHRT(nn.Module):
    def __init__(self, config: BEHRTConfig):
        # 1. Embeddings (code + age + visit + position)
        self.embeddings = BEHRTEmbedding(...)
        
        # 2. Optional projection if embedding_dim != hidden_dim
        self.embedding_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # 3. Transformer encoder (PyTorch native)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LayerNorm like modern transformers
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 4. Output layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
```

**Forward Pass:**

```python
def forward(self, codes, ages, visit_ids, attention_mask=None):
    # Step 1: Create embeddings
    embeddings = self.embeddings(codes, ages, visit_ids)  # [B, L, E]
    
    # Step 2: Project to hidden dimension
    hidden_states = self.embedding_projection(embeddings)  # [B, L, H]
    
    # Step 3: Pass through transformer
    hidden_states = self.encoder(
        hidden_states, 
        src_key_padding_mask=~attention_mask  # True = masked
    )
    
    # Step 4: Final layer norm
    hidden_states = self.layer_norm(hidden_states)  # [B, L, H]
    
    return hidden_states
```

#### 2. Task-Specific Heads

The implementation provides several task-specific model variants:

##### a. **BEHRTForMLM** (Masked Language Modeling)

Used for pre-training with masked prediction objective:

```python
class BEHRTForMLM(nn.Module):
    def __init__(self, config):
        self.behrt = BEHRT(config)
        
        # MLM prediction head
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, codes, ages, visit_ids, attention_mask, labels=None):
        hidden_states = self.behrt(codes, ages, visit_ids, attention_mask)
        logits = self.mlm_head(hidden_states)  # [B, L, vocab_size]
        
        if labels is not None:
            loss = CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, vocab_size), 
                labels.view(-1)
            )
            return logits, loss
        return logits, None
```

**Key Design Choice:** The MLM head is a 2-layer MLP with GELU activation and LayerNorm, similar to BERT's design.

##### b. **BEHRTForNextVisitPrediction**

Predicts codes in the next visit (multi-label classification):

```python
class BEHRTForNextVisitPrediction(nn.Module):
    def forward(self, codes, ages, visit_ids, attention_mask, labels=None):
        # Get patient embedding from CLS token
        patient_emb = self.behrt.get_patient_embedding(
            codes, ages, visit_ids, attention_mask, pooling='cls'
        )
        
        # Predict next visit codes (multi-hot)
        logits = self.nvp_head(patient_emb)  # [B, vocab_size]
        
        if labels is not None:
            loss = BCEWithLogitsLoss()(logits, labels.float())
            return logits, loss
        return logits, None
```

##### c. **BEHRTForSequenceClassification**

For downstream tasks (diagnosis prediction, readmission, etc.):

```python
class BEHRTForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels, pooling='cls'):
        self.behrt = BEHRT(config)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )
```

**Pooling Strategies:**
- `'cls'`: Use first token (CLS) representation
- `'mean'`: Average over valid positions (masked average pooling)
- `'max'`: Max pooling over valid positions

---

## Embedding Design

**Location:** `src/ehrsequencing/models/embeddings.py`

### Multi-Component Temporal Embeddings

BEHRT uses a **composite embedding** that captures multiple aspects of temporal information:

```
final_embedding = code_emb + age_emb + visit_emb + position_emb
```

### 1. Code Embeddings

Standard learnable embeddings for medical codes:

```python
self.code_embedding = nn.Embedding(
    vocab_size, 
    embedding_dim, 
    padding_idx=0  # Padding token = 0
)
```

### 2. Age Embeddings

Ages are **binned into discrete intervals** before embedding:

```python
class AgeEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_age=100, age_bin_size=5):
        self.num_bins = (max_age // age_bin_size) + 2  # +2 for boundaries
        self.embedding = nn.Embedding(self.num_bins, embedding_dim)
    
    def forward(self, ages):
        # Bin continuous ages: [0-5, 5-10, ..., 95-100, 100+]
        age_bins = (ages / self.age_bin_size).long()
        return self.embedding(age_bins)
```

**Design Rationale:** Discrete bins allow the model to learn age-specific patterns (e.g., pediatric vs. geriatric conditions).

### 3. Visit Embeddings

Each visit in a patient's sequence gets a unique embedding:

```python
class VisitEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_visits=512):
        self.embedding = nn.Embedding(max_visits, embedding_dim)
    
    def forward(self, visit_ids):
        # visit_ids: [0, 0, 0, 1, 1, 2, 2, 2, ...]
        return self.embedding(visit_ids)
```

**Key Insight:** Visit embeddings capture visit-level patterns (e.g., first visit vs. follow-up).

### 4. Position Embeddings

Two options available:

**a. Learnable Positional Embeddings (default):**

```python
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_position=512):
        self.embedding = nn.Embedding(max_position, embedding_dim)
    
    def forward(self, seq_length, device):
        positions = torch.arange(seq_length, device=device).unsqueeze(0)
        return self.embedding(positions)  # [1, seq_length, embedding_dim]
```

**b. Sinusoidal Positional Encodings (optional):**

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_position=512):
        pe = torch.zeros(max_position, embedding_dim)
        position = torch.arange(0, max_position).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
```

### Combined BEHRT Embedding Layer

```python
class BEHRTEmbedding(nn.Module):
    def forward(self, codes, ages, visit_ids):
        # Get individual embeddings
        code_emb = self.code_embedding(codes)          # [B, L, E]
        age_emb = self.age_embedding(ages)              # [B, L, E]
        visit_emb = self.visit_embedding(visit_ids)     # [B, L, E]
        pos_emb = self.position_embedding(seq_length)   # [1, L, E]
        
        # Sum all embeddings
        embeddings = code_emb + age_emb + visit_emb + pos_emb
        
        # Layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
```

**Design Note:** Unlike concatenation, **element-wise addition** preserves dimensionality and follows BERT's design.

---

## Pretrained Embeddings

**Location:** `src/ehrsequencing/models/pretrained_embeddings.py`

### Loading Pretrained Embeddings

The implementation supports loading embeddings from various sources:

#### 1. Med2Vec Embeddings

```python
from ehrsequencing.models.pretrained_embeddings import (
    load_med2vec_embeddings,
    initialize_embedding_layer
)

# Load pretrained embeddings
embeddings = load_med2vec_embeddings(
    'path/to/med2vec_embeddings.pt',
    vocab_size=1000,
    embedding_dim=128
)
# embeddings shape: [1000, 128]

# Initialize BEHRT's code embedding layer
model = BEHRTForMLM(config)
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    pretrained_embeddings=embeddings,
    freeze=True  # Freeze embeddings during training
)
```

#### 2. Word2Vec Embeddings

```python
from ehrsequencing.models.pretrained_embeddings import load_word2vec_embeddings

vocab_mapping = {
    0: '[PAD]',
    1: 'ICD9_250.00',  # Diabetes
    2: 'ICD9_401.9',   # Hypertension
    # ...
}

embeddings = load_word2vec_embeddings(
    'path/to/word2vec.model',
    vocab_mapping=vocab_mapping,
    embedding_dim=128
)
```

### Pretrained Embeddings Workflow

**Typical workflow when using pretrained embeddings:**

```python
# 1. Create model
config = BEHRTConfig.large(vocab_size=1000)
model = BEHRTForMLM(config)

# 2. Load pretrained embeddings (e.g., from Med2Vec)
pretrained_embeddings = load_med2vec_embeddings(
    'med2vec_embeddings.pt', 
    vocab_size=1000, 
    embedding_dim=256
)

# 3. Initialize code embeddings
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    pretrained_embeddings,
    freeze=False  # Allow fine-tuning
)

# 4. Apply LoRA to transformer (freeze base, train adapters)
model = apply_lora_to_behrt(
    model,
    rank=16,
    train_embeddings=True,  # Keep embeddings trainable
    train_head=True         # Keep MLM head trainable
)

# 5. Train
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

### Saving and Loading Embeddings

```python
from ehrsequencing.models.pretrained_embeddings import (
    save_embeddings,
    load_embeddings
)

# After training, save learned embeddings
embeddings = model.behrt.embeddings.code_embedding.weight.data
save_embeddings(
    embeddings,
    'behrt_trained_embeddings.pt',
    metadata={'vocab_size': 1000, 'embedding_dim': 256}
)

# Load later
embeddings, metadata = load_embeddings('behrt_trained_embeddings.pt')
```

---

## LoRA (Low-Rank Adaptation)

**Location:** `src/ehrsequencing/models/lora.py`

### What is LoRA?

LoRA (Hu et al., 2021) is a **parameter-efficient fine-tuning** method that:
- Freezes pretrained weights W
- Injects trainable low-rank matrices: **ΔW = BA**
  - B ∈ R^(d×r), A ∈ R^(r×k), where **r << min(d, k)**
- Forward pass: **h = Wx + (BA)x**

**Key Benefits:**
- Reduces trainable parameters by 90-99%
- Maintains comparable performance
- Enables fine-tuning on consumer hardware
- Only need to save LoRA weights (much smaller)

### LoRA Implementation

#### Core LoRA Layer

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank  # Scaling factor
        
        # Low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # Ensures ΔW=0 at start
    
    def forward(self, x):
        # (BA)x with scaling
        result = x @ self.lora_A.T  # [*, rank]
        result = result @ self.lora_B.T  # [*, out_features]
        return result * self.scaling
```

#### Linear Layer with LoRA

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16.0):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)  # Original + adaptation
```

### Applying LoRA to BEHRT

The `apply_lora_to_behrt()` function provides a **high-level interface** for applying LoRA:

```python
from ehrsequencing.models.lora import apply_lora_to_behrt

model = BEHRTForMLM(config)

model = apply_lora_to_behrt(
    model,
    rank=16,                    # LoRA rank (higher = more capacity)
    alpha=16.0,                 # Scaling factor (typically = rank)
    dropout=0.0,                # LoRA dropout
    lora_attention=True,        # Apply to attention layers
    lora_feedforward=False,     # Apply to FFN layers (optional)
    freeze_base=True,           # Freeze all base weights
    train_embeddings=True,      # Keep embeddings trainable
    train_head=True             # Keep task head trainable
)
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rank` | 8 | Low-rank dimension (8-64 typical) |
| `alpha` | 16.0 | Scaling factor (α/r) |
| `lora_attention` | True | Apply LoRA to attention layers |
| `lora_feedforward` | False | Apply LoRA to FFN layers |
| `train_embeddings` | True | **Critical when training from scratch** |
| `train_head` | True | **Critical for task-specific heads** |

### LoRA Training Patterns

#### Pattern 1: Training from Scratch with LoRA

```python
# When training from scratch (no pretrained weights)
model = apply_lora_to_behrt(
    model,
    rank=16,
    train_embeddings=True,  # MUST be True
    train_head=True          # MUST be True
)

# Result: Only LoRA adapters in transformer are trainable
# But embeddings and head are also trainable
```

**Parameter Breakdown:**
- LoRA adapters: ~0.5% of total parameters
- Embeddings: ~30% of total parameters
- MLM head: ~20% of total parameters
- **Total trainable: ~50% of parameters**

#### Pattern 2: Fine-tuning Pretrained Model

```python
# When fine-tuning a pretrained BEHRT model
model = apply_lora_to_behrt(
    model,
    rank=8,
    train_embeddings=False,  # Freeze pretrained embeddings
    train_head=True          # Adapt head to new task
)

# Result: Only LoRA adapters and head are trainable
# Embeddings are frozen
```

**Parameter Breakdown:**
- LoRA adapters: ~0.5% of total parameters
- Head: ~20% of total parameters
- **Total trainable: ~20% of parameters**

### Parameter Counting

```python
from ehrsequencing.models.lora import count_parameters

param_counts = count_parameters(model)
print(f"Total: {param_counts['total']:,}")
print(f"Trainable: {param_counts['trainable']:,} ({param_counts['trainable_percent']:.1f}%)")
print(f"LoRA: {param_counts['lora']:,} ({param_counts['lora_percent']:.1f}%)")
print(f"Embeddings: {param_counts['embedding_trainable']:,}/{param_counts['embedding_total']:,}")
print(f"Head: {param_counts['head_trainable']:,}/{param_counts['head_total']:,}")
```

**Example Output:**

```
Total: 26,358,784
Trainable: 13,179,392 (50.0%)
LoRA: 131,072 (0.5%)
Embeddings: 6,553,600/6,553,600
Head: 6,553,600/6,553,600
```

### Saving and Loading LoRA Weights

#### Save Only LoRA Weights (Efficient)

```python
from ehrsequencing.models.lora import save_lora_weights

# Save only LoRA adapters (very small file)
save_lora_weights(model, 'lora_weights.pt')
# Saved LoRA weights to lora_weights.pt
# LoRA parameters: 131,072
```

**File Size Comparison:**
- Full model: ~100 MB
- LoRA weights only: ~0.5 MB (**200x smaller!**)

#### Load LoRA Weights

```python
from ehrsequencing.models.lora import load_lora_weights

# 1. Create base model
model = BEHRTForMLM(config)

# 2. Apply LoRA (recreates architecture)
model = apply_lora_to_behrt(model, rank=16)

# 3. Load trained LoRA weights
load_lora_weights(model, 'lora_weights.pt')
```

---

## Training Workflow

**Location:** `examples/pretrain_finetune/train_behrt_demo.py`

### Complete Training Pipeline

The training script demonstrates the full workflow:

```python
# 1. Configure model
config = BEHRTConfig.large(vocab_size=1000)
model = BEHRTForMLM(config).to(device)

# 2. Apply LoRA
model = apply_lora_to_behrt(
    model,
    rank=16,
    train_embeddings=True,  # Training from scratch
    train_head=True
)

# 3. Generate synthetic data
codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_demo_dataset(
    num_patients=5000,
    vocab_size=1000,
    max_seq_length=200,
    seed=42
)

# 4. Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# 5. Setup optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

# 6. Training loop
for epoch in range(100):
    # Train
    model.train()
    for batch in train_loader:
        masked_codes, ages, visit_ids, attention_mask, labels = batch
        
        logits, loss = model(masked_codes, ages, visit_ids, attention_mask, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validate
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            logits, loss = model(...)
            # Compute metrics...
```

### Auto-Resource Detection

The script includes **intelligent resource detection** that automatically configures:
- Model size (small/medium/large)
- Batch size
- Number of patients
- Training epochs
- LoRA rank

```python
from ehrsequencing.utils.resource_manager import get_recommended_config

recommended_config, resources = get_recommended_config(
    task='demo',
    model_size_override=None,  # Auto-detect
    verbose=True
)

print(f"Detected: {resources.gpu_type}")
print(f"VRAM: {resources.vram_gb} GB")
print(f"Recommended: {recommended_config.model_size} model")
print(f"Batch size: {recommended_config.batch_size}")
```

**Example Output:**

```
Detected GPU: NVIDIA A40 (48.0 GB VRAM)
Detected hardware: A40
VRAM: 48.0 GB
Recommended: large model
Batch size: 128
```

### Experiment Tracking

The script uses `ExperimentTracker` for comprehensive logging:

```python
from ehrsequencing.utils.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker('behrt_large_mlm_lora16', output_dir='experiments')

# Log hyperparameters
tracker.log_hyperparameters({
    'model_size': 'large',
    'vocab_size': 1000,
    'batch_size': 128,
    'lr': 1e-4,
    'use_lora': True,
    'lora_rank': 16
})

# Log metrics each epoch
tracker.log_metrics(epoch, {
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_accuracy': val_acc,
    'val_top_5_accuracy': val_top5,
    'val_macro_f1': val_f1
})

# Save checkpoints
tracker.save_lora_checkpoint(model, epoch, metrics, is_best=True)

# Generate plots
tracker.plot_training_curves()

# Save summary
tracker.save_summary()
```

**Output Structure:**

```
experiments/behrt_large_mlm_lora16/
├── checkpoints/
│   ├── best_lora.pt
│   └── latest_lora.pt
├── plots/
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── top_5_accuracy_curve.png
│   ├── macro_f1_curve.png
│   └── perplexity_curve.png
├── logs/
│   └── metrics_history.json
├── hyperparameters.json
├── metadata.json
├── summary.json
└── SUMMARY.txt
```

---

## Key Implementation Details

### 1. **Does BEHRT use pretrained models from HuggingFace?**

**No.** The current implementation:
- Does **NOT** use HuggingFace transformers
- Does **NOT** load pretrained weights from HuggingFace Hub
- Uses **PyTorch native transformers** (`nn.TransformerEncoder`)
- Trains from scratch or from custom pretrained embeddings

**Rationale:**
- Medical codes (ICD-9, ICD-10) have different vocabulary than NLP
- Pre-training is done on EHR data, not general text
- Custom embedding design (age, visit, position) is domain-specific

**Future Extension:**
Could potentially integrate with HuggingFace by:
1. Creating a HuggingFace-compatible model wrapper
2. Using HuggingFace's training utilities
3. Hosting pretrained checkpoints on HuggingFace Hub

### 2. **How to provide pretrained embeddings?**

**Three approaches:**

#### Approach 1: Load from Med2Vec

```python
from ehrsequencing.models.pretrained_embeddings import (
    load_med2vec_embeddings,
    initialize_embedding_layer
)

# After training Med2Vec in Phase 2
embeddings = load_med2vec_embeddings('med2vec_embeddings.pt', 1000, 256)

model = BEHRTForMLM(config)
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    embeddings,
    freeze=False  # Allow fine-tuning
)
```

#### Approach 2: Load from Word2Vec

```python
from ehrsequencing.models.pretrained_embeddings import load_word2vec_embeddings

embeddings = load_word2vec_embeddings(
    'word2vec.model',
    vocab_mapping={...},
    embedding_dim=256
)

model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    embeddings,
    freeze=False
)
```

#### Approach 3: Transfer from another BEHRT model

```python
# Load pretrained BEHRT
checkpoint = torch.load('pretrained_behrt.pt')
pretrained_model = BEHRTForMLM(config)
pretrained_model.load_state_dict(checkpoint)

# Extract embeddings
embeddings = pretrained_model.behrt.embeddings.code_embedding.weight.data

# Transfer to new model
new_model = BEHRTForMLM(config)
new_model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    new_model.behrt.embeddings.code_embedding,
    embeddings,
    freeze=False
)
```

### 3. **How to apply LoRA to BEHRT?**

**Step-by-step:**

```python
from ehrsequencing.models.behrt import BEHRTForMLM, BEHRTConfig
from ehrsequencing.models.lora import apply_lora_to_behrt, count_parameters

# 1. Create base model
config = BEHRTConfig.large(vocab_size=1000)
model = BEHRTForMLM(config)

# 2. Apply LoRA
model = apply_lora_to_behrt(
    model,
    rank=16,                    # Rank of low-rank decomposition
    alpha=16.0,                 # Scaling factor
    lora_attention=True,        # Apply to Q, K, V, O projections
    lora_feedforward=False,     # Don't apply to FFN (usually not needed)
    train_embeddings=True,      # Keep embeddings trainable (from scratch)
    train_head=True             # Keep MLM head trainable
)

# 3. Check parameters
params = count_parameters(model)
print(f"Total: {params['total']:,}")
print(f"Trainable: {params['trainable']:,} ({params['trainable_percent']:.1f}%)")
print(f"LoRA: {params['lora']:,} ({params['lora_percent']:.1f}%)")

# 4. Train as usual
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

**LoRA Target Modules:**

The implementation applies LoRA to:
- `self_attn.in_proj_weight` (Q, K, V projections)
- `self_attn.out_proj` (output projection)

Optionally:
- `linear1` (first FFN layer)
- `linear2` (second FFN layer)

### 4. **Key Design Decisions**

#### a. **Attention Implementation**

Uses PyTorch's native `nn.TransformerEncoderLayer` with:
- `batch_first=True` (modern convention)
- `norm_first=True` (Pre-LayerNorm, more stable)
- `activation='gelu'` (smoother than ReLU)

#### b. **Embedding Projection**

If `embedding_dim != hidden_dim`, a linear projection is used:

```python
if config.embedding_dim != config.hidden_dim:
    self.embedding_projection = nn.Linear(embedding_dim, hidden_dim)
```

This allows flexibility in embedding size independent of transformer hidden size.

#### c. **Attention Mask Convention**

- Input mask: `1` = valid, `0` = padding
- Transformer mask: `True` = padding, `False` = valid

```python
# Convert from input to transformer convention
src_key_padding_mask = ~attention_mask.bool()
```

#### d. **MLM Label Convention**

- `-100` = ignore position (not masked)
- `0` to `vocab_size-1` = true label for masked position

```python
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
```

---

## Summary

### Key Takeaways

1. **Architecture**: BEHRT uses PyTorch native transformers with custom temporal embeddings (code + age + visit + position)

2. **No HuggingFace**: Implementation is self-contained, does not use pretrained HuggingFace models

3. **Pretrained Embeddings**: Support for Med2Vec, Word2Vec, and custom embeddings via `initialize_embedding_layer()`

4. **LoRA**: Efficient fine-tuning via `apply_lora_to_behrt()`, reduces trainable parameters by 50-99%

5. **Training Workflow**: Comprehensive pipeline with auto-resource detection, experiment tracking, and comprehensive metrics

6. **Model Sizes**: Three configs (small/medium/large) optimized for different hardware

### Usage Patterns

#### Pattern 1: Train from Scratch

```python
config = BEHRTConfig.large(vocab_size=1000)
model = BEHRTForMLM(config)
model = apply_lora_to_behrt(model, rank=16, train_embeddings=True, train_head=True)
# Train on data...
```

#### Pattern 2: Use Pretrained Embeddings

```python
model = BEHRTForMLM(config)
embeddings = load_med2vec_embeddings('embeddings.pt', 1000, 256)
model.behrt.embeddings.code_embedding = initialize_embedding_layer(..., freeze=False)
model = apply_lora_to_behrt(model, rank=16, train_embeddings=True, train_head=True)
# Train on data...
```

#### Pattern 3: Fine-tune Pretrained BEHRT

```python
# Load pretrained
model = BEHRTForMLM(config)
model.load_state_dict(torch.load('pretrained_behrt.pt'))

# Apply LoRA for efficient fine-tuning
model = apply_lora_to_behrt(model, rank=8, train_embeddings=False, train_head=True)
# Fine-tune on downstream task...
```

---

## References

- **BEHRT Paper**: Li et al. (2019). "BEHRT: Transformer for Electronic Health Records"
- **LoRA Paper**: Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
- **Transformer**: Vaswani et al. (2017). "Attention is All You Need"

---

**Next:** See `02_training_guide.md` for detailed training instructions and best practices.
