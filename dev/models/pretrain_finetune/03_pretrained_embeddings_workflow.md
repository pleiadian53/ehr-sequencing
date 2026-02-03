# Pretrained Embeddings Workflow

**Last Updated:** 2026-02-02  
**Author:** Private development notes

## Overview

This document explains how to integrate pretrained medical code embeddings (from Med2Vec, Word2Vec, etc.) into BEHRT models for improved performance and faster convergence.

## Table of Contents

1. [Why Use Pretrained Embeddings?](#why-use-pretrained-embeddings)
2. [Med2Vec Integration](#med2vec-integration)
3. [Word2Vec Integration](#word2vec-integration)
4. [Custom Embeddings](#custom-embeddings)
5. [Embedding Analysis](#embedding-analysis)
6. [Complete Workflows](#complete-workflows)
7. [Best Practices](#best-practices)

---

## Why Use Pretrained Embeddings?

### Benefits

1. **Faster Convergence**: Pretrained embeddings provide a better initialization than random
2. **Better Performance**: Leverage knowledge from unsupervised pre-training
3. **Transfer Learning**: Use embeddings trained on larger datasets
4. **Domain Knowledge**: Med2Vec captures medical semantics (co-occurrence patterns)

### When to Use

✅ **Use pretrained embeddings when:**
- You have Med2Vec/Word2Vec trained on similar data
- Dataset is small (< 10K patients)
- Need faster convergence
- Vocabulary matches pretrained embeddings

❌ **Train from scratch when:**
- No suitable pretrained embeddings available
- Vocabulary is very different
- Have large dataset (> 100K patients)
- Want to learn task-specific embeddings

### Performance Comparison

| Initialization | Epochs to Converge | Final Accuracy | Training Time |
|----------------|-------------------|----------------|---------------|
| Random | 50-100 | 75% | Long |
| Med2Vec | 20-30 | 82% | Medium |
| Word2Vec | 30-50 | 78% | Medium |

---

## Med2Vec Integration

### Overview

Med2Vec (Choi et al., 2016) learns medical code embeddings by predicting visit sequences. The embeddings capture:
- Code co-occurrence patterns
- Disease progression patterns
- Treatment patterns

### Step 1: Train or Load Med2Vec Embeddings

**Option A: Load existing embeddings**

```python
from ehrsequencing.models.pretrained_embeddings import load_med2vec_embeddings

# Load pretrained Med2Vec embeddings
embeddings = load_med2vec_embeddings(
    embedding_path='path/to/med2vec_embeddings.pt',
    vocab_size=1000,
    embedding_dim=256
)
# embeddings: torch.Tensor of shape [1000, 256]
```

**Option B: Train Med2Vec (from Phase 2)**

```python
# Assume Med2Vec model was trained in Phase 2
from med2vec import Med2Vec

# Train Med2Vec on EHR data
med2vec = Med2Vec(
    num_codes=1000,
    embedding_dim=256,
    hidden_dim=512
)
med2vec.train(visit_sequences, ...)

# Extract code embeddings
embeddings = med2vec.code_embeddings.weight.data  # [1000, 256]

# Save for later use
torch.save(embeddings, 'med2vec_embeddings.pt')
```

### Step 2: Initialize BEHRT with Med2Vec Embeddings

```python
from ehrsequencing.models.behrt import BEHRTConfig, BEHRTForMLM
from ehrsequencing.models.pretrained_embeddings import (
    load_med2vec_embeddings,
    initialize_embedding_layer
)

# 1. Create BEHRT model
config = BEHRTConfig.large(vocab_size=1000)
model = BEHRTForMLM(config)

# 2. Load Med2Vec embeddings
embeddings = load_med2vec_embeddings(
    'med2vec_embeddings.pt',
    vocab_size=1000,
    embedding_dim=256
)

# 3. Initialize code embedding layer
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    embedding_layer=model.behrt.embeddings.code_embedding,
    pretrained_embeddings=embeddings,
    freeze=False  # Allow fine-tuning
)
# ✅ Initialized embedding layer with pre-trained weights (trainable)
```

### Step 3: Apply LoRA and Train

```python
from ehrsequencing.models.lora import apply_lora_to_behrt

# Apply LoRA to transformer (freeze base, train adapters)
model = apply_lora_to_behrt(
    model,
    rank=16,
    train_embeddings=True,  # Keep Med2Vec embeddings trainable
    train_head=True         # Keep MLM head trainable
)

# Train as usual
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=0.01
)

# Training loop...
```

### Step 4: Monitor Convergence

**Expected behavior with Med2Vec:**
- Faster initial loss decrease
- Higher initial accuracy (10-20%)
- Converges in 20-30 epochs (vs. 50+ from scratch)

```
Epoch 1/100 | Train Loss: 3.2341 Acc: 0.1234 | Val Loss: 2.8765 Acc: 0.1456
Epoch 5/100 | Train Loss: 1.8234 Acc: 0.4234 | Val Loss: 1.6765 Acc: 0.4856
Epoch 10/100 | Train Loss: 0.9234 Acc: 0.6534 | Val Loss: 0.8965 Acc: 0.6756
Epoch 20/100 | Train Loss: 0.4234 Acc: 0.8134 | Val Loss: 0.4565 Acc: 0.8056 🏆
```

### Complete Med2Vec Workflow Script

```python
"""
Complete workflow for using Med2Vec embeddings with BEHRT.
"""
import torch
from ehrsequencing.models.behrt import BEHRTConfig, BEHRTForMLM
from ehrsequencing.models.lora import apply_lora_to_behrt, count_parameters
from ehrsequencing.models.pretrained_embeddings import (
    load_med2vec_embeddings,
    initialize_embedding_layer
)

def train_behrt_with_med2vec(
    med2vec_path: str,
    vocab_size: int,
    embedding_dim: int,
    device: str = 'cuda'
):
    """Train BEHRT with Med2Vec pretrained embeddings."""
    
    # 1. Load Med2Vec embeddings
    print("Loading Med2Vec embeddings...")
    embeddings = load_med2vec_embeddings(
        med2vec_path,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim
    )
    print(f"✅ Loaded embeddings: {embeddings.shape}")
    
    # 2. Create BEHRT model
    print("\nCreating BEHRT model...")
    config = BEHRTConfig.large(vocab_size=vocab_size)
    config.embedding_dim = embedding_dim  # Match Med2Vec dimension
    model = BEHRTForMLM(config).to(device)
    
    # 3. Initialize with Med2Vec embeddings
    print("\nInitializing code embeddings with Med2Vec...")
    model.behrt.embeddings.code_embedding = initialize_embedding_layer(
        model.behrt.embeddings.code_embedding,
        pretrained_embeddings=embeddings,
        freeze=False  # Allow fine-tuning
    )
    
    # 4. Apply LoRA
    print("\nApplying LoRA...")
    model = apply_lora_to_behrt(
        model,
        rank=16,
        train_embeddings=True,  # Keep embeddings trainable
        train_head=True
    )
    
    # 5. Show parameter counts
    params = count_parameters(model)
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {params['total']:,}")
    print(f"   Trainable: {params['trainable']:,} ({params['trainable_percent']:.1f}%)")
    print(f"   LoRA: {params['lora']:,} ({params['lora_percent']:.1f}%)")
    print(f"   Embeddings: {params['embedding_trainable']:,}/{params['embedding_total']:,}")
    
    # 6. Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    print("\n✅ Model ready for training with Med2Vec embeddings!")
    
    return model, optimizer

# Usage
model, optimizer = train_behrt_with_med2vec(
    med2vec_path='path/to/med2vec_embeddings.pt',
    vocab_size=1000,
    embedding_dim=256
)
```

---

## Word2Vec Integration

### Overview

Word2Vec (Mikolov et al., 2013) learns embeddings by predicting context. Applied to medical codes, it captures:
- Code co-occurrence patterns
- Local context (within same visit)

### Step 1: Train or Load Word2Vec

```python
from gensim.models import Word2Vec

# Train Word2Vec on code sequences
sequences = [
    ['ICD9_250.00', 'ICD9_401.9', 'ICD9_272.4'],
    ['ICD9_250.00', 'ICD9_357.2', 'ICD9_250.01'],
    # ... more sequences
]

model = Word2Vec(
    sentences=sequences,
    vector_size=128,
    window=5,
    min_count=1,
    workers=4,
    sg=1  # Skip-gram
)

# Save model
model.save('word2vec.model')
```

### Step 2: Map to BEHRT Vocabulary

```python
from ehrsequencing.models.pretrained_embeddings import load_word2vec_embeddings

# Create vocabulary mapping
vocab_mapping = {
    0: '[PAD]',
    1: '[MASK]',
    2: '[CLS]',
    3: 'ICD9_250.00',  # Diabetes
    4: 'ICD9_401.9',   # Hypertension
    5: 'ICD9_272.4',   # Hyperlipidemia
    # ... more codes
}

# Load and map embeddings
embeddings = load_word2vec_embeddings(
    embedding_path='word2vec.model',
    vocab_mapping=vocab_mapping,
    embedding_dim=128
)
# Loaded Word2Vec embeddings: 987/1000 codes found

# Note: Missing codes get random initialization
```

### Step 3: Initialize BEHRT

```python
from ehrsequencing.models.pretrained_embeddings import initialize_embedding_layer

# Create BEHRT model
config = BEHRTConfig.large(vocab_size=1000)
config.embedding_dim = 128  # Match Word2Vec dimension
model = BEHRTForMLM(config)

# Initialize embeddings
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    pretrained_embeddings=embeddings,
    freeze=False
)

# Apply LoRA and train...
```

### Word2Vec vs. Med2Vec

| Feature | Word2Vec | Med2Vec |
|---------|----------|---------|
| Context | Within visit | Cross-visit sequences |
| Semantics | Co-occurrence | Disease progression |
| Training | Unsupervised | Semi-supervised |
| Best for | Static patterns | Temporal patterns |

**Recommendation:** Use Med2Vec for BEHRT (better captures temporal patterns)

---

## Custom Embeddings

### Scenario 1: Transfer from Another BEHRT Model

```python
# Load pretrained BEHRT model
checkpoint = torch.load('pretrained_behrt.pt')
pretrained_model = BEHRTForMLM(config)
pretrained_model.load_state_dict(checkpoint)

# Extract code embeddings
embeddings = pretrained_model.behrt.embeddings.code_embedding.weight.data

# Save embeddings
from ehrsequencing.models.pretrained_embeddings import save_embeddings
save_embeddings(
    embeddings,
    'behrt_embeddings.pt',
    metadata={'vocab_size': 1000, 'embedding_dim': 256, 'source': 'BEHRT-MLM'}
)

# Load into new model
new_embeddings, metadata = load_embeddings('behrt_embeddings.pt')
new_model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    new_model.behrt.embeddings.code_embedding,
    new_embeddings,
    freeze=False
)
```

### Scenario 2: Use External Embeddings (CUI2Vec, etc.)

```python
import numpy as np

# Load external embeddings (e.g., CUI2Vec)
external_embeddings = np.load('cui2vec_embeddings.npy')  # [N, D]

# Map to BEHRT vocabulary
vocab_size = 1000
embedding_dim = 256
embeddings = torch.randn(vocab_size, embedding_dim) * 0.01  # Random init

# Fill in mapped embeddings
for behrt_id, external_id in id_mapping.items():
    if external_id < len(external_embeddings):
        embeddings[behrt_id] = torch.from_numpy(external_embeddings[external_id])

# Initialize BEHRT
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    embeddings,
    freeze=False
)
```

### Scenario 3: Combine Multiple Embedding Sources

```python
# Load multiple sources
med2vec_emb = load_med2vec_embeddings('med2vec.pt', 1000, 128)
word2vec_emb = load_word2vec_embeddings('word2vec.model', vocab_mapping, 128)

# Combine (e.g., average)
combined_emb = (med2vec_emb + word2vec_emb) / 2

# Or concatenate (need to adjust embedding_dim)
combined_emb = torch.cat([med2vec_emb, word2vec_emb], dim=1)  # [1000, 256]

# Initialize BEHRT
config.embedding_dim = combined_emb.shape[1]
model = BEHRTForMLM(config)
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    combined_emb,
    freeze=False
)
```

---

## Embedding Analysis

### Visualizing Embeddings

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, labels, title='Embeddings'):
    """Visualize embeddings in 2D using PCA."""
    # Reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings.cpu().numpy())
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
    
    # Annotate some points
    for i, label in enumerate(labels[:20]):  # First 20
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

# Usage
embeddings = model.behrt.embeddings.code_embedding.weight.data
labels = [f'Code_{i}' for i in range(len(embeddings))]
visualize_embeddings(embeddings, labels, 'BEHRT Code Embeddings')
```

### Computing Embedding Similarity

```python
import torch.nn.functional as F

def find_similar_codes(code_id, embeddings, top_k=10):
    """Find most similar codes to a given code."""
    # Get embedding for code
    query_emb = embeddings[code_id].unsqueeze(0)  # [1, D]
    
    # Compute cosine similarity with all codes
    similarities = F.cosine_similarity(query_emb, embeddings)  # [vocab_size]
    
    # Get top-k most similar
    top_k_values, top_k_indices = similarities.topk(top_k + 1)  # +1 to exclude self
    
    # Exclude self
    top_k_values = top_k_values[1:]
    top_k_indices = top_k_indices[1:]
    
    return top_k_indices, top_k_values

# Usage
code_id = 250  # Diabetes code
similar_ids, similarities = find_similar_codes(
    code_id,
    model.behrt.embeddings.code_embedding.weight.data,
    top_k=10
)

print(f"Top 10 codes similar to Code {code_id}:")
for i, (sim_id, sim_score) in enumerate(zip(similar_ids, similarities)):
    print(f"{i+1}. Code {sim_id.item()} (similarity: {sim_score.item():.4f})")
```

### Embedding Statistics

```python
from ehrsequencing.models.pretrained_embeddings import (
    get_embedding_statistics,
    print_embedding_statistics
)

# Get statistics
embeddings = model.behrt.embeddings.code_embedding.weight.data
stats = get_embedding_statistics(embeddings)

print_embedding_statistics(embeddings, "BEHRT Code Embeddings")
# 📊 BEHRT Code Embeddings Statistics:
#    Shape: torch.Size([1000, 256])
#    Value range: [-0.1234, 0.1234]
#    Mean: 0.0012, Std: 0.0234
#    Norm mean: 0.5678, Norm std: 0.0123
```

### Comparing Embeddings Before/After Training

```python
# Save initial embeddings
initial_emb = model.behrt.embeddings.code_embedding.weight.data.clone()

# Train model...

# Get final embeddings
final_emb = model.behrt.embeddings.code_embedding.weight.data

# Compute change
change = torch.norm(final_emb - initial_emb, dim=1)  # [vocab_size]
mean_change = change.mean().item()
max_change = change.max().item()

print(f"Embedding change:")
print(f"  Mean: {mean_change:.4f}")
print(f"  Max: {max_change:.4f}")

# Codes that changed most
top_changed_indices = change.topk(10).indices
print(f"\nTop 10 codes with largest embedding change:")
for i, idx in enumerate(top_changed_indices):
    print(f"{i+1}. Code {idx.item()}: change = {change[idx].item():.4f}")
```

---

## Complete Workflows

### Workflow 1: Train BEHRT with Med2Vec Embeddings

**Full end-to-end script:**

```python
"""
Train BEHRT with Med2Vec pretrained embeddings.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from ehrsequencing.models.behrt import BEHRTConfig, BEHRTForMLM
from ehrsequencing.models.lora import apply_lora_to_behrt
from ehrsequencing.models.pretrained_embeddings import (
    load_med2vec_embeddings,
    initialize_embedding_layer
)
from ehrsequencing.data import generate_demo_dataset
from ehrsequencing.utils.experiment_tracker import ExperimentTracker

# Configuration
VOCAB_SIZE = 1000
EMBEDDING_DIM = 256
NUM_PATIENTS = 5000
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-4
DEVICE = 'cuda'

# 1. Load Med2Vec embeddings
print("Loading Med2Vec embeddings...")
embeddings = load_med2vec_embeddings(
    'path/to/med2vec_embeddings.pt',
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM
)

# 2. Create BEHRT model
print("Creating BEHRT model...")
config = BEHRTConfig.large(vocab_size=VOCAB_SIZE)
config.embedding_dim = EMBEDDING_DIM
model = BEHRTForMLM(config).to(DEVICE)

# 3. Initialize with Med2Vec embeddings
print("Initializing embeddings...")
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    pretrained_embeddings=embeddings,
    freeze=False  # Allow fine-tuning
)

# 4. Apply LoRA
print("Applying LoRA...")
model = apply_lora_to_behrt(
    model,
    rank=16,
    train_embeddings=True,
    train_head=True
)

# 5. Generate data
print("Generating data...")
codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_demo_dataset(
    num_patients=NUM_PATIENTS,
    vocab_size=VOCAB_SIZE,
    max_seq_length=200,
    seed=42
)

# 6. Create dataloaders
train_size = int(0.8 * NUM_PATIENTS)
train_dataset = TensorDataset(
    masked_codes[:train_size],
    ages[:train_size],
    visit_ids[:train_size],
    attention_mask[:train_size],
    labels[:train_size]
)
val_dataset = TensorDataset(
    masked_codes[train_size:],
    ages[train_size:],
    visit_ids[train_size:],
    attention_mask[train_size:],
    labels[train_size:]
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 7. Setup optimizer and tracker
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
tracker = ExperimentTracker('behrt_med2vec', output_dir='experiments')

# 8. Training loop
print("Starting training...")
for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss = 0
    for batch in train_loader:
        masked_codes, ages, visit_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
        
        optimizer.zero_grad()
        logits, loss = model(masked_codes, ages, visit_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    train_loss = total_loss / len(train_loader)
    
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            masked_codes, ages, visit_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
            logits, loss = model(masked_codes, ages, visit_ids, attention_mask, labels)
            val_loss += loss.item()
    
    val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Track metrics
    tracker.log_metrics(epoch, {'train_loss': train_loss, 'val_loss': val_loss})
    tracker.save_lora_checkpoint(model, epoch, {'val_loss': val_loss}, is_best=(val_loss < best_loss))

# 9. Save final model
tracker.save_summary()
print("Training complete!")
```

### Workflow 2: Compare Random vs. Med2Vec Initialization

```python
"""
Compare BEHRT training with random vs. Med2Vec initialization.
"""
import torch
from ehrsequencing.models.behrt import BEHRTConfig, BEHRTForMLM
from ehrsequencing.models.pretrained_embeddings import (
    load_med2vec_embeddings,
    initialize_embedding_layer
)

def train_and_compare():
    # Model 1: Random initialization
    print("Training Model 1: Random initialization")
    config = BEHRTConfig.large(vocab_size=1000)
    model_random = BEHRTForMLM(config)
    # Train...
    results_random = train_model(model_random, name='random')
    
    # Model 2: Med2Vec initialization
    print("\nTraining Model 2: Med2Vec initialization")
    model_med2vec = BEHRTForMLM(config)
    embeddings = load_med2vec_embeddings('med2vec.pt', 1000, 256)
    model_med2vec.behrt.embeddings.code_embedding = initialize_embedding_layer(
        model_med2vec.behrt.embeddings.code_embedding,
        embeddings,
        freeze=False
    )
    # Train...
    results_med2vec = train_model(model_med2vec, name='med2vec')
    
    # Compare results
    print("\n" + "="*80)
    print("Comparison Results")
    print("="*80)
    print(f"{'Metric':<30} {'Random':<15} {'Med2Vec':<15} {'Improvement':<15}")
    print("-"*80)
    
    metrics = ['epochs_to_converge', 'final_accuracy', 'final_loss', 'training_time']
    for metric in metrics:
        random_val = results_random[metric]
        med2vec_val = results_med2vec[metric]
        if metric == 'training_time':
            improvement = f"{(random_val - med2vec_val) / random_val * 100:.1f}% faster"
        elif metric == 'epochs_to_converge':
            improvement = f"{random_val - med2vec_val:.0f} fewer epochs"
        else:
            improvement = f"{(med2vec_val - random_val) / random_val * 100:.1f}% better"
        
        print(f"{metric:<30} {random_val:<15.4f} {med2vec_val:<15.4f} {improvement:<15}")

# Run comparison
train_and_compare()
```

---

## Best Practices

### 1. Match Embedding Dimensions

✅ **Do:**
```python
# Med2Vec embedding_dim = 256
config = BEHRTConfig.large(vocab_size=1000)
config.embedding_dim = 256  # Match Med2Vec
```

❌ **Don't:**
```python
# Mismatch will cause errors
config.embedding_dim = 128  # Med2Vec is 256!
```

### 2. Keep Embeddings Trainable (Usually)

✅ **Do:**
```python
initialize_embedding_layer(..., freeze=False)  # Allow fine-tuning
```

❌ **Don't:**
```python
initialize_embedding_layer(..., freeze=True)  # Too restrictive
```

**Exception:** Freeze only when:
- Very small dataset
- Pretrained embeddings are high quality
- Want to prevent catastrophic forgetting

### 3. Verify Embedding Quality

✅ **Do:**
```python
# Check embeddings before training
from ehrsequencing.models.pretrained_embeddings import print_embedding_statistics
embeddings = load_med2vec_embeddings(...)
print_embedding_statistics(embeddings, "Med2Vec")
# Verify: mean ≈ 0, std ≈ 0.02, no NaN/Inf
```

### 4. Save Trained Embeddings

✅ **Do:**
```python
# After training, save learned embeddings
from ehrsequencing.models.pretrained_embeddings import save_embeddings
embeddings = model.behrt.embeddings.code_embedding.weight.data
save_embeddings(embeddings, 'behrt_trained_embeddings.pt', metadata={...})
```

### 5. Use Appropriate Learning Rate

✅ **Do:**
```python
# With pretrained embeddings, can use slightly higher LR
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
```

❌ **Don't:**
```python
# Too high LR can destroy pretrained knowledge
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

### 6. Monitor Embedding Drift

✅ **Do:**
```python
# Save initial embeddings
initial_emb = model.behrt.embeddings.code_embedding.weight.data.clone()

# After training, compute drift
final_emb = model.behrt.embeddings.code_embedding.weight.data
drift = torch.norm(final_emb - initial_emb, dim=1).mean()
print(f"Average embedding drift: {drift:.4f}")
# Expected: 0.01-0.1 (too high = instability)
```

---

## Summary

### Key Takeaways

1. **Pretrained embeddings accelerate convergence** (50-100 epochs → 20-30 epochs)
2. **Med2Vec is preferred** for temporal EHR data
3. **Always keep embeddings trainable** when training from scratch
4. **Verify embedding quality** before training
5. **Monitor convergence** - should be faster with pretrained embeddings

### Quick Reference

**Load Med2Vec embeddings:**
```python
embeddings = load_med2vec_embeddings('med2vec.pt', vocab_size, embedding_dim)
```

**Initialize BEHRT:**
```python
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    embeddings,
    freeze=False
)
```

**Expected improvements:**
- Convergence: 50% faster (50 → 25 epochs)
- Performance: 5-10% better accuracy
- Training time: 30-50% reduction

---

**Related Documentation:**
- `01_behrt_model_design.md` - Model architecture details
- `02_training_guide.md` - Training best practices
- `examples/pretrain_finetune/PRETRAINED_EMBEDDINGS_GUIDE.md` - Additional examples
