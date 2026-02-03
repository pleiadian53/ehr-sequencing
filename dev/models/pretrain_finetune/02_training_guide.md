# BEHRT Training Guide

**Last Updated:** 2026-02-02  
**Author:** Private development notes

## Overview

This guide provides practical instructions for training BEHRT models, including command-line usage, hyperparameter tuning, best practices, and troubleshooting.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Script Overview](#training-script-overview)
3. [Command-Line Usage](#command-line-usage)
4. [Data Generation](#data-generation)
5. [Hyperparameter Guide](#hyperparameter-guide)
6. [Training Strategies](#training-strategies)
7. [Monitoring and Debugging](#monitoring-and-debugging)
8. [Best Practices](#best-practices)
9. [Common Issues](#common-issues)

---

## Quick Start

### Basic Training (Auto-Detect Resources)

The simplest way to train BEHRT:

```bash
cd /Users/pleiadian53/work/ehr-sequencing

# Auto-detect hardware and use demo data (recommended for first run)
python examples/pretrain_finetune/train_behrt_demo.py
```

**What happens:**
- Detects your hardware (CPU/GPU/VRAM)
- Selects appropriate model size (small/medium/large)
- Sets optimal batch size and epochs
- Uses high-signal demo data (70%+ accuracy expected)
- Saves results to `experiments/behrt_<size>_mlm_lora<rank>/`

### Expected Output

```
Using device: cuda

================================================================================
BEHRT Pre-training Demo: behrt_large_mlm_lora16
================================================================================

Detected GPU: NVIDIA A40 (48.0 GB VRAM)
Detected hardware: A40
Recommended model size: large
Recommended batch size: 128
Recommended epochs: 100
☁️  Large model (for A40 cloud GPU)

🔧 Applying LoRA (rank=16)...
Applied LoRA to encoder.layers.0.self_attn.in_proj_weight
Applied LoRA to encoder.layers.0.self_attn.out_proj
[... more layers ...]

📊 Model Parameters:
   Total: 26,358,784
   Trainable: 13,179,392 (50.0%)
   Frozen: 13,179,392
   LoRA: 131,072 (0.5%)
   Embeddings: 6,553,600/6,553,600 trainable
   Head: 6,553,600/6,553,600 trainable

🔬 Generating synthetic data...
Using HIGH-SIGNAL demo data with very strong patterns (70%+ accuracy expected)
💡 Tip: Use --realistic-data for more challenging, realistic patterns

🚀 Starting training...
   Train batches: 125
   Val batches: 32
   Early stopping patience: 10 epochs

Epoch 1/100 | Train Loss: 5.2341 Acc: 0.0234 | Val Loss: 4.8765 Acc: 0.0456 Top5: 0.1234 F1: 0.0123 🏆 | Patience: 0/10
Epoch 2/100 | Train Loss: 4.1234 Acc: 0.1234 | Val Loss: 3.5678 Acc: 0.2345 Top5: 0.4567 F1: 0.1234 🏆 | Patience: 0/10
[... training continues ...]
```

---

## Training Script Overview

**Location:** `examples/pretrain_finetune/train_behrt_demo.py`

### Key Features

1. **Auto-Resource Detection**: Automatically configures parameters based on your hardware
2. **Multiple Model Sizes**: small/medium/large configs
3. **LoRA Support**: Efficient fine-tuning by default
4. **Comprehensive Metrics**: Accuracy, Top-5, F1, Precision, Recall, Perplexity
5. **Experiment Tracking**: Full logging with plots and checkpoints
6. **Early Stopping**: Prevents overfitting
7. **Flexible Data**: Demo (high-signal) or realistic synthetic data

### Script Structure

```python
# 1. Parse arguments and auto-detect resources
args = parser.parse_args()
if args.auto_resources:
    recommended_config = get_recommended_config(task='demo')
    # Fill in unspecified parameters

# 2. Create model
config = BEHRTConfig.large(vocab_size=args.vocab_size)
model = BEHRTForMLM(config).to(device)

# 3. Apply LoRA
if args.use_lora:
    model = apply_lora_to_behrt(model, rank=args.lora_rank)

# 4. Generate synthetic data
codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_demo_dataset(...)

# 5. Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# 6. Setup optimizer and tracking
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
tracker = ExperimentTracker(args.experiment_name)

# 7. Training loop
for epoch in range(args.epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
    val_metrics = validate(model, val_loader, device, args.vocab_size)
    
    tracker.log_metrics(epoch, {
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_loss': val_metrics['loss'],
        'val_accuracy': val_metrics['accuracy'],
        'val_top_5_accuracy': val_metrics['top_5_accuracy'],
        'val_macro_f1': val_metrics['macro_f1'],
        'val_weighted_f1': val_metrics['weighted_f1'],
        'val_perplexity': val_metrics['perplexity']
    })
    
    # Save checkpoints
    if args.use_lora:
        tracker.save_lora_checkpoint(model, epoch, metrics, is_best=is_best)
    else:
        tracker.save_checkpoint(model, optimizer, epoch, metrics, is_best=is_best)
    
    # Early stopping
    if patience_counter >= args.early_stopping_patience:
        break

# 8. Generate plots and save summary
tracker.plot_training_curves()
tracker.save_summary()
```

---

## Command-Line Usage

### Basic Usage Patterns

#### 1. Auto-Detect Everything (Recommended)

```bash
# Uses demo data by default
python examples/pretrain_finetune/train_behrt_demo.py
```

#### 2. Use Realistic Data

```bash
# More challenging patterns (30-60% accuracy)
python examples/pretrain_finetune/train_behrt_demo.py --realistic-data
```

#### 3. Override Specific Parameters

```bash
# Auto-detect hardware, but set specific batch size
python examples/pretrain_finetune/train_behrt_demo.py \
    --batch-size 64 \
    --epochs 50
```

#### 4. Force Specific Model Size

```bash
# Force large model regardless of hardware
python examples/pretrain_finetune/train_behrt_demo.py \
    --model-size large \
    --batch-size 32  # Lower batch size if needed
```

#### 5. Disable Auto-Detection

```bash
# Use fixed defaults (not recommended)
python examples/pretrain_finetune/train_behrt_demo.py \
    --no-auto-resources \
    --model-size large \
    --batch-size 128 \
    --num-patients 5000 \
    --epochs 100 \
    --lora-rank 16
```

#### 6. Train Without LoRA

```bash
# Train full model (uses more memory)
python examples/pretrain_finetune/train_behrt_demo.py \
    --no-lora
```

#### 7. Custom Experiment Name

```bash
python examples/pretrain_finetune/train_behrt_demo.py \
    --experiment-name my_behrt_experiment \
    --output-dir my_experiments
```

### All Available Arguments

```bash
# Resource management
--auto-resources          # Enable auto-detection (default: True)
--no-auto-resources       # Disable auto-detection

# Model configuration
--model-size {small,medium,large}  # Model size
--use-lora                # Enable LoRA (default: True)
--no-lora                 # Disable LoRA
--lora-rank INT           # LoRA rank (default: auto-detected)

# Training parameters
--num-patients INT        # Number of synthetic patients
--vocab-size INT          # Vocabulary size (default: 1000)
--epochs INT              # Training epochs
--batch-size INT          # Batch size
--lr FLOAT                # Learning rate (default: 1e-4)
--weight-decay FLOAT      # Weight decay (default: 0.01)
--dropout FLOAT           # Dropout (default: 0.2)
--early-stopping-patience INT  # Early stopping patience (default: 10)

# Data options
--realistic-data          # Use realistic synthetic data
--demo-data               # Use high-signal demo data (default)

# Output
--experiment-name STR     # Experiment name
--output-dir STR          # Output directory (default: 'experiments')
```

### Example Commands for Different Scenarios

#### Local Development (M1 MacBook)

```bash
# Small model for M1 MacBook Pro 16GB
python examples/pretrain_finetune/train_behrt_demo.py \
    --model-size small \
    --batch-size 16 \
    --num-patients 500 \
    --epochs 20
```

#### Local Workstation (RTX 3090)

```bash
# Medium model for local GPU
python examples/pretrain_finetune/train_behrt_demo.py \
    --model-size medium \
    --batch-size 64 \
    --num-patients 2000 \
    --epochs 50
```

#### Cloud GPU (A40/A100)

```bash
# Large model for cloud GPU
python examples/pretrain_finetune/train_behrt_demo.py \
    --model-size large \
    --batch-size 128 \
    --num-patients 10000 \
    --epochs 100 \
    --realistic-data
```

#### Quick Test Run

```bash
# Fast test with small model and data
python examples/pretrain_finetune/train_behrt_demo.py \
    --model-size small \
    --batch-size 32 \
    --num-patients 100 \
    --epochs 5
```

---

## Data Generation

### Two Data Generation Modes

#### 1. Demo Data (Default)

**High-signal patterns for compelling demos:**

```python
from ehrsequencing.data import generate_demo_dataset

codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_demo_dataset(
    num_patients=5000,
    vocab_size=1000,
    max_seq_length=200,
    seed=42
)
```

**Characteristics:**
- Very strong patterns (70-85% accuracy)
- Clear temporal dependencies
- Strong disease progressions
- Good for demos and proof-of-concept

**Use when:**
- First-time training
- Demonstrating the model
- Verifying implementation
- Quick validation

#### 2. Realistic Data

**More challenging, realistic patterns:**

```python
from ehrsequencing.data import generate_realistic_dataset

codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_realistic_dataset(
    num_patients=5000,
    vocab_size=1000,
    max_seq_length=200,
    seed=42
)
```

**Characteristics:**
- Realistic patterns (30-60% accuracy)
- More noise and variability
- Reflects real EHR complexity
- Better for research

**Use when:**
- Evaluating model capacity
- Research experiments
- Realistic benchmarking

### Data Statistics

After generation, the script prints statistics:

```
Dataset Statistics:
  Total patients: 5000
  Average sequence length: 127.3
  Vocabulary size: 1000
  Total visits: 15,432
  Average visits per patient: 3.1
  Masking rate: 15.0%
  Masked tokens: 97,543
```

### Understanding Masking

**MLM (Masked Language Modeling) Strategy:**

```python
# Original sequence
codes = [120, 450, 780, 230, 560]

# Masked sequence (15% random masking)
masked_codes = [120, [MASK], 780, [MASK], 560]

# Labels (-100 = not masked)
labels = [-100, 450, -100, 230, -100]
```

**Masking process:**
1. Select 15% of tokens randomly
2. Replace with `[MASK]` token (vocab_size - 1)
3. Model predicts original token
4. Loss computed only on masked positions

---

## Hyperparameter Guide

### Model Architecture Hyperparameters

#### Embedding Dimension

```python
config.embedding_dim = 256  # Dimension of embeddings
```

**Guidelines:**
- Small model: 64
- Medium model: 128
- Large model: 256
- Larger = more capacity, more memory

**Trade-offs:**
- Higher → better representations, more parameters
- Lower → faster training, less memory

#### Hidden Dimension

```python
config.hidden_dim = 512  # Transformer hidden dimension
```

**Guidelines:**
- Small model: 128
- Medium model: 256
- Large model: 512
- Should be divisible by num_heads

**Trade-offs:**
- Higher → more model capacity
- Lower → faster, less memory

#### Number of Layers

```python
config.num_layers = 6  # Transformer layers
```

**Guidelines:**
- Small model: 2
- Medium model: 4
- Large model: 6-12
- More layers = deeper reasoning

**Trade-offs:**
- More layers → better long-range dependencies
- Fewer layers → faster, less overfitting risk

#### Number of Attention Heads

```python
config.num_heads = 8  # Multi-head attention
```

**Guidelines:**
- Must divide hidden_dim evenly
- Typical: 4, 8, 12, 16
- More heads = more attention patterns

**Constraint:**
```python
assert hidden_dim % num_heads == 0
head_dim = hidden_dim // num_heads
```

### Training Hyperparameters

#### Learning Rate

```python
lr = 1e-4  # Default learning rate
```

**Guidelines:**
- Training from scratch: 1e-4 to 5e-4
- Fine-tuning with LoRA: 1e-4 to 1e-3
- Fine-tuning full model: 1e-5 to 5e-5

**Rule of thumb:**
- Start with 1e-4
- If loss plateaus early → increase (2e-4, 5e-4)
- If loss is unstable → decrease (5e-5, 1e-5)

**Scaling with batch size:**
- Linear scaling rule: `lr ∝ batch_size`
- If batch_size doubles → consider 1.5x lr

#### Batch Size

```python
batch_size = 128  # Samples per batch
```

**Guidelines by hardware:**

| Hardware | VRAM | Recommended Batch Size |
|----------|------|------------------------|
| CPU | - | 4-8 |
| M1 MacBook | 16GB | 16-32 |
| RTX 3090 | 24GB | 64-96 |
| A40 | 48GB | 128-256 |
| A100 | 80GB | 256-512 |

**Trade-offs:**
- Larger → more stable gradients, better GPU utilization
- Smaller → more updates per epoch, less memory

**If OOM (Out of Memory):**
1. Reduce batch size by 50%
2. Enable gradient accumulation
3. Reduce model size
4. Reduce sequence length

#### Weight Decay

```python
weight_decay = 0.01  # L2 regularization
```

**Guidelines:**
- Small models: 0.001
- Medium models: 0.01
- Large models: 0.01-0.1

**Purpose:**
- Prevents overfitting
- Regularizes large weights
- Improves generalization

#### Dropout

```python
dropout = 0.2  # Dropout probability
```

**Guidelines:**
- Small datasets: 0.2-0.3
- Large datasets: 0.1-0.2
- No dropout: 0.0 (not recommended)

**Where applied:**
- Attention dropout
- Feedforward dropout
- Embedding dropout

**Rule of thumb:**
- More data → less dropout
- Overfitting → increase dropout
- Underfitting → decrease dropout

#### Early Stopping Patience

```python
early_stopping_patience = 10  # Epochs without improvement
```

**Guidelines:**
- Quick experiments: 5 epochs
- Standard training: 10 epochs
- Patient training: 20 epochs

**When to stop:**
- Validation loss stops improving
- Model is overfitting
- Time/compute budget exhausted

### LoRA Hyperparameters

#### LoRA Rank

```python
lora_rank = 16  # Rank of low-rank decomposition
```

**Guidelines:**
- Small models: 4-8
- Medium models: 8-16
- Large models: 16-64

**Trade-offs:**
- Higher rank → more capacity, more parameters
- Lower rank → more efficient, faster

**Typical values:**
- 4: Very efficient, may underfit
- 8: Good balance for most tasks
- 16: More capacity, still efficient
- 32-64: High capacity, less efficient

#### LoRA Alpha

```python
lora_alpha = 16.0  # Scaling factor
```

**Guidelines:**
- Typically set equal to rank: `alpha = rank`
- Can be adjusted independently

**Effect:**
- Scaling factor = alpha / rank
- Higher alpha → larger LoRA contribution
- Lower alpha → smaller LoRA contribution

### Recommended Configurations

#### Configuration 1: Quick Test

```python
config = BEHRTConfig.small(vocab_size=1000)
model_size = 'small'
batch_size = 32
num_patients = 100
epochs = 5
lr = 1e-4
lora_rank = 8
```

**Purpose:** Quick validation, debugging

#### Configuration 2: Local Development

```python
config = BEHRTConfig.medium(vocab_size=1000)
model_size = 'medium'
batch_size = 64
num_patients = 2000
epochs = 50
lr = 1e-4
lora_rank = 16
dropout = 0.2
```

**Purpose:** Local GPU training, experiments

#### Configuration 3: Cloud Production

```python
config = BEHRTConfig.large(vocab_size=1000)
model_size = 'large'
batch_size = 128
num_patients = 10000
epochs = 100
lr = 1e-4
lora_rank = 16
dropout = 0.2
weight_decay = 0.01
```

**Purpose:** Full-scale training, best results

---

## Training Strategies

### Strategy 1: Standard Pre-training from Scratch

**Scenario:** No pretrained weights, training BEHRT from random initialization

```python
# 1. Create model
config = BEHRTConfig.large(vocab_size=1000)
model = BEHRTForMLM(config)

# 2. Apply LoRA
model = apply_lora_to_behrt(
    model,
    rank=16,
    train_embeddings=True,  # MUST be True
    train_head=True          # MUST be True
)

# 3. Train with MLM objective
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# 4. Use longer training
epochs = 100
patience = 20
```

**Expected results:**
- Demo data: 70-85% accuracy
- Realistic data: 30-60% accuracy
- Convergence: 20-50 epochs

### Strategy 2: Pre-training with Med2Vec Embeddings

**Scenario:** Use pretrained embeddings from Med2Vec, train transformer from scratch

```python
# 1. Load pretrained embeddings
from ehrsequencing.models.pretrained_embeddings import (
    load_med2vec_embeddings,
    initialize_embedding_layer
)

embeddings = load_med2vec_embeddings('med2vec_embeddings.pt', 1000, 256)

# 2. Create model and initialize embeddings
config = BEHRTConfig.large(vocab_size=1000)
model = BEHRTForMLM(config)
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    embeddings,
    freeze=False  # Allow fine-tuning
)

# 3. Apply LoRA
model = apply_lora_to_behrt(
    model,
    rank=16,
    train_embeddings=True,  # Keep embeddings trainable
    train_head=True
)

# 4. Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

**Benefits:**
- Faster convergence (10-30 epochs)
- Better initial representations
- Improved final performance

### Strategy 3: Fine-tuning Pretrained BEHRT

**Scenario:** Fine-tune a pretrained BEHRT model on downstream task

```python
# 1. Load pretrained model
checkpoint = torch.load('pretrained_behrt_mlm.pt')
config = BEHRTConfig.large(vocab_size=1000)
model = BEHRTForMLM(config)
model.load_state_dict(checkpoint)

# 2. Apply LoRA for efficient fine-tuning
model = apply_lora_to_behrt(
    model,
    rank=8,                  # Lower rank for fine-tuning
    train_embeddings=False,  # Freeze pretrained embeddings
    train_head=True          # Adapt head to new task
)

# 3. Use lower learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

# 4. Shorter training
epochs = 20
patience = 5
```

**Guidelines:**
- Lower learning rate than pre-training
- Shorter training (10-20 epochs)
- Smaller LoRA rank (4-8)

### Strategy 4: Two-Stage Pre-training

**Scenario:** Pre-train with MLM, then next visit prediction

```python
# Stage 1: MLM Pre-training
model_mlm = BEHRTForMLM(config)
model_mlm = apply_lora_to_behrt(model_mlm, rank=16)
# Train with MLM objective...
torch.save(model_mlm.behrt.state_dict(), 'behrt_mlm_pretrained.pt')

# Stage 2: Next Visit Prediction
model_nvp = BEHRTForNextVisitPrediction(config)
model_nvp.behrt.load_state_dict(torch.load('behrt_mlm_pretrained.pt'))
model_nvp = apply_lora_to_behrt(
    model_nvp,
    rank=8,
    train_embeddings=False,  # Freeze embeddings from Stage 1
    train_head=True          # New head for NVP
)
# Train with NVP objective...
```

**Benefits:**
- Multi-objective learning
- Better representations
- Improved downstream performance

### Strategy 5: Progressive Training

**Scenario:** Start with small model, progressively increase size

```python
# Phase 1: Small model (quick convergence)
config_small = BEHRTConfig.small(vocab_size=1000)
model_small = BEHRTForMLM(config_small)
# Train for 20 epochs...

# Phase 2: Transfer to medium model
config_medium = BEHRTConfig.medium(vocab_size=1000)
model_medium = BEHRTForMLM(config_medium)
# Transfer embeddings from small model
model_medium.behrt.embeddings.code_embedding.weight.data[:, :64] = \
    model_small.behrt.embeddings.code_embedding.weight.data
# Train for 50 epochs...

# Phase 3: Transfer to large model
# Similar process...
```

**Use when:**
- Limited initial compute budget
- Iterative development
- Exploring model capacity

---

## Monitoring and Debugging

### Key Metrics to Watch

#### 1. Training Loss

**What it tells you:**
- Model's ability to fit training data
- Should decrease steadily

**Expected behavior:**
- Initial: 5-7 (random predictions)
- After 10 epochs: 2-4
- After 50 epochs: 0.5-2
- Final: 0.1-1

**Warning signs:**
- Not decreasing → learning rate too low, model too small
- Erratic → learning rate too high, batch size too small
- Plateau early → model capacity issue, data issue

#### 2. Validation Loss

**What it tells you:**
- Model's generalization ability
- Overfitting indicator

**Expected behavior:**
- Should track training loss initially
- May plateau or increase (overfitting)
- Gap from training loss indicates overfitting

**Warning signs:**
- Much higher than training → overfitting
- Not decreasing → model not learning
- Increasing → severe overfitting

#### 3. Accuracy

**What it tells you:**
- Percentage of correctly predicted masked tokens

**Expected values:**
- Random baseline: 0.1% (1/vocab_size)
- Demo data: 70-85%
- Realistic data: 30-60%

**Interpretation:**
- Below 10% → model not learning
- 30-60% → good for realistic data
- 70%+ → excellent for demo data

#### 4. Top-5 Accuracy

**What it tells you:**
- Is correct code in top 5 predictions?
- More forgiving metric

**Expected values:**
- Should be 2-3x higher than accuracy
- Demo data: 85-95%
- Realistic data: 60-80%

#### 5. Macro F1

**What it tells you:**
- Performance averaged across all codes
- Treats rare codes equally

**Use when:**
- Vocabulary has rare codes
- Care about minority class performance

**Warning signs:**
- Much lower than accuracy → model ignoring rare codes

#### 6. Perplexity

**What it tells you:**
- Exp(cross-entropy loss)
- Lower = better

**Expected values:**
- Initial: 100-1000
- Final: 2-10 (demo), 5-20 (realistic)

**Formula:**
```python
perplexity = torch.exp(loss)
```

### Visualization

The script automatically generates plots:

```
experiments/behrt_large_mlm_lora16/plots/
├── loss_curve.png          # Train/val loss over time
├── accuracy_curve.png      # Accuracy over time
├── top_5_accuracy_curve.png
├── macro_f1_curve.png
├── perplexity_curve.png
└── weighted_f1_curve.png
```

**How to interpret:**

1. **Loss curves should decrease**
   - Train loss decreases faster than val loss
   - Gap indicates overfitting

2. **Accuracy should increase**
   - Plateaus indicate convergence
   - Sudden drops indicate instability

3. **F1 tracks accuracy**
   - Macro F1 < weighted F1 indicates rare code issues

### Real-Time Monitoring

**During training:**

```
Epoch 1/100 | Train Loss: 5.2341 Acc: 0.0234 | Val Loss: 4.8765 Acc: 0.0456 Top5: 0.1234 F1: 0.0123 🏆 | Patience: 0/10
```

**What to look for:**

1. **🏆 Trophy icon**: Significant improvement (>0.5%)
2. **✓ Check mark**: Small improvement
3. **Patience counter**: How many epochs without improvement

**Good signs:**
- Regular trophies in first 10-20 epochs
- Steady accuracy increase
- Val loss tracking train loss

**Bad signs:**
- No trophies after 20+ epochs
- Erratic metrics
- Val loss >> train loss

### Debugging Checklist

#### Issue: Model not learning (accuracy < 5%)

**Check:**
1. Data generation working? Print some samples
2. Masking applied correctly? Check `labels != -100`
3. Loss function configured? `ignore_index=-100`
4. Learning rate too low? Try 5e-4
5. Model too small? Try larger model

#### Issue: Training loss decreasing, val loss increasing

**Diagnosis:** Overfitting

**Solutions:**
1. Increase dropout (0.2 → 0.3)
2. Increase weight decay (0.01 → 0.1)
3. Reduce model size
4. More training data
5. Early stopping working?

#### Issue: Loss is NaN

**Diagnosis:** Numerical instability

**Solutions:**
1. Lower learning rate (1e-4 → 1e-5)
2. Check for exploding gradients (add grad clipping)
3. Check data for invalid values
4. Reduce batch size

```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Issue: Out of memory

**Solutions:**
1. Reduce batch size
2. Reduce sequence length
3. Use smaller model
4. Enable gradient checkpointing
5. Use mixed precision training

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    logits, loss = model(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Best Practices

### 1. Start Simple

✅ **Do:**
- Start with small model and small data
- Verify training works
- Then scale up

❌ **Don't:**
- Start with largest model
- Use full dataset immediately
- Add complexity before verifying basics

### 2. Use Auto-Detection

✅ **Do:**
- Let script detect your hardware
- Trust recommended configurations
- Override only when needed

❌ **Don't:**
- Manually specify all parameters
- Ignore OOM warnings
- Use configs from different hardware

### 3. Monitor Early

✅ **Do:**
- Check first 5 epochs carefully
- Ensure loss is decreasing
- Verify accuracy improving

❌ **Don't:**
- Wait until end to check results
- Ignore warning signs
- Let bad runs continue

### 4. Save Checkpoints

✅ **Do:**
- Use experiment tracking
- Save best model
- Keep training history

❌ **Don't:**
- Skip checkpointing
- Overwrite best model
- Lose training progress

### 5. Use LoRA by Default

✅ **Do:**
- Enable LoRA for efficiency
- Use rank 8-16 typically
- Keep embeddings/head trainable when training from scratch

❌ **Don't:**
- Train full model unnecessarily
- Use LoRA rank > 64
- Freeze embeddings when training from scratch

### 6. Experiment Systematically

✅ **Do:**
- Change one thing at a time
- Keep detailed notes
- Compare results objectively

❌ **Don't:**
- Change multiple hyperparameters
- Rely on memory
- Cherry-pick results

### 7. Validate Thoroughly

✅ **Do:**
- Use separate validation set
- Check multiple metrics
- Examine failure cases

❌ **Don't:**
- Evaluate on training data
- Rely on single metric
- Ignore model weaknesses

---

## Common Issues

### Issue 1: "CUDA out of memory"

**Cause:** Batch size or model too large for GPU

**Solutions:**
```bash
# Reduce batch size
python train_behrt_demo.py --batch-size 32

# Use smaller model
python train_behrt_demo.py --model-size medium

# Reduce sequence length in code generation
```

### Issue 2: "Validation accuracy not improving"

**Cause:** Model overfitting or too simple

**Solutions:**
```bash
# Increase dropout
python train_behrt_demo.py --dropout 0.3

# More training data
python train_behrt_demo.py --num-patients 10000

# Use realistic data (if using demo)
python train_behrt_demo.py --realistic-data
```

### Issue 3: "Training too slow"

**Cause:** Model too large, batch size too small, CPU training

**Solutions:**
```bash
# Increase batch size (if memory allows)
python train_behrt_demo.py --batch-size 256

# Use smaller model
python train_behrt_demo.py --model-size medium

# Reduce sequence length
# Reduce number of patients for testing
python train_behrt_demo.py --num-patients 1000 --epochs 20
```

### Issue 4: "Loss is NaN"

**Cause:** Numerical instability, learning rate too high

**Solutions:**
```bash
# Lower learning rate
python train_behrt_demo.py --lr 1e-5

# Reduce dropout
python train_behrt_demo.py --dropout 0.1

# Check data for invalid values
```

### Issue 5: "Model predicting same token always"

**Cause:** Model collapsed to majority class

**Solutions:**
- Check data generation (is it diverse?)
- Increase model capacity
- Lower learning rate
- Check loss function (ignore_index=-100?)

### Issue 6: "Experiment directory not found"

**Cause:** Output directory doesn't exist

**Solutions:**
```bash
# Create output directory
mkdir -p experiments

# Or specify different directory
python train_behrt_demo.py --output-dir my_experiments
```

---

## Summary

### Quick Reference

**Basic training:**
```bash
python examples/pretrain_finetune/train_behrt_demo.py
```

**Custom configuration:**
```bash
python examples/pretrain_finetune/train_behrt_demo.py \
    --model-size large \
    --batch-size 128 \
    --num-patients 10000 \
    --epochs 100 \
    --lora-rank 16 \
    --realistic-data
```

**Key hyperparameters:**
- Learning rate: 1e-4 (default)
- Batch size: Auto-detected (32-256)
- LoRA rank: 8-16
- Dropout: 0.2
- Weight decay: 0.01

**Expected results:**
- Demo data: 70-85% accuracy
- Realistic data: 30-60% accuracy
- Convergence: 20-50 epochs

**Output location:**
```
experiments/behrt_<size>_mlm_lora<rank>/
├── checkpoints/best_lora.pt
├── plots/
├── SUMMARY.txt
└── ...
```

---

**Next:** See `03_pretrained_embeddings_workflow.md` for detailed instructions on using Med2Vec and other pretrained embeddings.
