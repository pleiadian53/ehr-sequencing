# BEHRT Training Troubleshooting Guide

This guide documents common failure modes, diagnostic techniques, and solutions for training BEHRT models on EHR data.

---

## Table of Contents

1. [Overfitting: The Most Common Failure Mode](#overfitting-the-most-common-failure-mode)
2. [Diagnostic Checklist](#diagnostic-checklist)
3. [Hyperparameter Tuning Guide](#hyperparameter-tuning-guide)
4. [Other Common Issues](#other-common-issues)
5. [Performance Expectations](#performance-expectations)

---

## Overfitting: The Most Common Failure Mode

### Real-World Example (A40 Pod Training)

**Observed Behavior:**
```
Epoch 1/100  | Train Loss: 6.9651 Acc: 0.0010 | Val Loss: 6.9181 Acc: 0.0012 🏆
Epoch 2/100  | Train Loss: 6.9097 Acc: 0.0011 | Val Loss: 6.9157 Acc: 0.0010 🏆
...
Epoch 10/100 | Train Loss: 6.7145 Acc: 0.0058 | Val Loss: 7.0427 Acc: 0.0011
Epoch 20/100 | Train Loss: 6.2970 Acc: 0.0215 | Val Loss: 7.1356 Acc: 0.0010
Epoch 21/100 | Train Loss: 6.2555 Acc: 0.0232 | Val Loss: 7.1442 Acc: 0.0012
```

**Key Symptoms:**
- ✅ Training loss decreasing: 6.97 → 6.26
- ❌ Validation loss **increasing**: 6.92 → 7.14
- ✅ Training accuracy improving: 0.10% → 2.32%
- ❌ Validation accuracy **stagnant**: ~0.10%

**Diagnosis: Severe Overfitting**

The model is memorizing the training data instead of learning generalizable patterns.

---

### Root Causes Identified

#### 1. **Too Many Trainable Parameters (92.3%!)**

**What Happened:**
```
📊 Model Parameters:
   Total: 20,366,312
   Trainable: 18,790,376 (92.3%)  ❌ TOO HIGH!
   Frozen: 1,575,936
   LoRA: 98,304 (0.5%)
```

**Problem:**
- LoRA is supposed to freeze most weights and only train low-rank adapters
- Only 98K parameters should be trainable (0.5%)
- But 18.8M parameters (92.3%) were trainable!
- This defeats the purpose of LoRA

**Why It Happened:**
- LoRA was only applied to attention output projections
- All other layers (embeddings, feedforward, MLM head) remained trainable
- With so many trainable parameters, the model easily memorizes training data

**Expected Behavior:**
```
📊 Model Parameters (with proper LoRA):
   Total: 20,366,312
   Trainable: 500,000 (2-5%)  ✅ GOOD
   Frozen: 19,866,312
   LoRA: 98,304 (0.5%)
```

#### 2. **Insufficient Regularization**

**Missing Components:**
- ❌ No weight decay (L2 regularization)
- ❌ Low dropout (0.1 default, too weak for large models)
- ❌ No early stopping (training continued despite worsening validation)

**Impact:**
- Weights grew unconstrained
- Model complexity not penalized
- Training continued long after optimal point

#### 3. **Embeddings and MLM Head Were Frozen (Critical Bug!)**

**What Happened (Second Iteration):**
After fixing the 92% trainable issue, the model STILL wasn't learning:
```
Epoch 1  | Train Loss: 7.0 | Val Loss: 6.9
Epoch 10 | Train Loss: 6.8 | Val Loss: 7.0  ❌ Barely moved!
```

**Root Cause:**
- `freeze_base=True` froze ALL parameters including embeddings and MLM head
- Embeddings were **randomly initialized** and frozen at random values
- MLM head was frozen at random weights
- Only LoRA adapters (98K params) were trainable
- Model couldn't learn because:
  - Random frozen embeddings → no meaningful representations
  - Frozen MLM head → can't map to vocabulary
  - Loss ≈ ln(1000) = 6.9 (random guessing)

**The Fix:**
Added `train_embeddings=True` and `train_head=True` parameters to `apply_lora_to_behrt()`:
```python
model = apply_lora_to_behrt(
    model,
    rank=16,
    freeze_base=True,        # Freeze transformer encoder
    train_embeddings=True,   # ✅ Unfreeze embeddings (must learn from scratch)
    train_head=True          # ✅ Unfreeze MLM head (must learn to predict)
)
```

**Results After Fix:**

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| Trainable params | 874,984 (4.3%) | 1,450,984 (7.1%) |
| Embeddings trainable | 0 ❌ | 109,824 ✅ |
| MLM head trainable | 0 ❌ | 145,768 ✅ |
| Training loss (10 epochs) | 7.0 → 6.8 | 7.0 → 5.6 |
| Training accuracy | ~0.1% | ~5% |

**Key Lesson:**
When training from scratch, embeddings and task heads MUST be trainable. Only freeze them when fine-tuning a pre-trained model.

#### 4. **Synthetic Data Overfitting (Expected Behavior)**

**Problem:**
- Synthetic data has random patterns
- Model memorizes these random patterns
- Doesn't learn real medical code relationships

**Evidence:**
- Training accuracy improving (memorization working)
- Validation accuracy not improving (memorization doesn't generalize)

**Why This Is Expected:**
- Random codes have no learnable structure
- Model can memorize training data but can't generalize
- With real EHR data (e.g., diabetes codes co-occurring with insulin), the model would generalize

---

## Diagnostic Checklist

Use this checklist to diagnose training issues:

### 1. **Check Loss Curves**

**Healthy Training:**
```
Epoch 1  | Train: 7.00 | Val: 6.95
Epoch 5  | Train: 6.50 | Val: 6.48
Epoch 10 | Train: 6.20 | Val: 6.22
Epoch 15 | Train: 6.00 | Val: 6.05
```
- Both losses decreasing
- Val loss tracks train loss closely
- Small gap between train and val

**Overfitting:**
```
Epoch 1  | Train: 7.00 | Val: 6.95  ✅
Epoch 5  | Train: 6.50 | Val: 6.48  ✅
Epoch 10 | Train: 6.00 | Val: 6.55  ⚠️ Val increasing!
Epoch 15 | Train: 5.50 | Val: 6.80  ❌ Large gap!
```
- Train loss decreasing, val loss increasing
- Growing gap between train and val
- **Action: Stop training, add regularization**

**Underfitting:**
```
Epoch 1  | Train: 7.00 | Val: 6.95
Epoch 5  | Train: 6.90 | Val: 6.88
Epoch 10 | Train: 6.85 | Val: 6.82
Epoch 15 | Train: 6.80 | Val: 6.78
```
- Both losses high and decreasing slowly
- Small gap but poor performance
- **Action: Increase model capacity or training time**

### 2. **Check Parameter Counts**

**Run this diagnostic:**
```python
from ehrsequencing.models.lora import count_parameters

params = count_parameters(model)
print(f"Total: {params['total']:,}")
print(f"Trainable: {params['trainable']:,} ({params['trainable_percent']:.1f}%)")
print(f"LoRA: {params['lora']:,} ({params['lora_percent']:.1f}%)")
```

**Expected Ranges:**

| Model Size | Total Params | Trainable % | LoRA % | Status |
|------------|-------------|-------------|--------|--------|
| Small | 500K | 2-10% | 0.5-2% | ✅ Good |
| Medium | 2-3M | 2-10% | 0.5-2% | ✅ Good |
| Large | 15-20M | 2-10% | 0.5-2% | ✅ Good |
| **Any** | **Any** | **>50%** | **Any** | ❌ **Too many trainable!** |

**If trainable % > 50%:**
- LoRA not applied correctly
- Too many layers unfrozen
- Risk of severe overfitting

### 3. **Check Accuracy Trends**

**Healthy:**
- Train acc increasing steadily
- Val acc increasing (may lag slightly)
- Gap < 5-10%

**Overfitting:**
- Train acc increasing
- Val acc flat or decreasing
- Gap > 20%

**Underfitting:**
- Both accuracies low
- Both increasing slowly
- Small gap

### 4. **Monitor Training Speed**

**Signs of Issues:**

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Very slow (>10 min/epoch on A40) | Batch size too small | Increase batch size |
| OOM errors | Batch size too large | Decrease batch size |
| Loss = NaN | Learning rate too high | Decrease LR 10x |
| Loss not decreasing | Learning rate too low | Increase LR 10x |

---

## Hyperparameter Tuning Guide

### Starting Points (Recommended Defaults)

#### Small Model (M1 MacBook, 100-500 patients)
```bash
python train_behrt_demo.py \
    --model_size small \
    --use_lora \
    --lora_rank 8 \
    --num_patients 200 \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --dropout 0.1 \
    --early_stopping_patience 10
```

#### Large Model (A40 Pod, 1000-10000 patients)
```bash
python train_behrt_demo.py \
    --model_size large \
    --use_lora \
    --lora_rank 16 \
    --num_patients 5000 \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --dropout 0.2 \
    --early_stopping_patience 10
```

### Tuning Strategy

#### If Overfitting (Val loss increasing)

**Priority 1: Increase Regularization**
```bash
# Try these in order:
--dropout 0.3              # Increase dropout
--weight_decay 0.05        # Stronger L2 regularization
--early_stopping_patience 5  # Stop sooner
```

**Priority 2: Reduce Model Capacity**
```bash
--lora_rank 8              # Reduce LoRA rank (if using 16)
--model_size medium        # Use smaller model (if using large)
```

**Priority 3: More Data**
```bash
--num_patients 10000       # More data helps generalization
```

#### If Underfitting (Both losses high)

**Priority 1: Increase Model Capacity**
```bash
--model_size large         # Larger model
--lora_rank 32             # Higher LoRA rank
--dropout 0.05             # Reduce dropout
```

**Priority 2: Train Longer**
```bash
--epochs 200               # More epochs
--early_stopping_patience 20  # More patience
```

**Priority 3: Adjust Learning Rate**
```bash
--lr 5e-4                  # Try higher LR
# or
--lr 5e-5                  # Try lower LR
```

#### If Training is Unstable (Loss spikes, NaN)

**Reduce Learning Rate:**
```bash
--lr 1e-5                  # Much lower LR
--weight_decay 0.001       # Reduce weight decay
```

**Add Gradient Clipping:**
```python
# In training loop:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Hyperparameter Interaction Matrix

| Symptom | Adjust | Direction | Why |
|---------|--------|-----------|-----|
| Val loss increasing | Dropout | ↑ 0.1→0.3 | More regularization |
| Val loss increasing | Weight decay | ↑ 0.01→0.05 | Penalize large weights |
| Val loss increasing | Early stopping | ↓ 10→5 | Stop sooner |
| Both losses high | Model size | ↑ small→large | More capacity |
| Both losses high | LoRA rank | ↑ 8→32 | More trainable params |
| Both losses high | Dropout | ↓ 0.2→0.05 | Less regularization |
| Loss = NaN | Learning rate | ↓ 1e-4→1e-5 | Prevent instability |
| Slow convergence | Learning rate | ↑ 1e-4→5e-4 | Faster learning |
| OOM error | Batch size | ↓ 128→64 | Reduce memory |
| Slow training | Batch size | ↑ 32→128 | Better GPU utilization |

---

## Other Common Issues

### 1. **LoRA Not Applied Correctly**

**Symptom:**
```
📊 Model Parameters:
   Trainable: 18,790,376 (92.3%)  ❌ Too high!
```

**Diagnosis:**
```python
# Check which layers have LoRA
for name, module in model.named_modules():
    if 'lora' in name.lower():
        print(f"LoRA applied to: {name}")
```

**Expected Output:**
```
LoRA applied to: encoder.layers.0.self_attn.out_proj.lora
LoRA applied to: encoder.layers.1.self_attn.out_proj.lora
...
```

**Fix:**
- Ensure LoRA applied to all attention layers
- Freeze embeddings and MLM head if not using LoRA there

### 2. **Validation Set Too Small**

**Symptom:**
```
Val Loss: 6.9181 Acc: 0.0012
Val Loss: 6.9157 Acc: 0.0010
Val Loss: 6.9163 Acc: 0.0009  # Noisy, jumping around
```

**Diagnosis:**
- Validation accuracy jumping randomly
- Hard to tell if model is improving

**Fix:**
```bash
--num_patients 5000  # Use more patients
# With 80/20 split, val set = 1000 patients
```

**Rule of Thumb:**
- Validation set should have ≥500 patients
- For 500 val patients, need 2500 total patients

### 3. **Batch Size Too Small**

**Symptom:**
- Training very slow
- Noisy loss curves
- Poor GPU utilization

**Diagnosis:**
```
🚀 Starting training...
   Train batches: 200  ❌ Too many batches!
```

**Fix:**
```bash
# On A40 (40GB VRAM):
--batch_size 128  # Large model
--batch_size 256  # Medium model

# On M1 (16GB RAM):
--batch_size 16   # Small model
```

### 4. **Learning Rate Too High/Low**

**Too High (Loss = NaN or exploding):**
```
Epoch 1 | Train Loss: 7.00
Epoch 2 | Train Loss: 15.32
Epoch 3 | Train Loss: nan  ❌
```

**Fix:**
```bash
--lr 1e-5  # Reduce by 10x
```

**Too Low (Loss barely decreasing):**
```
Epoch 1  | Train Loss: 7.00
Epoch 10 | Train Loss: 6.98
Epoch 20 | Train Loss: 6.95  ❌ Too slow!
```

**Fix:**
```bash
--lr 1e-3  # Increase by 10x
```

---

## Performance Expectations

### MLM Pre-training (Synthetic Data)

| Metric | Random Baseline | After 10 Epochs | After 50 Epochs | After 100 Epochs |
|--------|----------------|-----------------|-----------------|------------------|
| **Train Accuracy** | 0.1% | 1-2% | 10-20% | 30-40% |
| **Val Accuracy** | 0.1% | 0.5-1% | 5-10% | 15-25% |
| **Train Loss** | 6.91 | 6.5-6.8 | 5.5-6.0 | 4.5-5.5 |
| **Val Loss** | 6.91 | 6.5-6.8 | 5.8-6.2 | 5.5-6.0 |

**Notes:**
- Random baseline = 1/vocab_size = 1/1000 = 0.1%
- Synthetic data has lower ceiling than real data
- Val accuracy should be 50-80% of train accuracy
- If val accuracy < 20% of train accuracy → overfitting

### Real EHR Data (Expected)

| Metric | After 50 Epochs | After 100 Epochs |
|--------|-----------------|------------------|
| **MLM Accuracy** | 40-50% | 50-60% |
| **Val Accuracy** | 35-45% | 45-55% |

### Downstream Tasks (After Pre-training)

| Task | Metric | Expected Performance |
|------|--------|---------------------|
| **Survival Prediction** | C-index | 0.65-0.75 |
| **Disease Prediction** | AUROC | 0.75-0.85 |
| **Readmission** | AUROC | 0.70-0.80 |
| **Phenotyping** | F1 Score | 0.60-0.75 |

---

## Quick Diagnostic Commands

### Check Training Progress
```bash
# Monitor training in real-time
tail -f experiments/behrt_large_mlm/logs/metrics_history.json

# View training curves
ls experiments/behrt_large_mlm/plots/
open experiments/behrt_large_mlm/plots/loss_curve.png
```

### Check Model Parameters
```python
from ehrsequencing.models.lora import count_parameters
from ehrsequencing.models.behrt import BEHRT, BEHRTConfig

config = BEHRTConfig.large(vocab_size=1000)
model = BEHRT(config)

# Before LoRA
params = count_parameters(model)
print(f"Before LoRA: {params['trainable_percent']:.1f}% trainable")

# After LoRA
from ehrsequencing.models.lora import apply_lora_to_behrt
model = apply_lora_to_behrt(model, rank=16)
params = count_parameters(model)
print(f"After LoRA: {params['trainable_percent']:.1f}% trainable")
```

### Analyze Training History
```python
import json
import matplotlib.pyplot as plt

# Load metrics
with open('experiments/behrt_large_mlm/logs/metrics_history.json') as f:
    metrics = json.load(f)

# Plot train vs val loss
train_loss = [m['value'] for m in metrics['train_loss']]
val_loss = [m['value'] for m in metrics['val_loss']]

plt.plot(train_loss, label='Train')
plt.plot(val_loss, label='Val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.savefig('loss_analysis.png')
```

---

## Summary: Overfitting Failure Mode

**What Went Wrong:**
1. 92.3% of parameters trainable (should be 2-10%)
2. No weight decay (L2 regularization)
3. Insufficient dropout (0.1 too low for large model)
4. No early stopping (training continued despite worsening val loss)

**How to Diagnose:**
1. Check loss curves: Val loss increasing = overfitting
2. Check parameter counts: >50% trainable = too many
3. Check accuracy gap: >20% gap = overfitting
4. Monitor patience counter: Stops at right time

**How to Fix:**
1. Add early stopping: `--early_stopping_patience 10`
2. Add weight decay: `--weight_decay 0.01`
3. Increase dropout: `--dropout 0.2`
4. Verify LoRA applied correctly (2-10% trainable)

**Lesson Learned:**
- Always monitor validation metrics, not just training metrics
- Regularization is critical for large models
- Early stopping prevents wasted compute on overfitting
- Parameter count is a key diagnostic signal

---

## References

- **LoRA Paper**: Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
- **BEHRT Paper**: Li et al. (2019) "BEHRT: Transformer for Electronic Health Records"
- **Dropout**: Srivastava et al. (2014) "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- **Weight Decay**: Loshchilov & Hutter (2019) "Decoupled Weight Decay Regularization"
- **Early Stopping**: Prechelt (1998) "Early Stopping - But When?"
