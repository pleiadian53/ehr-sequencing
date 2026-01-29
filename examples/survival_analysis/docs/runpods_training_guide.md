# RunPods Survival LSTM Training Guide

## Problem Analysis: Your Current Training Run

### Issues Identified

From your training output with 106 patients:

| Issue | Current | Impact |
|-------|---------|--------|
| **Dataset size** | 106 patients (84 train, 22 val) | Severe overfitting |
| **Model size** | 1.09M parameters | ~10,000 params per patient (terrible ratio) |
| **Overfitting** | Train loss: 0.57 → Val loss: 5.0 | Model memorizing, not learning |
| **C-index** | Plateaus at 0.49 | Random performance (0.5 = coin flip) |
| **GPU utilization** | 3 batches only | A40 40GB VRAM massively underutilized |
| **Batch size** | 32 | Too small for GPU efficiency |

### Why C-index is Poor

The C-index measures if the model correctly ranks patients by risk. A C-index of 0.49 means:
- The model ranks patients **randomly** (no better than flipping a coin)
- High-risk patients are NOT predicted to have earlier events
- The model has learned nothing useful about survival patterns

**Root cause**: With only 106 patients and 1M+ parameters, the model overfits to training noise and fails to generalize.

---

## Solution: Use Larger Dataset + Optimized Configuration

### Data Availability

You have access to larger Synthea cohorts:

```bash
# On RunPods VM
ls -lh /workspace/loinc-predictor/data/synthea/

# Available datasets:
# - all_cohorts/          (106 patients - what you used)
# - large_cohort_1000/    (1000 patients - USE THIS!)
```

### Recommended Configuration for A40 GPU

**Hardware**: A40 with 40GB VRAM

**Optimal settings**:
```bash
python examples/train_survival_lstm_runpods.py \
    --data_dir /workspace/loinc-predictor/data/synthea/large_cohort_1000/ \
    --outcome synthetic \
    --epochs 100 \
    --batch_size 64 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --grad_clip 1.0 \
    --early_stopping_patience 10 \
    --lr_scheduler reduce_on_plateau \
    --output_dir checkpoints_large
```

### Expected Improvements

| Metric | Current (106 patients) | Expected (1000 patients) |
|--------|------------------------|--------------------------|
| **Training samples** | 84 | 800 |
| **Validation samples** | 22 | 200 |
| **Params per patient** | ~10,000 | ~1,000 |
| **Train loss** | 0.57 (overfit) | 1.5-2.5 (healthy) |
| **Val loss** | 5.0 (diverged) | 2.0-3.0 (stable) |
| **C-index** | 0.49 (random) | **0.65-0.75** (good) |
| **Training time** | 2 min/epoch | 5-10 min/epoch |
| **GPU batches** | 3 | 12-15 |
| **GPU utilization** | ~5% | ~40-60% |

---

## Key Improvements in `train_survival_lstm_runpods.py`

### 1. Early Stopping
Prevents overfitting by stopping when validation loss stops improving:

```python
--early_stopping_patience 10  # Stop if no improvement for 10 epochs
--min_delta 0.001              # Minimum improvement threshold
```

**Why this helps**: Your current run shows val loss increasing after epoch 14 (2.55 → 5.0), indicating overfitting. Early stopping would have stopped at epoch 14, saving time and preventing performance degradation.

### 2. Learning Rate Scheduling
Reduces learning rate when validation loss plateaus:

```python
--lr_scheduler reduce_on_plateau
--lr_patience 5      # Reduce LR if no improvement for 5 epochs
--lr_factor 0.5      # Reduce LR by 50%
```

**Why this helps**: Allows model to escape local minima and fine-tune in later epochs.

### 3. Gradient Clipping
Prevents exploding gradients in RNNs:

```python
--grad_clip 1.0  # Clip gradients to max norm of 1.0
```

**Why this helps**: LSTMs can have unstable gradients with long sequences. Clipping ensures stable training.

### 4. Larger Batch Size
Better GPU utilization and more stable gradients:

```python
--batch_size 64  # vs. 32 in original script
```

**Why this helps**: 
- A40 has 40GB VRAM - batch size 32 only uses ~10%
- Larger batches → more stable gradient estimates
- Better GPU throughput (fewer kernel launches)

### 5. Stronger Regularization
Prevents overfitting on larger datasets:

```python
--dropout 0.3         # vs. 0.1 in original
--weight_decay 0.0001 # vs. 0.00001 in original
```

**Why this helps**: With 1000 patients, we can afford stronger regularization without underfitting.

---

## Step-by-Step RunPods Training

### 1. Ensure Data is Available

```bash
# SSH into RunPods
ssh runpod-ehr-sequencing

# Check data directory
ls -lh /workspace/loinc-predictor/data/synthea/large_cohort_1000/

# Should see:
# - conditions.csv (6.0M)
# - medications.csv
# - procedures.csv
# - encounters.csv
# etc.
```

If data is missing, transfer it:

```bash
# On local machine
scp -r ~/work/loinc-predictor/data/synthea/large_cohort_1000/ \
    runpod-ehr-sequencing:/workspace/loinc-predictor/data/synthea/
```

### 2. Activate Environment

```bash
mamba activate ehrsequencing

# Verify installation
python -c "from ehrsequencing.models.survival_lstm import DiscreteTimeSurvivalLSTM; print('OK')"
```

### 3. Start Training in tmux

```bash
# Create tmux session (persists if SSH disconnects)
tmux new -s survival_training

# Navigate to project
cd /workspace/ehr-sequencing

# Start training
python examples/train_survival_lstm_runpods.py \
    --data_dir /workspace/loinc-predictor/data/synthea/large_cohort_1000/ \
    --outcome synthetic \
    --epochs 100 \
    --batch_size 64 \
    --early_stopping_patience 10 \
    --output_dir checkpoints_large

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t survival_training
```

### 4. Monitor Training

```bash
# Reattach to tmux
tmux attach -t survival_training

# Or check logs
tail -f checkpoints_large/training.log  # If logging enabled

# Or monitor GPU usage
watch -n 1 nvidia-smi
```

### 5. Expected Output

```
Loading data...
Loaded 50000 events from 1000 patients
Grouping events into visits...
Building vocabulary...
Vocabulary size: 2500
Building patient sequences...
Loaded 950 patient sequences
Creating survival labels for outcome: synthetic
Event rate: 70.00%
Median event/censoring time: 12.0 visits
Train size: 760, Val size: 190
Creating model...
Model parameters: 1,086,977
Parameters per patient: 1430.2

Starting training...

Epoch 1/100
Training: 100%|██████████| 12/12 [00:08<00:00,  1.45it/s]
Train loss: 8.2341
Evaluating: 100%|██████████| 3/3 [00:01<00:00,  2.15it/s]
Val loss: 3.1245, C-index: 0.5234
Saved best model to checkpoints_large/best_model.pt

Epoch 2/100
Training: 100%|██████████| 12/12 [00:07<00:00,  1.62it/s]
Train loss: 4.5123
Evaluating: 100%|██████████| 3/3 [00:01<00:00,  2.31it/s]
Val loss: 2.8934, C-index: 0.5891
Saved best model to checkpoints_large/best_model.pt

...

Epoch 25/100
Training: 100%|██████████| 12/12 [00:07<00:00,  1.58it/s]
Train loss: 2.1234
Evaluating: 100%|██████████| 3/3 [00:01<00:00,  2.18it/s]
Val loss: 2.3456, C-index: 0.6823
Saved best model to checkpoints_large/best_model.pt

...

Epoch 35/100
Training: 100%|██████████| 12/12 [00:07<00:00,  1.61it/s]
Train loss: 1.9876
Evaluating: 100%|██████████| 3/3 [00:01<00:00,  2.22it/s]
Val loss: 2.3512, C-index: 0.6798

Early stopping triggered after 35 epochs
Best validation loss: 2.3456
Best C-index: 0.6823

Training complete!
```

---

## Understanding the Results

### Good Training Indicators

✅ **C-index > 0.65**: Model learns meaningful survival patterns
✅ **Val loss stable**: No divergence from train loss
✅ **Early stopping**: Prevents overfitting automatically
✅ **Gradual improvement**: C-index increases steadily

### Warning Signs

⚠️ **C-index < 0.55**: Model barely better than random
⚠️ **Val loss increasing**: Overfitting (train loss decreasing, val loss increasing)
⚠️ **C-index decreasing**: Model learning inverse relationship
⚠️ **Loss = NaN**: Gradient explosion (reduce learning rate or add gradient clipping)

### Interpreting C-index

| C-index | Interpretation | Clinical Utility |
|---------|----------------|------------------|
| 0.50 | Random (coin flip) | Useless |
| 0.55-0.60 | Weak signal | Limited utility |
| 0.60-0.70 | Moderate discrimination | Useful for stratification |
| 0.70-0.80 | Good discrimination | Clinically actionable |
| 0.80+ | Excellent (rare in EHR) | High confidence predictions |

**Your target**: 0.65-0.75 with 1000 patients

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.5 GB
```

**Solutions**:
```bash
# Reduce batch size
--batch_size 32  # or 16

# Reduce model size
--embedding_dim 64 --hidden_dim 128

# Reduce sequence length (if very long visits)
# Add to PatientSequenceBuilder: max_visits=50
```

### Issue: Training Too Slow

**Symptoms**: >15 min/epoch

**Solutions**:
```bash
# Increase batch size (if memory allows)
--batch_size 128

# Reduce model complexity
--num_layers 1

# Use fewer patients for debugging
# (but defeats the purpose of RunPods!)
```

### Issue: C-index Not Improving

**Symptoms**: C-index stuck at 0.50-0.55

**Possible causes**:
1. **Synthetic outcomes are random**: Check outcome generation
2. **Model too simple**: Increase `hidden_dim` or `num_layers`
3. **Learning rate too high**: Reduce `--lr 0.0001`
4. **Need more epochs**: Increase `--epochs 200`

**Diagnostic**:
```python
# Check synthetic outcome quality
from ehrsequencing.synthetic.survival import DiscreteTimeSurvivalGenerator
import scipy.stats as stats

# Should see negative correlation between risk and event time
correlation, p = stats.pearsonr(risk_scores, event_times)
print(f"Correlation: {correlation:.3f} (should be < -0.3)")
```

### Issue: Validation Loss Diverging

**Symptoms**: Train loss decreasing, val loss increasing

**Solutions**:
```bash
# Stronger regularization
--dropout 0.5
--weight_decay 0.001

# Earlier stopping
--early_stopping_patience 5

# More data (if available)
# Use all_cohorts instead of large_cohort_1000
```

---

## Comparison: Original Script vs. RunPods Script

| Feature | `train_survival_lstm.py` | `train_survival_lstm_runpods.py` |
|---------|--------------------------|----------------------------------|
| **Target** | Local testing | Cloud GPU training |
| **Dataset size** | Small (100-200) | Large (1000+) |
| **Batch size** | 32 | 64 (configurable) |
| **Early stopping** | ❌ No | ✅ Yes |
| **LR scheduling** | ❌ No | ✅ Yes |
| **Gradient clipping** | ❌ No | ✅ Yes |
| **Regularization** | Light | Strong |
| **Training history** | ❌ Not saved | ✅ Saved to JSON |
| **Expected C-index** | 0.50-0.60 | 0.65-0.75 |

---

## Next Steps After Training

### 1. Evaluate Model

```python
# Load best model
checkpoint = torch.load('checkpoints_large/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on test set (if you have one)
test_loss, test_c_index = evaluate(model, test_loader, criterion, device)
print(f"Test C-index: {test_c_index:.4f}")
```

### 2. Analyze Predictions

```python
# Get survival curves for specific patients
survival_probs, cumulative_hazards = model.predict_survival(
    visit_codes, visit_mask, sequence_mask
)

# Plot survival curves
import matplotlib.pyplot as plt
for i in range(5):
    plt.plot(survival_probs[i].cpu().numpy(), label=f'Patient {i}')
plt.xlabel('Visit')
plt.ylabel('Survival Probability')
plt.legend()
plt.show()
```

### 3. Feature Importance

```python
# Analyze which medical codes are most predictive
# (Requires additional analysis code)
```

### 4. Deploy Model

```python
# Save for inference
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': builder.vocab,
    'config': {
        'vocab_size': vocab_size,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2
    }
}, 'production_model.pt')
```

---

## Cost Estimation

**RunPods A40 GPU**: ~$0.40/hour

**Training time estimates**:
- 1000 patients, 100 epochs: ~1-2 hours
- Cost: **$0.40-$0.80** per training run

**Tips to save money**:
1. Use early stopping (stops automatically when done)
2. Debug locally first with small dataset
3. Use spot instances (cheaper but can be interrupted)
4. Terminate pod immediately after training

---

## Summary

### What You Should Do

1. **Use the new script**: `train_survival_lstm_runpods.py`
2. **Use larger dataset**: `large_cohort_1000/` (1000 patients)
3. **Use recommended config**: Batch size 64, early stopping, LR scheduling
4. **Monitor training**: Watch for C-index > 0.65
5. **Stop when satisfied**: Don't waste GPU time

### Expected Results

With 1000 patients and optimized configuration:
- **C-index**: 0.65-0.75 (vs. 0.49 currently)
- **Training time**: 1-2 hours (vs. 2 min with 106 patients)
- **Cost**: $0.40-$0.80 (RunPods A40)
- **Model quality**: Clinically useful predictions

### Why This Will Work

1. **10x more data**: 1000 vs. 106 patients → better generalization
2. **Better param ratio**: 1000 params/patient vs. 10,000 → less overfitting
3. **Early stopping**: Prevents the val loss explosion you saw
4. **LR scheduling**: Helps model converge to better solution
5. **Stronger regularization**: Dropout 0.3 + weight decay prevents memorization

---

**Ready to train? Run this on your RunPods A40:**

```bash
cd /workspace/ehr-sequencing
mamba activate ehrsequencing
tmux new -s training

python examples/train_survival_lstm_runpods.py \
    --data_dir /workspace/loinc-predictor/data/synthea/large_cohort_1000/ \
    --outcome synthetic \
    --epochs 100 \
    --batch_size 64 \
    --early_stopping_patience 10 \
    --output_dir checkpoints_large
```

Good luck! 🚀
