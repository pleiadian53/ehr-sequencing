# Benchmarking EHR Models

Comprehensive framework for comparing BEHRT, LSTM, and external models (PyHealth) on EHR tasks.

---

## Overview

The benchmarking module provides tools for:

- **Model Comparison** - BEHRT vs LSTM vs PyHealth Transformer
- **Performance Tracking** - Metrics across multiple training runs
- **Visualization** - Training curves, ROC/PR curves, performance bars
- **Reproducibility** - Consistent evaluation across experiments

---

## Quick Start

### Basic Comparison

```python
from ehrsequencing.benchmarks import (
    BenchmarkTracker,
    BenchmarkVisualizer,
    train_model
)

# Initialize tracker
tracker = BenchmarkTracker(output_dir='experiments/comparison')

# Add runs
tracker.add_run('BEHRT-large', config={'model_size': 'large'})
tracker.add_run('LSTM-baseline', config={'model_size': 'medium'})

# Train models (automatically tracked)
train_model('BEHRT-large', behrt_model, train_loader, val_loader,
            optimizer, device, epochs=50, tracker=tracker)

train_model('LSTM-baseline', lstm_model, train_loader, val_loader,
            optimizer, device, epochs=50, tracker=tracker)

# Generate visualizations
visualizer = BenchmarkVisualizer(output_dir='experiments/plots')
visualizer.plot_all(tracker.get_all_runs())

# Generate summary report
tracker.generate_summary_table()
```

---

## Benchmarking Infrastructure

### 1. BenchmarkTracker

Track metrics across multiple training runs.

**Features:**
- Log epoch-by-epoch metrics (loss, accuracy)
- Track training time and final metrics
- Generate comparison tables (JSON, CSV, text)
- Save/load tracker state

**Example:**
```python
tracker = BenchmarkTracker(output_dir='experiments/benchmark')

# Add run with config
tracker.add_run('my-model', config={
    'model_size': 'large',
    'lora_rank': 16,
    'dropout': 0.2,
    'learning_rate': 1e-4,
    'trainable_params': 15_000_000
})

# Log metrics each epoch
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = evaluate(...)
    tracker.log_epoch('my-model', epoch, train_loss, train_acc, val_loss, val_acc)

# Set final metrics
tracker.set_final_metrics('my-model', {
    'roc_auc': 0.85,
    'pr_auc': 0.78,
    'average_precision': 0.82
})

# Generate summary
summary = tracker.generate_summary_table()
```

### 2. BenchmarkVisualizer

Create publication-quality visualizations.

**Plots:**
- Training/validation curves (loss, accuracy)
- Performance metrics bar charts
- ROC curves comparison
- Precision-Recall curves
- Convergence comparison
- Training time comparison

**Example:**
```python
visualizer = BenchmarkVisualizer(output_dir='experiments/plots')

# Generate all standard plots
visualizer.plot_all(tracker.get_all_runs(), roc_data=roc_data, pr_data=pr_data)

# Or individual plots
visualizer.plot_training_curves(tracker.get_all_runs())
visualizer.plot_performance_metrics(tracker.get_all_runs())
visualizer.plot_roc_curves(roc_data)
visualizer.plot_pr_curves(pr_data)
```

### 3. Training Utilities

Reusable training and evaluation functions.

**Functions:**
- `train_epoch()` - Train for one epoch
- `evaluate()` - Evaluate on validation/test set
- `train_model()` - Full training loop with early stopping
- `compute_metrics()` - ROC-AUC, PR-AUC, Average Precision
- `compute_roc_curve()` - Macro-averaged ROC curve
- `compute_pr_curve()` - Macro-averaged PR curve

---

## Example Benchmarks

### 1. BEHRT vs LSTM

Compare BEHRT (EHR-specific) vs LSTM (baseline) on same task.

**Script:**
```bash
python examples/benchmarking/benchmark_training_comparison.py \
    --num-patients 10000 \
    --epochs 50
```

**What it compares:**
- Small BEHRT with LoRA vs Medium BEHRT with LoRA
- Training curves and convergence speed
- Final performance metrics
- Training time and parameter count

**Expected Output:**
```
experiments/comparison/
├── SUMMARY.txt                    # Text summary table
├── summary.json                   # Machine-readable summary
├── summary.csv                    # Spreadsheet format
├── training_curves.png            # Loss/accuracy curves
├── performance_metrics.png        # Bar chart
├── roc_curves.png                 # ROC comparison
├── pr_curves.png                  # PR comparison
└── training_time.png              # Time comparison
```

---

### 2. BEHRT vs PyHealth

Compare BEHRT vs PyHealth's generic Transformer.

**Script:**
```bash
python examples/benchmarking/benchmark_pyhealth.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --realistic-data
```

**What it tests:**
- Does BEHRT's EHR-specific design (age/visit embeddings) help?
- How much better is domain-specific pre-training?
- Training efficiency comparison

**Expected Results:**
- BEHRT should achieve 5-10% higher accuracy
- BEHRT should converge faster (fewer epochs)
- BEHRT should have smaller train-val gap (better generalization)

---

### 3. Pre-trained Embeddings Comparison

Compare training from scratch vs using pre-trained embeddings.

**Script:**
```bash
python examples/pretrain_finetune/benchmark_pretrained_embeddings.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --external-embedding-path pretrained/med2vec_embeddings.pt
```

**What it compares:**
1. **Run 1:** Training from scratch (learning embeddings)
2. **Run 2:** Fine-tuning with learned embeddings (from Run 1)
3. **Run 3:** Fine-tuning with external embeddings (Med2Vec)

**Expected Results:**
- Pre-trained embeddings should converge 10-20% faster
- Final performance within 5% of training from scratch
- Med2Vec may help if aligned with task domain

---

## Metrics

### Discrimination Metrics

**ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**
- Measures ability to distinguish between classes
- Range: 0.5 (random) to 1.0 (perfect)
- Good: > 0.75, Excellent: > 0.85

**PR-AUC (Precision-Recall - Area Under Curve)**
- Better for imbalanced datasets
- Focuses on positive class performance
- More informative than ROC-AUC for rare events

**Average Precision**
- Summary of precision-recall curve
- Weighted mean of precisions at each threshold
- Equivalent to area under PR curve

### Training Metrics

**Convergence Speed**
- Epochs to best validation loss
- Faster convergence = more efficient training

**Generalization Gap**
- Difference between train and validation performance
- Smaller gap = better generalization

**Training Time**
- Wall-clock time to convergence
- Important for practical deployment

---

## Best Practices

### 1. Fair Comparison

**Same Data:**
- Use identical train/val/test splits
- Same data preprocessing
- Same vocabulary and tokenization

**Same Evaluation:**
- Consistent metrics across all models
- Same evaluation frequency (every epoch)
- Same early stopping criteria

**Same Resources:**
- Same hardware (GPU type, memory)
- Same batch size (adjust if needed)
- Same number of epochs or early stopping

### 2. Reproducibility

**Set Random Seeds:**
```python
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

**Save Configuration:**
```python
tracker.add_run('my-model', config={
    'seed': 42,
    'model_size': 'large',
    'batch_size': 128,
    'learning_rate': 1e-4,
    'dropout': 0.2,
    'optimizer': 'AdamW',
    'weight_decay': 0.01
})
```

**Version Control:**
- Track code version (git commit hash)
- Track dependency versions (environment.yml)
- Save hyperparameters with results

### 3. Statistical Significance

**Multiple Runs:**
- Run each experiment 3-5 times with different seeds
- Report mean ± standard deviation
- Use statistical tests (t-test, Wilcoxon)

**Confidence Intervals:**
- Bootstrap confidence intervals for metrics
- Report 95% CI for ROC-AUC, PR-AUC

**Example:**
```python
results = []
for seed in [42, 123, 456, 789, 1011]:
    set_seed(seed)
    model = train_model(...)
    metrics = evaluate(model, test_loader)
    results.append(metrics['roc_auc'])

mean_auc = np.mean(results)
std_auc = np.std(results)
print(f"ROC-AUC: {mean_auc:.3f} ± {std_auc:.3f}")
```

---

## Advanced Topics

### Model Adapters

Integrate external libraries for comparison.

**PyHealth Adapter:**
```python
from ehrsequencing.benchmarks import PyHealthAdapter

# Create PyHealth model
pyhealth = PyHealthAdapter(config={
    'vocab_size': 1000,
    'embedding_dim': 256,
    'hidden_dim': 512,
    'num_layers': 6,
    'num_heads': 8,
    'dropout': 0.2
})

# Use in benchmark
tracker.add_run('PyHealth-Transformer', config=pyhealth.get_config())
train_model('PyHealth-Transformer', pyhealth, ...)
```

**Custom Adapter:**
```python
from ehrsequencing.benchmarks.adapters import BaseModelAdapter

class MyLibraryAdapter(BaseModelAdapter):
    def build_model(self):
        # Initialize model from external library
        pass
    
    def prepare_data(self, codes, ages, visit_ids, attention_mask, labels):
        # Convert data format
        pass
    
    def train(self, train_loader, val_loader, epochs, learning_rate):
        # Training loop
        pass
    
    def evaluate(self, test_loader):
        # Evaluation
        pass
```

### Ablation Studies

Systematically test which components matter.

**Example: BEHRT Components**
```python
# Full BEHRT
behrt_full = BEHRTForMLM(config)

# Without age embeddings
config_no_age = config.copy()
config_no_age.use_age_embeddings = False
behrt_no_age = BEHRTForMLM(config_no_age)

# Without visit embeddings
config_no_visit = config.copy()
config_no_visit.use_visit_embeddings = False
behrt_no_visit = BEHRTForMLM(config_no_visit)

# Compare all variants
for name, model in [
    ('BEHRT-full', behrt_full),
    ('BEHRT-no-age', behrt_no_age),
    ('BEHRT-no-visit', behrt_no_visit)
]:
    tracker.add_run(name, config=model.config)
    train_model(name, model, ...)
```

### Hyperparameter Search

Find optimal hyperparameters.

**Grid Search:**
```python
for lr in [1e-3, 1e-4, 1e-5]:
    for dropout in [0.1, 0.2, 0.3]:
        name = f'BEHRT-lr{lr}-dropout{dropout}'
        config = BEHRTConfig(learning_rate=lr, dropout=dropout)
        model = BEHRTForMLM(config)
        
        tracker.add_run(name, config=config.__dict__)
        train_model(name, model, ...)
```

**Random Search:**
```python
import random

for trial in range(20):
    lr = 10 ** random.uniform(-5, -3)
    dropout = random.uniform(0.1, 0.4)
    hidden_dim = random.choice([256, 512, 768])
    
    name = f'BEHRT-trial{trial}'
    config = BEHRTConfig(
        learning_rate=lr,
        dropout=dropout,
        hidden_dim=hidden_dim
    )
    model = BEHRTForMLM(config)
    
    tracker.add_run(name, config=config.__dict__)
    train_model(name, model, ...)
```

---

## Output Files

All benchmarks generate:

```
experiments/benchmark_name/
├── SUMMARY.txt                    # Human-readable summary
├── summary.json                   # Machine-readable summary
├── summary.csv                    # Spreadsheet format
├── training_curves.png            # Training/val curves
├── performance_metrics.png        # Bar chart of metrics
├── roc_curves.png                 # ROC comparison
├── pr_curves.png                  # PR comparison
├── convergence_loss.png           # Val loss convergence
├── convergence_accuracy.png       # Val accuracy convergence
├── training_time.png              # Time comparison
└── tracker_state.json             # Full tracker state
```

---

## References

### Documentation

- [Benchmarking Module README](../../src/ehrsequencing/benchmarks/README.md) - Technical details
- [Survival Analysis](survival_analysis.md) - Application-specific benchmarking

### Example Scripts

- `examples/benchmarking/benchmark_pyhealth.py` - BEHRT vs PyHealth
- `examples/benchmarking/benchmark_training_comparison.py` - Model size comparison
- `examples/pretrain_finetune/benchmark_pretrained_embeddings.py` - Embedding comparison

### Papers

- **BEHRT**: Li et al. (2020). "BEHRT: Transformer for Electronic Health Records"
- **PyHealth**: Zhao et al. (2021). "PyHealth: A Python Library for Health Predictive Models"

---

## Support

For questions or issues:
- Check the [benchmarking module README](../../src/ehrsequencing/benchmarks/README.md)
- Review [example scripts](../../examples/benchmarking/)
- See [survival analysis guide](survival_analysis.md) for task-specific benchmarking

---

**Status:** Complete and production-ready  
**Updated:** February 2026
