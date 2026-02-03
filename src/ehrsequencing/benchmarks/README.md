# Benchmarking Module

Reusable utilities for benchmarking EHR models across different frameworks and configurations.

## Overview

The `ehrsequencing.benchmarks` module provides a comprehensive toolkit for:
- **Tracking** multiple training runs with consistent metrics
- **Comparing** different model architectures and configurations
- **Visualizing** training curves, performance metrics, and ROC/PR curves
- **Adapting** external libraries (PyHealth, etc.) for fair comparison
- **Reporting** results in multiple formats (JSON, CSV, text, plots)

## Architecture

```
src/ehrsequencing/benchmarks/
├── __init__.py           # Public API
├── tracker.py            # BenchmarkTracker - track multiple runs
├── visualization.py      # BenchmarkVisualizer - plotting utilities
├── training.py           # Training/evaluation loops
├── metrics.py            # UnifiedMetrics - consistent metric computation
├── comparators.py        # ModelComparator - orchestrate comparisons
└── adapters/             # Adapters for external libraries
    ├── base.py           # BaseModelAdapter interface
    └── pyhealth.py       # PyHealth adapter
```

## Quick Start

### Basic Usage: Track and Compare Training Runs

```python
from ehrsequencing.benchmarks import (
    BenchmarkTracker,
    BenchmarkVisualizer,
    train_model
)

# Initialize tracker
tracker = BenchmarkTracker(output_dir='experiments/comparison')

# Add runs
tracker.add_run('BEHRT-small', config={'model_size': 'small'})
tracker.add_run('BEHRT-large', config={'model_size': 'large'})

# Train models (automatically tracked)
train_model('BEHRT-small', model_small, train_loader, val_loader,
            optimizer, device, epochs=50, tracker=tracker)

train_model('BEHRT-large', model_large, train_loader, val_loader,
            optimizer, device, epochs=50, tracker=tracker)

# Generate visualizations
visualizer = BenchmarkVisualizer(output_dir='experiments/plots')
visualizer.plot_all(tracker.get_all_runs())

# Generate summary report
tracker.generate_summary_table()
```

### Compare Against External Libraries

```python
from ehrsequencing.benchmarks import PyHealthAdapter, ModelComparator

# Create PyHealth adapter
pyhealth = PyHealthAdapter(config={
    'vocab_size': 1000,
    'embedding_dim': 256,
    'hidden_dim': 512,
    'num_layers': 6,
    'num_heads': 8,
    'dropout': 0.2
})

# Compare with BEHRT
comparator = ModelComparator(
    models=[behrt_model, pyhealth],
    output_dir='experiments/behrt_vs_pyhealth'
)

results = comparator.run_benchmark(
    train_loader, val_loader, test_loader,
    epochs=50
)
```

## Core Components

### 1. BenchmarkTracker

Track metrics across multiple training runs.

**Features:**
- Log epoch-by-epoch metrics (loss, accuracy)
- Track training time
- Store final evaluation metrics
- Generate comparison tables (JSON, CSV, text)
- Save/load tracker state

**Example:**
```python
tracker = BenchmarkTracker(output_dir='experiments/benchmark')

# Add run
tracker.add_run('my-model', config={'lr': 1e-4, 'dropout': 0.2})

# Log metrics each epoch
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = evaluate(...)
    tracker.log_epoch('my-model', epoch, train_loss, train_acc, val_loss, val_acc)

# Set final metrics
tracker.set_final_metrics('my-model', {'roc_auc': 0.85, 'pr_auc': 0.78})

# Generate summary
tracker.generate_summary_table()
```

### 2. BenchmarkVisualizer

Create publication-quality visualizations.

**Plots:**
- Training/validation curves (loss, accuracy)
- Performance metrics bar charts
- ROC curves
- Precision-Recall curves
- Convergence comparison
- Training time comparison

**Example:**
```python
visualizer = BenchmarkVisualizer(output_dir='experiments/plots')

# Plot all standard visualizations
visualizer.plot_all(tracker.get_all_runs(), roc_data=roc_data, pr_data=pr_data)

# Or individual plots
visualizer.plot_training_curves(tracker.get_all_runs())
visualizer.plot_performance_metrics(tracker.get_all_runs())
visualizer.plot_roc_curves(roc_data)
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

**Example:**
```python
from ehrsequencing.benchmarks import train_model, compute_metrics

# Train with automatic tracking
probs, labels = train_model(
    name='my-model',
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=50,
    tracker=tracker,  # Optional
    vocab_size=1000,
    patience=10
)

# Compute metrics
metrics = compute_metrics(probs, labels, vocab_size=1000)
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

### 4. External Library Adapters

Unified interface for external libraries.

**Available Adapters:**
- `PyHealthAdapter` - PyHealth's Transformer

**Create Custom Adapter:**
```python
from ehrsequencing.benchmarks.adapters import BaseModelAdapter

class MyLibraryAdapter(BaseModelAdapter):
    def build_model(self):
        # Initialize model from external library
        pass
    
    def prepare_data(self, codes, ages, visit_ids, attention_mask, labels):
        # Convert ehrsequencing data format to library format
        pass
    
    def train(self, train_loader, val_loader, epochs, learning_rate):
        # Training loop
        pass
    
    def evaluate(self, test_loader):
        # Evaluation
        pass
```

## Examples

### Example 1: Compare Model Sizes

```bash
python examples/benchmarking/benchmark_training_comparison.py
```

Compares small vs medium BEHRT models with LoRA.

### Example 2: BEHRT vs PyHealth

```bash
python examples/benchmarking/benchmark_pyhealth.py
```

Compares BEHRT (EHR-specific) vs PyHealth's generic Transformer.

### Example 3: Pre-training vs Fine-tuning

```bash
python examples/pretrain_finetune/benchmark_pretrained_embeddings.py \
    --num-patients 10000 \
    --epochs 100
```

Compares training from scratch vs using pre-trained embeddings.

## Output Files

All benchmarks generate:

```
experiments/benchmark_name/
├── SUMMARY.txt                    # Human-readable summary table
├── summary.json                   # Machine-readable summary
├── summary.csv                    # Spreadsheet-compatible summary
├── training_curves.png            # Training/val loss and accuracy
├── performance_metrics.png        # Bar chart of final metrics
├── roc_curves.png                 # ROC curves comparison
├── pr_curves.png                  # Precision-Recall curves
├── convergence_loss.png           # Validation loss convergence
├── convergence_accuracy.png       # Validation accuracy convergence
├── training_time.png              # Training time comparison
└── tracker_state.json             # Full tracker state (for resuming)
```

## Integration with Existing Code

The benchmarking utilities were extracted from `benchmark_pretrained_embeddings.py` and made reusable. You can now:

1. **Use in any script:**
   ```python
   from ehrsequencing.benchmarks import BenchmarkTracker, train_model
   ```

2. **Replace custom tracking code:**
   ```python
   # Old way
   history = {'train_loss': [], 'val_loss': []}
   history['train_loss'].append(loss)
   
   # New way
   tracker = BenchmarkTracker()
   tracker.add_run('my-model', config={})
   tracker.log_epoch('my-model', epoch, train_loss, train_acc, val_loss, val_acc)
   ```

3. **Reuse visualization code:**
   ```python
   # Old way
   plt.plot(train_losses, label='train')
   plt.plot(val_losses, label='val')
   
   # New way
   visualizer = BenchmarkVisualizer()
   visualizer.plot_training_curves(tracker.get_all_runs())
   ```

## Dependencies

**Core (always available):**
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

**Optional (for external library comparison):**
- PyHealth (`pip install pyhealth`)

## Best Practices

1. **Always use tracker for multi-run comparisons**
   - Ensures consistent metric computation
   - Automatic summary generation
   - Easy to add new runs

2. **Use descriptive run names**
   - Good: `'BEHRT-large-lora16-dropout0.2'`
   - Bad: `'run1'`, `'test'`

3. **Save tracker state for long experiments**
   ```python
   tracker.save_state('checkpoint.json')
   # Later...
   tracker.load_state('checkpoint.json')
   ```

4. **Include config in run metadata**
   ```python
   tracker.add_run('my-model', config={
       'model_size': 'large',
       'lora_rank': 16,
       'dropout': 0.2,
       'learning_rate': 1e-4,
       'trainable_params': count_parameters(model)
   })
   ```

## Contributing

To add support for a new external library:

1. Create adapter in `adapters/new_library.py`
2. Inherit from `BaseModelAdapter`
3. Implement required methods
4. Add to `__init__.py` with optional import
5. Create example in `examples/benchmarking/`
6. Update this README

## Citation

If you use this benchmarking framework, please cite the ehrsequencing project and the BEHRT paper:

```bibtex
@article{behrt2020,
  title={BEHRT: Transformer for Electronic Health Records},
  author={Li, Yikuan and Rao, Shishir and Solares, José Roberto Ayala and others},
  journal={Scientific Reports},
  year={2020}
}
```
