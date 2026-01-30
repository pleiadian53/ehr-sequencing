# BEHRT Sequence Encoders

This directory contains BEHRT (BERT for EHR) implementation with modern efficient fine-tuning and comprehensive experiment tracking for ephemeral GPU pods.

## Key Features

### 1. LoRA (Low-Rank Adaptation) for Efficient Fine-tuning
- **Reduces trainable parameters by 90-99%** while maintaining performance
- **Faster training** and lower memory usage
- **Smaller checkpoints** - save only LoRA weights (~1-10MB vs 100MB+ for full model)
- Perfect for fine-tuning large pre-trained models on downstream tasks

### 2. Comprehensive Experiment Tracking
- **Automatic checkpointing** - never lose progress on ephemeral pods
- **Training visualizations** - loss curves, accuracy plots, confusion matrices
- **Benchmarking framework** - compare multiple models systematically
- **Shareable results** - all outputs saved for analysis after pod terminates

## Training Modes

This directory provides **two training workflows** with different use cases:

### 1. Pre-training from Scratch (`train_behrt_demo.py`)

**Use when:** No pre-trained embeddings available

**Characteristics:**
- Learns medical code embeddings from scratch
- Requires large dataset (100K+ patients for good embeddings)
- Longer training time
- All embeddings + LoRA adapters trainable

**Example:**
```bash
python train_behrt_demo.py \
    --model_size large \
    --use_lora \
    --lora_rank 16 \
    --num_patients 100000 \
    --realistic_data  # Use realistic synthetic data
```

### 2. Fine-tuning with Pre-trained Embeddings (`train_behrt_finetune.py`)

**Use when:** Pre-trained embeddings available (e.g., from Med2Vec, Word2Vec)

**Characteristics:**
- Loads pre-trained embeddings and freezes them
- Requires smaller dataset (1K-10K patients sufficient)
- Faster convergence
- Only LoRA adapters + task head trainable

**Example:**
```bash
python train_behrt_finetune.py \
    --model_size large \
    --embedding_path pretrained/med2vec_embeddings.pt \
    --use_lora \
    --lora_rank 16 \
    --num_patients 5000 \
    --realistic_data
```

**Comparison:**

| Aspect | Pre-training | Fine-tuning |
|--------|--------------|-------------|
| **Embeddings** | Learn from scratch | Load pre-trained |
| **Dataset size** | 100K+ patients | 1K-10K patients |
| **Training time** | Longer | Faster |
| **Trainable params** | Embeddings + LoRA + head | LoRA + head only |
| **Use case** | No pre-trained available | Pre-trained available |

## Quick Start

### Test BEHRT Locally (Small Model)

```bash
cd examples/encoders

# Basic training (no LoRA)
python train_behrt_demo.py \
    --model_size small \
    --num_patients 100 \
    --epochs 10 \
    --batch_size 16

# With LoRA (90% fewer trainable parameters)
python train_behrt_demo.py \
    --model_size small \
    --use_lora \
    --lora_rank 8 \
    --num_patients 100 \
    --epochs 10
```

**Expected output:**
```
📊 Model Parameters:
   Total: 524,416
   Trainable: 52,480 (10.0%)  # With LoRA rank=8
   Frozen: 471,936
   LoRA: 52,480 (10.0%)

✅ Training complete!
📁 All outputs saved to: experiments/behrt_small_mlm_lora8/
```

### Train on Cloud GPU (Large Model)

```bash
# On RunPods A40
python train_behrt_demo.py \
    --model_size large \
    --use_lora \
    --lora_rank 16 \
    --num_patients 1000 \
    --epochs 50 \
    --batch_size 64 \
    --experiment_name behrt_large_mlm_a40
```

## LoRA Usage Examples

### Apply LoRA to Pre-trained BEHRT

```python
from ehrsequencing.models.behrt import BEHRT, BEHRTConfig
from ehrsequencing.models.lora import apply_lora_to_behrt, count_parameters

# Load pre-trained model
config = BEHRTConfig.large(vocab_size=1000)
model = BEHRT(config)
# ... load pre-trained weights ...

# Apply LoRA for fine-tuning
model = apply_lora_to_behrt(
    model,
    rank=8,              # Lower rank = fewer parameters
    alpha=16.0,          # Scaling factor
    lora_attention=True, # Apply to attention layers
    lora_feedforward=False  # Don't apply to FFN (optional)
)

# Check parameter reduction
params = count_parameters(model)
print(f"Trainable: {params['trainable']:,} ({params['trainable_percent']:.1f}%)")
# Output: Trainable: 524,288 (3.5%)  # 96.5% reduction!
```

### Save and Load LoRA Weights

```python
from ehrsequencing.models.lora import save_lora_weights, load_lora_weights

# After training, save only LoRA weights (tiny file)
save_lora_weights(model, 'checkpoints/lora_weights.pt')
# Saved LoRA weights to checkpoints/lora_weights.pt
# LoRA parameters: 524,288  (~2MB vs 60MB for full model)

# Later: load LoRA weights into same architecture
model = BEHRT(config)
model = apply_lora_to_behrt(model, rank=8)
load_lora_weights(model, 'checkpoints/lora_weights.pt')
```

## Experiment Tracking Usage

### Basic Tracking

```python
from ehrsequencing.utils.experiment_tracker import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(
    experiment_name='behrt_mlm_large',
    output_dir='experiments',
    save_plots=True,
    save_checkpoints=True
)

# Log hyperparameters
tracker.log_hyperparameters({
    'model_size': 'large',
    'learning_rate': 1e-4,
    'batch_size': 64,
    'vocab_size': 1000
})

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer)
    val_loss, val_acc = validate(model, val_loader)
    
    # Log metrics
    tracker.log_metrics(epoch, {
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc
    })
    
    # Save checkpoint
    is_best = val_loss < best_val_loss
    tracker.save_checkpoint(model, optimizer, epoch, 
                           {'val_loss': val_loss}, is_best=is_best)

# Generate plots and summary
tracker.plot_training_curves()
tracker.save_summary()
```

**Output structure:**
```
experiments/behrt_mlm_large/
├── checkpoints/
│   ├── best_model.pt          # Best model weights
│   ├── latest.pt              # Latest checkpoint (for resuming)
│   ├── checkpoint_epoch_10.pt
│   └── checkpoint_epoch_20.pt
├── plots/
│   ├── loss_curve.png         # Training/validation loss
│   ├── accuracy_curve.png     # Training/validation accuracy
│   └── confusion_matrix.png   # If applicable
├── logs/
│   └── metrics_history.json   # All metrics over time
├── hyperparameters.json       # Model config
├── metadata.json              # Experiment metadata
├── summary.json               # Machine-readable summary
└── SUMMARY.txt                # Human-readable summary
```

### Benchmarking Multiple Models

```python
from ehrsequencing.utils.experiment_tracker import BenchmarkTracker

# Initialize benchmark
benchmark = BenchmarkTracker(
    benchmark_name='survival_models_comparison',
    output_dir='benchmarks'
)

# Add results from different experiments
benchmark.add_result('LSTM_baseline', {
    'c_index': 0.53,
    'params': 1.1e6,
    'training_time_hours': 1.5
})

benchmark.add_result('BEHRT_small', {
    'c_index': 0.65,
    'params': 0.5e6,
    'training_time_hours': 0.5
})

benchmark.add_result('BEHRT_large', {
    'c_index': 0.72,
    'params': 15e6,
    'training_time_hours': 3.0
})

benchmark.add_result('BEHRT_large_lora', {
    'c_index': 0.71,
    'params': 15e6,
    'trainable_params': 0.5e6,  # Only LoRA parameters
    'training_time_hours': 2.0
})

# Generate comparison report
benchmark.create_comparison_table()
benchmark.plot_performance_vs_size()
benchmark.save_report()
```

**Output:**
```markdown
# Benchmark Report: survival_models_comparison

## Model Comparison

| Model | c_index | params | training_time_hours | trainable_params |
|---|---|---|---|---|
| LSTM_baseline | 0.5300 | 1100000.0000 | 1.5000 | N/A |
| BEHRT_small | 0.6500 | 500000.0000 | 0.5000 | N/A |
| BEHRT_large | 0.7200 | 15000000.0000 | 3.0000 | N/A |
| BEHRT_large_lora | 0.7100 | 15000000.0000 | 2.0000 | 500000.0000 |

### Best Models by Metric
- **c_index**: BEHRT_large (0.7200)
- **training_time_hours**: BEHRT_small (0.5000)
```

## Model Size Comparison

| Model | Parameters | Embedding | Hidden | Layers | Heads | Max Seq | Use Case |
|-------|-----------|-----------|--------|--------|-------|---------|----------|
| **Small** | ~500K | 64 | 128 | 2 | 4 | 50 | Local dev (M1 16GB) |
| **Medium** | ~2-3M | 128 | 256 | 4 | 8 | 100 | Local/small GPU |
| **Large** | ~10-15M | 256 | 512 | 6 | 8 | 200 | Cloud GPU (A40) |

### With LoRA (rank=8)

| Model | Total Params | Trainable | Reduction | Checkpoint Size |
|-------|-------------|-----------|-----------|----------------|
| **Small** | 500K | 50K | 90% | ~200KB |
| **Medium** | 2.5M | 250K | 90% | ~1MB |
| **Large** | 15M | 500K | 97% | ~2MB |

## Training Workflow for Ephemeral Pods

### 1. Local Development (M1 MacBook)
```bash
# Test on small model with synthetic data
python train_behrt_demo.py \
    --model_size small \
    --num_patients 100 \
    --epochs 10 \
    --experiment_name behrt_small_test

# Verify outputs
ls experiments/behrt_small_test/
```

### 2. Pre-training on Cloud GPU (A40)
```bash
# SSH into RunPods A40
ssh root@<pod-ip>

# Clone repo and setup
git clone <repo-url>
cd ehr-sequencing
pip install -e .

# Pre-train large model with MLM
python examples/encoders/train_behrt_demo.py \
    --model_size large \
    --num_patients 5000 \
    --epochs 100 \
    --batch_size 128 \
    --experiment_name behrt_large_pretrain_mlm \
    --output_dir /workspace/experiments

# IMPORTANT: Download results before terminating pod!
# All outputs are in /workspace/experiments/behrt_large_pretrain_mlm/
```

### 3. Download Results from Pod
```bash
# On local machine
scp -r root@<pod-ip>:/workspace/experiments/behrt_large_pretrain_mlm ./experiments/

# Now you have:
# - Best model checkpoint
# - Training curves
# - Metrics history
# - Summary report
```

### 4. Fine-tuning with LoRA (Local or Pod)
```bash
# Load pre-trained model and fine-tune with LoRA
python examples/encoders/train_behrt_finetune.py \
    --pretrained_checkpoint experiments/behrt_large_pretrain_mlm/checkpoints/best_model.pt \
    --use_lora \
    --lora_rank 8 \
    --task survival \
    --num_patients 1000 \
    --epochs 20
```

## Cost Optimization with LoRA

### Scenario: Fine-tuning BEHRT Large on 5 downstream tasks

**Without LoRA:**
- Full model size: 60MB per task
- Total storage: 300MB (5 tasks × 60MB)
- Training time: 3 hours per task = 15 hours
- GPU cost: 15 hours × $0.40/hr = **$6.00**

**With LoRA (rank=8):**
- LoRA weights: 2MB per task
- Total storage: 10MB (5 tasks × 2MB) + 60MB base = 70MB
- Training time: 2 hours per task = 10 hours (faster convergence)
- GPU cost: 10 hours × $0.40/hr = **$4.00**
- **Savings: $2.00 (33% reduction)**

## Performance Expectations

### Pre-training (MLM)
- **Random baseline**: 0.1% accuracy (1/1000 vocab)
- **Target**: >40% accuracy after 50-100 epochs
- **Large model**: 50-60% accuracy

### Downstream Tasks (with pre-training)
- **Survival prediction**: C-index 0.65-0.75 (vs 0.53 for LSTM)
- **Disease prediction**: AUROC 0.75-0.85
- **Readmission**: AUROC 0.70-0.80

### LoRA Fine-tuning
- **Performance**: 95-99% of full fine-tuning
- **Training time**: 60-80% of full fine-tuning
- **Parameters**: 3-10% of full model

## Best Practices for Ephemeral Pods

### ✅ DO
1. **Always use experiment tracker** - automatic checkpointing and visualization
2. **Save frequently** - checkpoint every epoch or every N batches
3. **Use LoRA for fine-tuning** - faster training, smaller checkpoints
4. **Download immediately** - get all outputs before terminating pod
5. **Test locally first** - verify code works on small model before expensive training
6. **Log everything** - hyperparameters, metrics, system info

### ❌ DON'T
1. **Don't skip checkpointing** - pods can terminate unexpectedly
2. **Don't save only final model** - you want training history too
3. **Don't forget plots** - visualizations are crucial for analysis
4. **Don't use full fine-tuning** - LoRA is almost always better
5. **Don't train without tracking** - you need reproducible results

## Files in This Directory

- `train_behrt_demo.py` - Demo training script with LoRA and tracking
- `train_behrt.py` - Production training script (coming soon)
- `README.md` - This file

## Next Steps

1. Test BEHRT locally with demo script
2. Verify experiment tracking outputs
3. Test LoRA parameter reduction
4. Pre-train large model on A40 pod
5. Fine-tune on downstream tasks with LoRA
6. Compare benchmarks across models

## References

- **BEHRT**: Li et al. (2019) "BEHRT: Transformer for Electronic Health Records"
- **LoRA**: Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
- **Transformers**: Vaswani et al. (2017) "Attention is All You Need"
