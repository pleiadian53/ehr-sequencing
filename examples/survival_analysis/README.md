# Survival Analysis Examples

This directory contains training scripts and documentation for discrete-time survival analysis on EHR sequences.

## Directory Structure

```
survival_analysis/
├── README.md                      # This file
├── train_lstm_basic.py            # Basic training script (local/small datasets)
├── train_lstm_runpods.py          # Optimized script for cloud GPU training
└── docs/
    └── runpods_training_guide.md  # Comprehensive RunPods training guide
```

## Scripts

### `train_lstm_basic.py`

**Purpose**: Basic training script for local development and testing

**Use cases**:
- Local testing on CPU/MPS
- Small datasets (100-200 patients)
- Quick prototyping and debugging
- Understanding the training pipeline

**Example usage**:
```bash
python examples/survival_analysis/train_lstm_basic.py \
    --data_dir ~/work/loinc-predictor/data/synthea/all_cohorts/ \
    --outcome synthetic \
    --epochs 50 \
    --batch_size 32
```

**Expected performance**: C-index 0.50-0.60 (limited by small dataset)

---

### `train_lstm_runpods.py`

**Purpose**: Optimized training script for cloud GPU training (RunPods, Vast.ai, etc.)

**Use cases**:
- Large datasets (1000+ patients)
- Production model training
- Achieving best performance
- GPU-accelerated training (A40, RTX 4090, etc.)

**Key features**:
- Early stopping to prevent overfitting
- Learning rate scheduling
- Gradient clipping for stability
- Larger batch sizes (64+)
- Training history logging

**Example usage**:
```bash
# On RunPods VM
python examples/survival_analysis/train_lstm_runpods.py \
    --data_dir /workspace/loinc-predictor/data/synthea/large_cohort_1000/ \
    --outcome synthetic \
    --epochs 100 \
    --batch_size 64 \
    --early_stopping_patience 10 \
    --output_dir checkpoints_large
```

**Expected performance**: C-index 0.65-0.75 (with 1000+ patients)

---

## Documentation

### `docs/runpods_training_guide.md`

Comprehensive guide covering:
- Problem analysis and troubleshooting
- RunPods setup and configuration
- Hyperparameter recommendations
- Expected performance metrics
- Cost estimation
- Common issues and solutions

**Read this if**:
- You're training on cloud GPUs for the first time
- Your model is overfitting or underfitting
- You want to optimize training performance
- You need to understand C-index and survival metrics

---

## Related Resources

### Notebooks
- `notebooks/02_survival_analysis/01_discrete_time_survival_lstm.ipynb` - Interactive tutorial with explanations and theory

### Source Code
- `src/ehrsequencing/models/survival_lstm.py` - Model implementations
- `src/ehrsequencing/models/losses.py` - Loss functions and C-index computation
- `src/ehrsequencing/synthetic/survival.py` - Synthetic data generation

---

## Quick Start

### 1. Local Testing (Small Dataset)

```bash
# Test on your local machine
python examples/survival_analysis/train_lstm_basic.py \
    --data_dir ~/work/loinc-predictor/data/synthea/all_cohorts/ \
    --outcome synthetic \
    --epochs 20 \
    --batch_size 16
```

### 2. Cloud GPU Training (Large Dataset)

```bash
# On RunPods A40 GPU
cd /workspace/ehr-sequencing
mamba activate ehrsequencing
tmux new -s training

python examples/survival_analysis/train_lstm_runpods.py \
    --data_dir /workspace/loinc-predictor/data/synthea/large_cohort_1000/ \
    --outcome synthetic \
    --epochs 100 \
    --batch_size 64 \
    --early_stopping_patience 10
```

---

## Performance Expectations

| Dataset Size | Script | C-index | Training Time | Hardware |
|--------------|--------|---------|---------------|----------|
| 100 patients | `train_lstm_basic.py` | 0.50-0.55 | 5-10 min | CPU/MPS |
| 200 patients | `train_lstm_basic.py` | 0.55-0.60 | 10-15 min | MPS/Small GPU |
| 1000 patients | `train_lstm_runpods.py` | 0.65-0.75 | 1-2 hours | A40/RTX 4090 |

---

## Troubleshooting

### Issue: Poor C-index (< 0.55)

**Possible causes**:
- Dataset too small (< 500 patients)
- Model too simple or too complex
- Synthetic outcomes are random

**Solutions**:
- Use larger dataset
- Check synthetic outcome quality
- Adjust model hyperparameters

See `docs/runpods_training_guide.md` for detailed troubleshooting.

### Issue: Overfitting

**Symptoms**: Train loss decreasing, val loss increasing

**Solutions**:
- Use `train_lstm_runpods.py` with early stopping
- Increase dropout: `--dropout 0.5`
- Increase weight decay: `--weight_decay 0.001`
- Use more data

### Issue: Out of Memory

**Solutions**:
- Reduce batch size: `--batch_size 16`
- Reduce model size: `--embedding_dim 64 --hidden_dim 128`
- Use gradient checkpointing (advanced)

---

## Contributing

When adding new survival analysis examples:

1. **Scripts** go in `examples/survival_analysis/`
2. **Documentation** goes in `examples/survival_analysis/docs/`
3. **Notebooks** go in `notebooks/02_survival_analysis/`
4. **Theory docs** go in `dev/explainer/discrete_time_survival_analysis/`

This maintains consistency with the project structure.

---

## Citation

If you use these scripts in your research, please cite:

```bibtex
@software{ehr_sequencing_survival,
  title = {EHR Sequencing: Discrete-Time Survival Analysis},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/pleiadian53/ehr-sequencing}
}
```
