# Benchmarking EHR Models

This directory contains scripts for benchmarking ehrsequencing models against external libraries like PyHealth.

## Overview

The benchmarking framework provides:
- **Adapters** for external libraries (PyHealth, etc.)
- **Unified metrics** for fair comparison
- **Automated comparison** of multiple models
- **Comprehensive reporting** (JSON, CSV, Markdown)

## Setup

### Option 1: Create Benchmarking Environment (Recommended)

```bash
# Create environment with PyHealth and benchmarking tools
mamba env create -f environment-benchmarking.yml
mamba activate ehrseq-benchmarking
```

### Option 2: Add PyHealth to Existing Environment

```bash
mamba activate ehrseq  # or your environment name
pip install pyhealth>=1.1.0
```

## Quick Start

### Basic Benchmark: BEHRT vs PyHealth Transformer

```bash
python examples/benchmarking/benchmark_pyhealth.py
```

This will:
1. Train both BEHRT and PyHealth's Transformer on the same data
2. Evaluate both models on the same test set
3. Generate comparison reports in `examples/benchmarking/results/`

### Custom Configuration

```bash
python examples/benchmarking/benchmark_pyhealth.py \
    --model-size large \
    --num-patients 5000 \
    --epochs 50 \
    --batch-size 128
```

## Architecture Comparison

### BEHRT (Our Implementation)
- ✅ **EHR-specific embeddings**: Code + Age + Visit + Segment
- ✅ **LoRA support**: Efficient fine-tuning
- ✅ **Flexible configs**: Small/Medium/Large
- ✅ **MLM pre-training**: Masked language modeling

### PyHealth Transformer (Baseline)
- ⚠️ **Generic transformer**: Only code embeddings
- ❌ **No age/visit awareness**: Missing EHR-specific features
- ⚠️ **Standard architecture**: Not specialized for EHR

**Expected Result**: BEHRT should outperform PyHealth's generic transformer on EHR tasks.

## Output

Benchmark results are saved to `examples/benchmarking/results/`:

```
results/
├── benchmark_results.json    # Full results with training history
├── comparison.json            # Detailed comparison statistics
├── summary.csv               # Summary table (CSV)
└── summary.md                # Summary table (Markdown)
```

### Example Summary

```markdown
| Model                | Test Accuracy | Test Loss | Training Time (s) |
|---------------------|---------------|-----------|-------------------|
| BEHRT-Large         | 0.7234        | 1.8456    | 245.3             |
| PyHealth-Transformer| 0.6512        | 2.1234    | 198.7             |
```

## Advanced Usage

### Compare Multiple Architectures

```bash
python examples/benchmarking/compare_architectures.py
```

This compares:
- BEHRT (full, with age/visit embeddings)
- BEHRT (no age embeddings)
- BEHRT (no visit embeddings)
- PyHealth Transformer (baseline)

### Custom Benchmark Script

```python
from ehrsequencing.benchmarks import PyHealthAdapter, ModelComparator
from ehrsequencing.models import BEHRTForMLM

# Create models
behrt = BEHRTForMLM(config)
pyhealth = PyHealthAdapter(config)

# Run benchmark
comparator = ModelComparator([behrt, pyhealth], output_dir='my_results')
results = comparator.run_benchmark(train_loader, val_loader, test_loader)

# Access results
print(results['comparison'])
```

## Key Differences: BEHRT vs PyHealth

| Feature | BEHRT | PyHealth |
|---------|-------|----------|
| Code Embeddings | ✅ | ✅ |
| Age Embeddings | ✅ | ❌ |
| Visit Embeddings | ✅ | ❌ |
| Segment Embeddings | ✅ | ❌ |
| LoRA Support | ✅ | ❌ |
| MLM Pre-training | ✅ | ⚠️ Generic |

## Troubleshooting

### PyHealth Not Found

```bash
# Install PyHealth
pip install pyhealth

# Or use benchmarking environment
mamba env create -f environment-benchmarking.yml
```

### CUDA Out of Memory

```bash
# Reduce batch size or model size
python benchmark_pyhealth.py --batch-size 64 --model-size medium
```

### Slow Training

```bash
# Reduce epochs or dataset size
python benchmark_pyhealth.py --epochs 20 --num-patients 2000
```

## Citation

If you use this benchmarking framework, please cite:

```bibtex
@article{behrt2020,
  title={BEHRT: Transformer for Electronic Health Records},
  author={Li, Yikuan and Rao, Shishir and Solares, José Roberto Ayala and others},
  journal={Scientific Reports},
  year={2020}
}
```

## Contributing

To add support for other libraries (TorchEHR, etc.):

1. Create adapter in `src/ehrsequencing/benchmarks/adapters/`
2. Implement `BaseModelAdapter` interface
3. Add example script in `examples/benchmarking/`
4. Update this README

## License

Same as ehrsequencing project.
