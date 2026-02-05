# Environment Configuration Guide

**Date:** January 20, 2026  
**Focus:** Platform-specific environment setup for ehr-sequencing

---

## Overview

The ehr-sequencing project provides three environment configurations optimized for different hardware platforms:

1. **environment-macos.yml** - M1/M2/M3 Mac (MPS acceleration)
2. **environment-cuda.yml** - Linux/Windows with NVIDIA GPU (CUDA acceleration)
3. **environment-cpu.yml** - CPU-only (any platform)

The default `environment.yml` is identical to `environment-macos.yml` for convenience.

---

## Platform Selection Guide

### macOS (M1/M2/M3 Mac)

**Use:** `environment-macos.yml`

**Features:**
- MPS (Metal Performance Shaders) GPU acceleration
- Optimized for Apple Silicon
- No CUDA dependencies

**Installation:**

```bash
mamba env create -f environment-macos.yml
mamba activate ehrsequencing
```

**Verify MPS:**

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
# Expected: MPS available: True
```

**Performance:**
- Small LSTM model: ~10 min/epoch
- Memory usage: ~3-4GB
- Suitable for development and small-scale training

---

### Linux/Windows with NVIDIA GPU

**Use:** `environment-cuda.yml`

**Features:**
- CUDA 12.1 support
- Full GPU acceleration
- Optimized for NVIDIA GPUs (RTX, A-series, etc.)

**Installation:**

```bash
mamba env create -f environment-cuda.yml
mamba activate ehrsequencing
```

**Verify CUDA:**

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
# Expected: CUDA available: True
#           GPU: NVIDIA GeForce RTX 3090 (or your GPU)
```

**Performance:**
- Medium LSTM model: ~2 min/epoch (24GB GPU)
- Large LSTM model: ~1 min/epoch (40GB+ GPU)
- Suitable for production training

**Prerequisites:**
- NVIDIA GPU with compute capability 7.0+
- NVIDIA driver 525.xx or newer
- CUDA 12.1 compatible GPU

---

### CPU-only (Any Platform)

**Use:** `environment-cpu.yml`

**Features:**
- No GPU dependencies
- Works on any platform
- Minimal setup

**Installation:**

```bash
mamba env create -f environment-cpu.yml
mamba activate ehrsequencing
```

**Performance:**
- Small LSTM model: ~30-60 min/epoch
- Not recommended for large-scale training
- Suitable for testing and small experiments

**Use Cases:**
- Testing code logic
- Small datasets (< 1000 patients)
- CI/CD pipelines
- Machines without GPU

---

## Environment File Comparison

### Common Dependencies (All Platforms)

```yaml
# Python
- python=3.10

# Data Science
- pandas>=2.0
- numpy>=1.24
- scikit-learn>=1.3
- scipy

# Visualization
- matplotlib>=3.7
- seaborn>=0.12
- plotly>=5.17

# Jupyter
- jupyter
- ipykernel
- notebook

# Pip packages
- poetry>=1.6.0
- gensim>=4.3.0
- lifelines>=0.27.0
- umap-learn>=0.5.0
- transformers>=4.30.0
- pytest>=7.4.0
- black>=23.7.0
- ruff>=0.0.285
```

### Platform-Specific Differences

| Dependency | macOS | CUDA | CPU-only |
|------------|-------|------|----------|
| `pytorch>=2.0` | ✅ | ✅ | ✅ |
| `pytorch-cuda=12.1` | ❌ | ✅ | ❌ |
| `cuda-toolkit=12.1` | ❌ | ✅ | ❌ |
| `cpuonly` | ❌ | ❌ | ✅ |
| **Acceleration** | MPS | CUDA | None |

---

## Switching Between Environments

### From macOS to CUDA (e.g., moving to RunPod)

```bash
# On local Mac
mamba env export > environment-backup.yml

# On RunPod
mamba env create -f environment-cuda.yml
mamba activate ehrsequencing
```

### From CUDA to macOS (e.g., moving back to local)

```bash
# On RunPod
# (no special action needed)

# On local Mac
mamba env create -f environment-macos.yml
mamba activate ehrsequencing
```

**Note:** Models trained on one platform work on another (PyTorch handles device differences).

---

## Troubleshooting

### macOS: CUDA Error

**Problem:**
```
error: pytorch-cuda =11.8 * is not installable
```

**Solution:**
```bash
# Use macOS-specific environment
mamba env create -f environment-macos.yml
```

### Linux: MPS Not Available

**Problem:**
```python
>>> torch.backends.mps.is_available()
False
```

**Solution:**
MPS is only available on Apple Silicon Macs. On Linux, use CUDA:
```bash
mamba env create -f environment-cuda.yml
```

### CUDA: Version Mismatch

**Problem:**
```
RuntimeError: CUDA version mismatch
```

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# Update CUDA version in environment-cuda.yml
# Change pytorch-cuda=12.1 to match your driver
# e.g., pytorch-cuda=11.8 for older drivers

mamba env create -f environment-cuda.yml
```

### CPU: Slow Training

**Problem:**
Training is very slow on CPU.

**Solution:**
- Use smaller model config: `--size small`
- Reduce batch size: `--batch-size 4`
- Use fewer epochs for testing
- Consider cloud GPU (RunPod) for production training

---

## Environment Management

### List Environments

```bash
mamba env list
# or
conda env list
```

### Remove Environment

```bash
mamba env remove -n ehrsequencing
```

### Export Environment

```bash
# Export current environment
mamba env export > environment-current.yml

# Export without builds (more portable)
mamba env export --no-builds > environment-portable.yml
```

### Clone Environment

```bash
# Clone to a new name
mamba create --name ehrsequencing-dev --clone ehrsequencing
```

---

## CI/CD Considerations

### GitHub Actions

Use `environment-cpu.yml` for CI/CD:

```yaml
# .github/workflows/test.yml
- name: Setup environment
  run: |
    mamba env create -f environment-cpu.yml
    mamba activate ehrsequencing
    pytest tests/
```

### Docker

```dockerfile
# For CPU-only
FROM continuumio/miniconda3
COPY environment-cpu.yml .
RUN mamba env create -f environment-cpu.yml

# For CUDA
FROM nvidia/cuda:12.1.0-base-ubuntu22.04
COPY environment-cuda.yml .
RUN mamba env create -f environment-cuda.yml
```

---

## Performance Benchmarks

### Training Speed (Medium LSTM, 10K patients)

| Platform | Environment | Time/Epoch | Memory |
|----------|-------------|------------|--------|
| M1 Mac 16GB | environment-macos.yml | ~10 min | ~4GB |
| RTX 3090 24GB | environment-cuda.yml | ~1.5 min | ~8GB |
| A40 48GB | environment-cuda.yml | ~1 min | ~16GB |
| CPU (16 cores) | environment-cpu.yml | ~45 min | ~8GB |

### Installation Time

| Platform | Environment | Time |
|----------|-------------|------|
| macOS | environment-macos.yml | ~5 min |
| Linux GPU | environment-cuda.yml | ~8 min |
| CPU-only | environment-cpu.yml | ~4 min |

---

## Best Practices

### Development Workflow

1. **Local (M1 Mac):** Use `environment-macos.yml`
   - Fast iteration with small models
   - Test code logic
   - Debug issues

2. **Cloud (RunPod):** Use `environment-cuda.yml`
   - Train medium/large models
   - Full dataset training
   - Production experiments

3. **CI/CD:** Use `environment-cpu.yml`
   - Automated testing
   - Code validation
   - No GPU required

### Version Control

**Commit all environment files:**
```bash
git add environment*.yml
git commit -m "Add platform-specific environments"
```

**Document in README:**
```markdown
## Installation

Choose your platform:
- macOS: `mamba env create -f environment-macos.yml`
- Linux/Windows GPU: `mamba env create -f environment-cuda.yml`
- CPU-only: `mamba env create -f environment-cpu.yml`
```

---

## Related Documentation

- [INSTALL.md](../../INSTALL.md) - Quick start guide
- [Local Setup](local-setup.md) - Platform-specific setup details
- [RunPod Setup](runpod-setup.md) - Cloud GPU training

---

**Last Updated:** January 20, 2026
