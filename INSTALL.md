# Installation Guide

## Prerequisites

- **Conda/Mamba**: For environment management
- **Python**: 3.10 or higher (< 3.13)
- **Git**: For version control

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd ehr-sequencing
```

### 2. Create Environment

**Choose the appropriate environment file for your system:**

**macOS (M1/M2/M3 Mac) - Recommended:**

```bash
# Uses MPS (Metal Performance Shaders) for GPU acceleration
mamba env create -f environment-macos.yml
mamba activate ehrsequencing
```

**Linux/Windows with NVIDIA GPU:**

```bash
# Uses CUDA 12.1 for GPU acceleration
mamba env create -f environment-cuda.yml
mamba activate ehrsequencing
```

**CPU-only (any platform):**

```bash
# No GPU acceleration
mamba env create -f environment-cpu.yml
mamba activate ehrsequencing
```

**Default (macOS-compatible):**

```bash
# Same as environment-macos.yml
mamba env create -f environment.yml
mamba activate ehrsequencing
```

> **Note:** Replace `mamba` with `conda` if you prefer conda over mamba.

### 3. Install Package with Poetry

```bash
# Install poetry if not already installed
pip install poetry

# Install package and dependencies
poetry install
```

**Alternative: Install with pip (editable mode)**

```bash
pip install -e .
```

### 4. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10+

# Test import
python -c "import ehrsequencing; print(ehrsequencing.__version__)"

# Run tests
pytest tests/
```

## Environment Management

### Activating the Environment

```bash
mamba activate ehrsequencing
# or
conda activate ehrsequencing
```

### Updating Dependencies

```bash
# Update from environment file (use your platform-specific file)
mamba env update -f environment-macos.yml  # macOS
# or
mamba env update -f environment-cuda.yml   # Linux/Windows GPU
# or
mamba env update -f environment-cpu.yml    # CPU-only

# Update with poetry
poetry update
```

### Deactivating

```bash
conda deactivate
```

## Development Setup

### Additional Development Tools

```bash
# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Install Jupyter extensions
jupyter contrib nbextension install --user
```

### IDE Setup

**VS Code:**
1. Install Python extension
2. Select interpreter: `ehrsequencing` environment
3. Enable linting (Ruff) and formatting (Black)

**PyCharm:**
1. Set project interpreter to `ehrsequencing` environment
2. Enable Black formatter
3. Configure Ruff for linting

## Data Setup

### Synthea (Synthetic Data)

```bash
# Download and install Synthea
# See: https://github.com/synthetichealth/synthea

# Generate synthetic data
./run_synthea -p 10000

# Move to project data directory
mkdir -p data/synthea
cp output/csv/*.csv data/synthea/
```

### MIMIC-III/IV (Real Data)

1. **Apply for access**: https://physionet.org/
2. **Complete CITI training**
3. **Sign data use agreement**
4. **Download data** (after approval)
5. **Set up PostgreSQL database** (optional)

```bash
# Create data directories
mkdir -p data/mimic
```

### Pre-trained Models

```bash
# Download CEHR-BERT pre-trained embeddings
# See: dev/installation/pretrained-models.md for details

mkdir -p checkpoints/cehrbert
# Download from Hugging Face or model repository
```

## Hardware-Specific Setup

### M1 MacBook (Local Development)

```bash
# Verify MPS (Metal Performance Shaders) support
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Use small model configs for development
# See: docs/implementation/resource-aware-models.md
```

### RunPod / Cloud GPU

For detailed setup on RunPod or cloud GPU instances, see:
- **[RunPod Setup Guide](dev/installation/runpod-setup.md)**
- Includes A40, A100, RTX 4090 configurations
- SSH setup, data transfer, and training workflows

## Troubleshooting

### Conda Environment Issues

```bash
# Remove and recreate environment
mamba env remove -n ehrsequencing
mamba env create -f environment.yml
```

### Poetry Installation Issues

```bash
# Clear poetry cache
poetry cache clear pypi --all

# Reinstall
poetry install --no-cache
```

### Import Errors

```bash
# Ensure package is installed in editable mode
pip install -e .

# Check PYTHONPATH
echo $PYTHONPATH
```

### PyTorch MPS Issues (M1 Mac)

```bash
# If MPS is not available, PyTorch will fall back to CPU
# Ensure you have the latest PyTorch version
mamba install pytorch::pytorch -c pytorch
```

### Database Connection (MIMIC)

```bash
# Test PostgreSQL connection
psql -h localhost -U your_username -d mimic3

# Set environment variables
export MIMIC_USER=your_username
export MIMIC_PASSWORD=your_password
```

## Next Steps

1. **Read documentation**: `docs/README.md`
2. **Explore notebooks**: `notebooks/README.md`
3. **Run examples**: `examples/README.md`
4. **Check implementation plan**: `docs/implementation/visit-grouped-sequences.md`
5. **Review model configs**: `docs/implementation/resource-aware-models.md`

## Detailed Installation Guides

For more detailed installation instructions, see:

- **[Local Development Setup](dev/installation/local-setup.md)** - M1 MacBook, Windows, Linux
- **[RunPod Setup](dev/installation/runpod-setup.md)** - Cloud GPU training
- **[Pre-trained Models](dev/installation/pretrained-models.md)** - CEHR-BERT, Med-BERT
- **[Database Setup](dev/installation/database-setup.md)** - MIMIC-III/IV PostgreSQL
- **[Troubleshooting Guide](dev/installation/troubleshooting.md)** - Common issues

## Getting Help

- Check `docs/` for detailed documentation
- Review `dev/workflow/` for internal notes (if you have access)
- Open an issue on GitHub

## Related Projects

- **[loinc-predictor](https://github.com/YOUR_USERNAME/loinc-predictor)** - LOINC code prediction
- **[genai-lab](https://github.com/YOUR_USERNAME/genai-lab)** - Generative AI for biomedical data
