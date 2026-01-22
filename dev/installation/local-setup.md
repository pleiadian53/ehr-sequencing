# Local Development Setup

**Date:** January 20, 2026  
**Focus:** Setting up ehr-sequencing on local machines (M1 Mac, Windows, Linux)

---

## Overview

This guide covers detailed setup for local development on different operating systems. For quick start, see [INSTALL.md](../../INSTALL.md).

---

## M1 MacBook Pro Setup (Recommended for Development)

### System Requirements

- **Hardware**: M1/M2/M3 MacBook Pro/Air
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 20GB free space
- **OS**: macOS 12.0 (Monterey) or later

### Step 1: Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Miniforge (Mamba + Conda)

```bash
# Download Miniforge for Apple Silicon
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"

# Install
bash Miniforge3-MacOSX-arm64.sh -b -p ~/miniforge3

# Initialize shell
~/miniforge3/bin/conda init zsh  # or bash

# Reload shell
source ~/.zshrc  # or ~/.bashrc
```

### Step 3: Clone and Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd ehr-sequencing

# Create environment
mamba env create -f environment.yml

# Activate
mamba activate ehrsequencing

# Install package
poetry install
```

### Step 4: Verify PyTorch MPS Support

```bash
# Check MPS (Metal Performance Shaders) availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"
```

**Expected output:**
```
PyTorch version: 2.1.0
MPS available: True
MPS built: True
```

### Step 5: Test Small Model

```bash
# Use small config for M1 MacBook
python -c "
from ehrsequencing.models.configs import get_model_config
config = get_model_config('small', 'lstm')
print(f'Model: {config.model_type}')
print(f'Parameters: ~{config.total_params_millions}M')
print(f'Memory estimate: ~{config.memory_estimate_gb(dtype_bytes=2)}GB (fp16)')
"
```

### M1-Specific Considerations

**Memory Management:**
- Use `batch_size=4` with `gradient_accumulation_steps=8`
- Enable gradient checkpointing: `use_checkpoint=True`
- Use mixed precision: `mixed_precision=True`

**Performance:**
- MPS is 2-3x faster than CPU for small models
- For large models, consider RunPod (see [runpod-setup.md](runpod-setup.md))
- Training time: ~10 min/epoch for small datasets

**Known Issues:**
- Some PyTorch operations not yet optimized for MPS
- Falls back to CPU automatically (no error)
- Monitor Activity Monitor for memory usage

---

## Windows Setup

### System Requirements

- **OS**: Windows 10/11 (64-bit)
- **RAM**: 16GB minimum
- **Storage**: 20GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Step 1: Install WSL2 (Recommended)

```powershell
# Open PowerShell as Administrator
wsl --install

# Restart computer
# After restart, set up Ubuntu
```

**Alternative: Native Windows (not recommended for development)**

### Step 2: Install Miniforge in WSL2

```bash
# In WSL2 Ubuntu terminal
cd /tmp
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

### Step 3: Install CUDA Toolkit (if using GPU)

```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA toolkit via conda
mamba install cuda -c nvidia
```

### Step 4: Clone and Setup

```bash
cd ~
git clone <repository-url>
cd ehr-sequencing

mamba env create -f environment.yml
mamba activate ehrsequencing
poetry install
```

### Step 5: Verify GPU Support

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Windows-Specific Considerations

**File Paths:**
- Use WSL2 paths: `/home/username/ehr-sequencing`
- Avoid Windows paths: `C:\Users\...` (slower I/O)

**Performance:**
- WSL2 with GPU: Similar to native Linux
- Native Windows: Slower, not recommended

**VS Code Integration:**
- Install "Remote - WSL" extension
- Open project in WSL: `code .` from WSL terminal

---

## Linux Setup

### System Requirements

- **OS**: Ubuntu 20.04+ / Debian 11+ / CentOS 8+
- **RAM**: 16GB minimum
- **Storage**: 20GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional)

### Step 1: Install Miniforge

```bash
cd /tmp
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

### Step 2: Install CUDA (if using GPU)

**Ubuntu/Debian:**

```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA via conda (recommended)
mamba install cuda -c nvidia

# Or install from NVIDIA repository
# See: https://developer.nvidia.com/cuda-downloads
```

**CentOS/RHEL:**

```bash
sudo yum install nvidia-driver-latest-dkms
sudo yum install cuda
```

### Step 3: Clone and Setup

```bash
git clone <repository-url>
cd ehr-sequencing

mamba env create -f environment.yml
mamba activate ehrsequencing
poetry install
```

### Step 4: Verify Installation

```bash
# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check package
python -c "import ehrsequencing; print(ehrsequencing.__version__)"

# Run tests
pytest tests/
```

---

## Development Workflow

### Recommended Directory Structure

```
~/work/
├── ehr-sequencing/          # This project
├── loinc-predictor/         # Related project
├── genai-lab/               # Related project
└── data/                    # Shared data directory
    ├── synthea/
    ├── mimic/
    └── embeddings/
```

### Environment Activation

```bash
# Always activate before working
mamba activate ehrsequencing

# Check active environment
conda env list

# Deactivate when done
conda deactivate
```

### Jupyter Notebook Setup

```bash
# Register kernel
python -m ipykernel install --user --name ehrsequencing --display-name "Python (ehrsequencing)"

# Start Jupyter
jupyter lab

# Or use VS Code with Jupyter extension
```

### Git Configuration

```bash
# Set up Git identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set up SSH key (recommended)
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub  # Add to GitHub
```

---

## IDE Configuration

### VS Code (Recommended)

**Extensions:**
- Python (Microsoft)
- Pylance
- Jupyter
- Black Formatter
- Ruff

**Settings (`.vscode/settings.json`):**

```json
{
  "python.defaultInterpreterPath": "~/miniforge3/envs/ehrsequencing/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

### PyCharm

1. **Set Interpreter:**
   - File → Settings → Project → Python Interpreter
   - Add → Conda Environment → Existing
   - Select: `~/miniforge3/envs/ehrsequencing/bin/python`

2. **Configure Tools:**
   - Black: Settings → Tools → Black
   - Ruff: Settings → Tools → External Tools

3. **Enable Type Checking:**
   - Settings → Editor → Inspections → Python → Type Checker
   - Select: Mypy

---

## Performance Optimization

### M1 MacBook

```python
# Use small configs
from ehrsequencing.models.configs import get_model_config
config = get_model_config('small', 'lstm')

# Enable optimizations
config.mixed_precision = True
config.use_checkpoint = True
config.batch_size = 4
config.gradient_accumulation_steps = 8
```

### Linux with GPU

```python
# Use medium configs
config = get_model_config('medium', 'lstm')

# Enable GPU optimizations
config.mixed_precision = True
config.use_flash_attention = True  # If available
config.batch_size = 32
```

### Memory Profiling

```bash
# Monitor memory usage
python -m memory_profiler scripts/train_disease_progression.py

# PyTorch profiler
python scripts/train_disease_progression.py --profile
```

---

## Troubleshooting

### Conda/Mamba Issues

**Problem: `mamba: command not found`**

```bash
# Reinitialize shell
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

**Problem: Environment conflicts**

```bash
# Remove and recreate
mamba env remove -n ehrsequencing
mamba clean --all
mamba env create -f environment.yml
```

### PyTorch Issues

**Problem: MPS not available on M1 Mac**

```bash
# Update PyTorch
mamba install pytorch::pytorch -c pytorch

# Check version (need 2.0+)
python -c "import torch; print(torch.__version__)"
```

**Problem: CUDA not detected on Linux**

```bash
# Check driver
nvidia-smi

# Reinstall PyTorch with CUDA
mamba install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Import Errors

**Problem: `ModuleNotFoundError: No module named 'ehrsequencing'`**

```bash
# Install in editable mode
pip install -e .

# Verify installation
pip list | grep ehrsequencing
```

### Poetry Issues

**Problem: Poetry lock file conflicts**

```bash
# Remove lock file and reinstall
rm poetry.lock
poetry install
```

---

## Data Management

### Synthetic Data (Synthea)

```bash
# Generate small dataset for testing
cd ~/synthea
./run_synthea -p 100

# Copy to project
cp output/csv/*.csv ~/work/ehr-sequencing/data/synthea/
```

### Data Directory Structure

```
data/
├── synthea/
│   ├── patients.csv
│   ├── encounters.csv
│   ├── conditions.csv
│   ├── observations.csv
│   └── medications.csv
├── mimic/
│   └── (requires access)
└── embeddings/
    └── cehrbert/
        └── pretrained_embeddings.pt
```

---

## Next Steps

After local setup:

1. **Verify installation**: Run `pytest tests/`
2. **Explore notebooks**: `jupyter lab notebooks/`
3. **Test small model**: See [resource-aware-models.md](../../docs/implementation/resource-aware-models.md)
4. **Scale to RunPod**: See [runpod-setup.md](runpod-setup.md) when ready

---

## Related Documentation

- [INSTALL.md](../../INSTALL.md) - Quick start guide
- [RunPod Setup](runpod-setup.md) - Cloud GPU training
- [Pre-trained Models](pretrained-models.md) - CEHR-BERT setup
- [Troubleshooting](troubleshooting.md) - Common issues

---

**Last Updated:** January 20, 2026
