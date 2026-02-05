# Installation Fix for SSL Certificate Issues

## Problem
SSL certificate verification errors prevent pip from downloading packages from PyPI, blocking `pip install -e .`

## Root Cause
The `pyproject.toml` specifies dependencies that pip tries to download, but SSL verification fails.

## Solution: Use Conda for All Dependencies

### Step 1: Update environment.yml
Move all packages from pip section to conda section (already done).

### Step 2: Update/recreate environment
```bash
# Update existing environment
mamba env update -f environment.yml --prune

# Or recreate from scratch
mamba env remove -n ehrsequencing
mamba env create -f environment.yml
```

### Step 3: Activate environment
```bash
mamba activate ehrsequencing
```

### Step 4: Install package without dependencies
```bash
# Install package in editable mode, but don't install dependencies
# (they're already in conda environment)
pip install --no-deps -e .
```

### Step 5: Verify installation
```bash
# Test import
python -c "import ehrsequencing; print('Success!')"

# Run tests
python -m pytest tests/test_data_pipeline.py -v
```

## Why This Works

1. **Conda bypasses SSL issues** - Uses its own certificate handling
2. **`--no-deps` flag** - Tells pip not to install dependencies
3. **setup.py with empty install_requires** - No dependencies to download
4. **All packages from conda-forge** - Reliable, pre-compiled binaries

## Commands to Run Now

```bash
# 1. Update environment with new packages
mamba env update -f environment.yml --prune

# 2. Install package without dependencies
pip install --no-deps -e .

# 3. Run tests
python -m pytest tests/test_data_pipeline.py -v
```

## If Still Having Issues

### Check what's missing:
```bash
python -c "import ehrsequencing.data; print('Data module OK')"
```

### Install missing packages individually:
```bash
mamba install lifelines umap-learn gensim biopython tqdm pytest
```

---

**Last Updated:** January 20, 2026
