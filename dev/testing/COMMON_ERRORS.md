# Common Testing Errors and Solutions

This document tracks common errors encountered during testing and their solutions.

---

## ✅ Working Solution Summary

**Problem**: SSL certificate errors prevented package installation and testing.

**Complete Solution**:
```bash
# 1. Update environment with all dependencies via conda (bypasses SSL)
mamba env update -f environment.yml --prune

# 2. Install package without build isolation or dependencies
mamba run -n ehrsequencing pip install --no-build-isolation --no-deps -e .

# 3. Run tests
mamba run -n ehrsequencing python -m pytest tests/test_data_pipeline.py -v
```

**Result**: All 16 tests passing with 76% code coverage.

**Key Changes Made**:
1. Moved `lifelines`, `umap-learn`, `gensim`, etc. from pip to conda in `environment.yml`
2. Created `setup.py` with empty `install_requires` as fallback
3. Used `--no-build-isolation --no-deps` flags to bypass Poetry and SSL issues
4. Fixed `SyntheaAdapter.__init__()` initialization order bug
5. Fixed test mock data to include proper CSV headers

---

## Error 1: ModuleNotFoundError: No module named 'ehrsequencing'

### Error Message
```
ImportError while importing test module '/Users/pleiadian53/work/ehr-sequencing/tests/test_data_pipeline.py'.
...
E   ModuleNotFoundError: No module named 'ehrsequencing'
```

### Cause
The `ehrsequencing` package is not installed in editable mode in the environment. Tests try to import from `ehrsequencing.data`, but Python can't find the package.

### Solution

**Option 1: Install with pip (Recommended)**
```bash
# Activate environment
mamba activate ehrsequencing

# Install in editable mode
pip install -e .
```

**Option 2: Install with poetry**
```bash
# Activate environment
mamba activate ehrsequencing

# Install with poetry
poetry install
```

**Option 3: Using mamba run (without activation)**
```bash
# From project root
mamba run -n ehrsequencing pip install -e .
```

### Verification
```bash
# Activate environment
mamba activate ehrsequencing

# Test import
python -c "import ehrsequencing; print(ehrsequencing.__version__)"
# Should output: 0.1.0

# Run tests
python -m pytest tests/test_data_pipeline.py -v
```

### Prevention
Always install the package in editable mode after:
- Creating a new environment
- Cloning the repository
- Switching to a different machine

---

## Error 2: SSL Certificate Error During pip install

### Error Message
```
SSLError(SSLCertVerificationError('"pypi.org" certificate is not trusted'))
ERROR: Could not find a version that satisfies the requirement poetry-core
```

### Cause
SSL certificate verification is failing when pip tries to connect to PyPI. This can happen due to:
- Corporate proxy/firewall
- Outdated certificates
- Network configuration issues

### Solution

**Option 1: Install without poetry (Recommended for this case)**
```bash
# Activate environment
mamba activate ehrsequencing

# Install dependencies directly from environment.yml (already done)
# Then install package without build dependencies
pip install --no-build-isolation -e .
```

**Option 2: Temporarily disable SSL verification (Use with caution)**
```bash
# Only use on trusted networks
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -e .
```

**Option 3: Fix SSL certificates**
```bash
# On macOS, install certificates
/Applications/Python\ 3.10/Install\ Certificates.command

# Or update certifi
pip install --upgrade certifi
```

**Option 4: Use conda/mamba instead**
```bash
# Install dependencies from environment.yml (already done)
# Package is already accessible via PYTHONPATH if src/ structure is correct
```

### Verification
```bash
# Test if package is importable
python -c "import sys; sys.path.insert(0, 'src'); import ehrsequencing; print('Success')"
```

---

## Error 3: pytest: command not found

### Error Message
```
zsh:1: command not found: pytest
```

### Cause
pytest is not installed in the current environment, or you're in the base environment.

### Solution

**Option 1: Use python -m pytest**
```bash
python -m pytest tests/ -v
```

**Option 2: Install pytest**
```bash
mamba activate ehrsequencing
mamba install pytest pytest-cov
```

**Option 3: Use mamba run**
```bash
mamba run -n ehrsequencing python -m pytest tests/ -v
```

---

## Error 3: Wrong environment activated

### Error Message
```
ModuleNotFoundError: No module named 'ehrsequencing'
# Even after installing the package
```

### Cause
You're in the wrong environment (e.g., base, or another project's environment).

### Solution
```bash
# Check current environment
echo $CONDA_DEFAULT_ENV

# If wrong, deactivate
mamba deactivate

# Activate correct environment
mamba activate ehrsequencing

# Verify
which python
# Should show: .../envs/ehrsequencing/bin/python
```

---

## Error 4: Import errors for specific modules

### Error Message
```
ImportError: cannot import name 'MedicalEvent' from 'ehrsequencing.data'
```

### Cause
- Module not properly exported in `__init__.py`
- Circular import
- File not saved

### Solution

**Check exports in `__init__.py`:**
```python
# src/ehrsequencing/data/__init__.py should have:
from .adapters import BaseEHRAdapter, MedicalEvent, PatientInfo, SyntheaAdapter
from .visit_grouper import Visit, VisitGrouper
from .sequence_builder import PatientSequence, PatientSequenceBuilder

__all__ = [
    'BaseEHRAdapter',
    'MedicalEvent',
    'PatientInfo',
    'SyntheaAdapter',
    'Visit',
    'VisitGrouper',
    'PatientSequence',
    'PatientSequenceBuilder'
]
```

**Reinstall package:**
```bash
mamba activate ehrsequencing
pip install -e . --force-reinstall --no-deps
```

---

## Error 5: Test collection errors

### Error Message
```
ERROR collecting tests/test_data_pipeline.py
```

### Cause
- Syntax errors in test file
- Missing `__init__.py` in tests directory
- Invalid test class/method names

### Solution

**Check test file structure:**
```python
# Tests must:
# 1. Be in files named test_*.py or *_test.py
# 2. Have classes named Test*
# 3. Have methods named test_*

class TestMedicalEvent:  # ✅ Correct
    def test_create_event(self):  # ✅ Correct
        pass
```

**Add `__init__.py`:**
```bash
touch tests/__init__.py
```

---

## Error 6: Fixture not found

### Error Message
```
fixture 'sample_events' not found
```

### Cause
Fixture defined in wrong scope or not imported.

### Solution

**Define fixture in same file or conftest.py:**
```python
# In test file
@pytest.fixture
def sample_events():
    return [...]

# Or in tests/conftest.py for shared fixtures
```

---

## Error 7: pandas/numpy version conflicts

### Error Message
```
ImportError: numpy.dtype size changed, may indicate binary incompatibility
```

### Cause
Version mismatch between numpy and pandas.

### Solution
```bash
mamba activate ehrsequencing
mamba update numpy pandas
```

---

## Error 8: CUDA/MPS errors on Mac

### Error Message
```
RuntimeError: MPS backend out of memory
```

### Cause
Using GPU-intensive operations on M1 Mac with limited memory.

### Solution

**Use CPU-only for testing:**
```python
import torch
device = torch.device('cpu')  # Force CPU for tests
```

**Or reduce batch size/data size in tests:**
```python
# Use smaller datasets for testing
patients = adapter.load_patients(limit=10)  # Instead of 1000
```

---

## Best Practices to Avoid Errors

1. **Always activate environment first**
   ```bash
   mamba activate ehrsequencing
   ```

2. **Install package in editable mode**
   ```bash
   pip install -e .
   ```

3. **Use `python -m pytest` instead of `pytest`**
   ```bash
   python -m pytest tests/ -v
   ```

4. **Verify environment before running tests**
   ```bash
   echo $CONDA_DEFAULT_ENV
   which python
   ```

5. **Keep dependencies updated**
   ```bash
   mamba update --all
   ```

6. **Use mock data in tests**
   - Avoid loading large datasets
   - Use `tempfile` for temporary files
   - Clean up after tests

---

**Last Updated:** January 20, 2026

**Contributing:** When you encounter a new error, add it to this document with:
- Error message
- Cause
- Solution
- Prevention tips
