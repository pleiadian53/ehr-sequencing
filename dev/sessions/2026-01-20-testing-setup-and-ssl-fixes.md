# Testing Setup and SSL Certificate Fixes

**Date:** January 20, 2026  
**Focus:** Resolving installation issues and getting all tests passing

---

## Session Summary

Successfully resolved SSL certificate errors blocking package installation and got all 16 data pipeline tests passing with 76% code coverage.

---

## Problems Encountered

### 1. SSL Certificate Verification Errors

**Error:**
```
SSLError(SSLCertVerificationError('"pypi.org" certificate is not trusted'))
ERROR: Could not find a version that satisfies the requirement poetry-core
ERROR: Could not find a version that satisfies the requirement lifelines
```

**Root Cause:**
- Network/system SSL certificate verification failing
- pip unable to download packages from PyPI
- Poetry build backend requiring `poetry-core` from PyPI

### 2. Package Not Installed

**Error:**
```
ModuleNotFoundError: No module named 'ehrsequencing'
```

**Cause:** Package not installed in editable mode in environment

### 3. SyntheaAdapter Initialization Bug

**Error:**
```
AttributeError: 'SyntheaAdapter' object has no attribute 'data_dir'
```

**Cause:** `super().__init__()` called before setting `self.data_dir`, but parent's `__init__` calls `_validate_data_path()` which needs `data_dir`

### 4. Empty CSV Mock Data

**Error:**
```
pandas.errors.EmptyDataError: No columns to parse from file
```

**Cause:** Test fixture created completely empty CSV files without headers

---

## Solutions Implemented

### Solution 1: Move Dependencies to Conda

**File:** `environment.yml`

Moved packages from pip section to conda section to bypass SSL:
- `lifelines>=0.27.0` → conda
- `umap-learn>=0.5.0` → conda  
- `gensim>=4.3.0` → conda
- `biopython>=1.81` → conda
- `tqdm>=4.66.0` → conda
- `pytest`, `black`, `ruff`, `mypy` → conda

Only `transformers` remains in pip (not available in conda-forge)

### Solution 2: Create setup.py Fallback

**File:** `setup.py`

Created simple setuptools-based installer with empty `install_requires`:
```python
setup(
    name="ehrsequencing",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],  # Empty - all deps from conda
)
```

### Solution 3: Install with Special Flags

**Command:**
```bash
mamba run -n ehrsequencing pip install --no-build-isolation --no-deps -e .
```

**Flags:**
- `--no-build-isolation`: Don't create isolated build environment (avoids downloading poetry-core)
- `--no-deps`: Don't install dependencies (already in conda)
- `-e`: Editable install

### Solution 4: Fix SyntheaAdapter Initialization

**File:** `src/ehrsequencing/data/adapters/synthea.py`

**Before:**
```python
def __init__(self, data_path: str, **kwargs):
    super().__init__(data_path, **kwargs)  # Calls _validate_data_path()
    self.data_dir = Path(data_path)  # Too late!
```

**After:**
```python
def __init__(self, data_path: str, **kwargs):
    self.data_dir = Path(data_path)  # Set first
    self._cache = {}
    super().__init__(data_path, **kwargs)  # Now _validate_data_path() works
```

### Solution 5: Fix Test Mock Data

**File:** `tests/test_data_pipeline.py`

**Before:**
```python
# Create empty files
for file in ['observations.csv', 'medications.csv', 'procedures.csv']:
    pd.DataFrame().to_csv(tmpdir / file, index=False)  # No columns!
```

**After:**
```python
# Create empty DataFrames with proper headers
observations_df = pd.DataFrame(columns=[
    'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'DATE', 
    'VALUE', 'UNITS', 'TYPE'
])
observations_df.to_csv(tmpdir / 'observations.csv', index=False)
# ... same for medications and procedures
```

---

## Final Working Commands

```bash
# 1. Update environment (if needed)
mamba env update -f environment.yml --prune

# 2. Activate environment
mamba activate ehrsequencing

# 3. Install package
pip install --no-build-isolation --no-deps -e .

# 4. Run tests
python -m pytest tests/test_data_pipeline.py -v
```

**Or using mamba run (without activation):**
```bash
mamba run -n ehrsequencing pip install --no-build-isolation --no-deps -e .
mamba run -n ehrsequencing python -m pytest tests/test_data_pipeline.py -v
```

---

## Test Results

```
============================= test session starts ==============================
collected 16 items

tests/test_data_pipeline.py::TestMedicalEvent::test_create_valid_event PASSED
tests/test_data_pipeline.py::TestMedicalEvent::test_event_requires_patient_id PASSED
tests/test_data_pipeline.py::TestMedicalEvent::test_event_requires_datetime PASSED
tests/test_data_pipeline.py::TestVisitGrouper::test_group_by_encounter PASSED
tests/test_data_pipeline.py::TestVisitGrouper::test_group_by_same_day PASSED
tests/test_data_pipeline.py::TestVisitGrouper::test_semantic_code_ordering PASSED
tests/test_data_pipeline.py::TestVisitGrouper::test_hybrid_strategy PASSED
tests/test_data_pipeline.py::TestPatientSequenceBuilder::test_build_vocabulary PASSED
tests/test_data_pipeline.py::TestPatientSequenceBuilder::test_build_sequences PASSED
tests/test_data_pipeline.py::TestPatientSequenceBuilder::test_encode_sequence PASSED
tests/test_data_pipeline.py::TestPatientSequenceBuilder::test_time_deltas PASSED
tests/test_data_pipeline.py::TestPatientSequenceBuilder::test_create_dataset PASSED
tests/test_data_pipeline.py::TestSyntheaAdapter::test_adapter_initialization PASSED
tests/test_data_pipeline.py::TestSyntheaAdapter::test_load_patients PASSED
tests/test_data_pipeline.py::TestSyntheaAdapter::test_load_events PASSED
tests/test_data_pipeline.py::TestSyntheaAdapter::test_get_statistics PASSED

============================== 16 passed in 2.69s ==============================

Coverage: 76%
```

---

## Files Created/Modified

### Created
- `setup.py` - Fallback installer without Poetry
- `dev/testing/COMMON_ERRORS.md` - Error documentation
- `dev/testing/INSTALLATION_FIX.md` - Installation guide
- `dev/testing/README.md` - Testing documentation index

### Modified
- `environment.yml` - Moved packages from pip to conda
- `src/ehrsequencing/data/adapters/synthea.py` - Fixed initialization order
- `tests/test_data_pipeline.py` - Fixed mock data with proper headers

---

## Key Learnings

1. **SSL issues are common** - Always have conda fallback for dependencies
2. **Initialization order matters** - Set attributes before calling `super().__init__()`
3. **Empty DataFrames need columns** - pandas can't parse truly empty CSV files
4. **Mamba + pip hybrid works** - Use conda for heavy packages, pip for pure Python
5. **`--no-build-isolation` bypasses Poetry** - Useful when SSL blocks build dependencies

---

## Documentation Created

- `dev/testing/COMMON_ERRORS.md` - 8 common errors with solutions
- `dev/testing/INSTALLATION_FIX.md` - Step-by-step SSL fix guide
- `dev/testing/README.md` - Testing documentation index
- `dev/sessions/` - Organized session notes by date
- `dev/workflow/MULTI_PROJECT_ENVIRONMENTS.md` - Multi-project environment guide
- `dev/workflow/TESTING_WORKFLOW.md` - Complete testing workflow

---

**Session Duration:** ~1.5 hours  
**Issues Resolved:** 5  
**Tests Passing:** 16/16 (100%)  
**Code Coverage:** 76%

---

**Status:** ✅ All tests passing, data pipeline fully functional  
**Next Steps:** Implement LSTM baseline model
