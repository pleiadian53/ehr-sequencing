# Testing Guide

This directory contains unit tests for the EHR Sequencing project.

## Running Tests

### Prerequisites

**IMPORTANT:** Always activate the `ehrsequencing` environment first. Never run tests in the base environment.

```bash
# Activate the environment (REQUIRED)
mamba activate ehrsequencing

# Verify you're in the correct environment
which python
# Should show: /Users/pleiadian53/mambaforge/envs/ehrsequencing/bin/python
```

### Run All Tests

```bash
# From project root
pytest tests/ -v

# Or with coverage report
pytest tests/ -v --cov=ehrsequencing --cov-report=html
```

### Run Specific Test File

```bash
# Run data pipeline tests
pytest tests/test_data_pipeline.py -v

# Run with detailed output
pytest tests/test_data_pipeline.py -vv
```

### Run Specific Test Class or Method

```bash
# Run specific test class
pytest tests/test_data_pipeline.py::TestVisitGrouper -v

# Run specific test method
pytest tests/test_data_pipeline.py::TestVisitGrouper::test_group_by_encounter -v
```

### Run Tests with Markers

```bash
# Run only fast tests (if marked)
pytest tests/ -v -m "not slow"

# Run only integration tests (if marked)
pytest tests/ -v -m "integration"
```

### Useful pytest Options

- `-v` - Verbose output (show test names)
- `-vv` - Very verbose (show full diffs)
- `-s` - Show print statements
- `-x` - Stop on first failure
- `--lf` - Run last failed tests
- `--ff` - Run failed tests first, then others
- `-k EXPRESSION` - Run tests matching expression (e.g., `-k "synthea"`)

## Test Structure

```
tests/
├── README.md                    # This file
├── test_data_pipeline.py        # Data loading and preprocessing tests
├── test_embeddings.py           # Code embedding tests (coming soon)
├── test_models.py               # Model tests (coming soon)
└── conftest.py                  # Shared fixtures (coming soon)
```

## Writing Tests

### Test Organization

Each test file should:
1. Import necessary modules
2. Define test classes for logical grouping
3. Use descriptive test method names
4. Include docstrings explaining what is tested

### Example Test Structure

```python
import pytest
from ehrsequencing.data import SyntheaAdapter

class TestSyntheaAdapter:
    """Test Synthea data adapter."""
    
    @pytest.fixture
    def sample_data_dir(self):
        """Create mock Synthea data directory."""
        # Setup code
        yield data_dir
        # Teardown code
    
    def test_load_patients(self, sample_data_dir):
        """Test loading patient demographics."""
        adapter = SyntheaAdapter(sample_data_dir)
        patients = adapter.load_patients()
        assert len(patients) > 0
```

## Current Test Coverage

### Data Pipeline (`test_data_pipeline.py`)

- ✅ `TestMedicalEvent` - Event creation and validation
- ✅ `TestVisitGrouper` - All grouping strategies
- ✅ `TestPatientSequenceBuilder` - Vocabulary, encoding, datasets
- ✅ `TestSyntheaAdapter` - Data loading with mock data

### Coming Soon

- ⬜ `test_embeddings.py` - Code embedding tests
- ⬜ `test_models.py` - LSTM and Transformer tests
- ⬜ `test_evaluation.py` - Metrics and evaluation tests

## Continuous Integration

Tests are automatically run on:
- Every commit (via pre-commit hooks)
- Pull requests (via GitHub Actions)
- Scheduled nightly builds

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Ensure package is installed in editable mode
poetry install

# Or with pip
pip install -e .
```

### Missing Dependencies

```bash
# Install test dependencies
poetry install --with dev

# Or manually
pip install pytest pytest-cov
```

### Test Data Issues

Some tests use mock data. If tests fail due to missing data:
- Check that `tempfile` is working correctly
- Ensure sufficient disk space for temporary files

---

**Last Updated:** January 20, 2026
