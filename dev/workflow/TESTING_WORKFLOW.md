# Testing Workflow for EHR Sequencing

## Environment Activation (CRITICAL)

**NEVER run tests in the base environment or outside a virtual environment.**

### Step-by-Step Testing Process

#### 1. Activate Environment

```bash
# Navigate to project directory
cd ~/work/ehr-sequencing

# Activate the ehrsequencing environment
mamba activate ehrsequencing

# Verify activation
echo $CONDA_DEFAULT_ENV
# Should output: ehrsequencing

# Verify Python path
which python
# Should show: /Users/pleiadian53/mambaforge/envs/ehrsequencing/bin/python
```

#### 2. Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_data_pipeline.py -v

# Run with coverage
python -m pytest tests/ -v --cov=ehrsequencing --cov-report=html

# Run specific test class
python -m pytest tests/test_data_pipeline.py::TestVisitGrouper -v

# Run specific test method
python -m pytest tests/test_data_pipeline.py::TestVisitGrouper::test_group_by_encounter -v
```

#### 3. Alternative: Using mamba run

If you can't activate the environment in your current shell:

```bash
# Run tests without activating
mamba run -n ehrsequencing python -m pytest tests/test_data_pipeline.py -v

# With coverage
mamba run -n ehrsequencing python -m pytest tests/ -v --cov=ehrsequencing
```

## Quick Reference Commands

### Environment Management

```bash
# List all environments
mamba env list

# Check current environment
echo $CONDA_DEFAULT_ENV

# Deactivate current environment
mamba deactivate

# Activate ehrsequencing
mamba activate ehrsequencing
```

### Common Test Commands

```bash
# Fast: Run only failed tests from last run
python -m pytest --lf -v

# Debug: Stop on first failure
python -m pytest -x -v

# Verbose: Show print statements
python -m pytest -s -v

# Filter: Run tests matching pattern
python -m pytest -k "synthea" -v
```

## Pre-commit Testing

Before committing code, always run:

```bash
# 1. Activate environment
mamba activate ehrsequencing

# 2. Run all tests
python -m pytest tests/ -v

# 3. Check code formatting
black src/ tests/
ruff check src/ tests/

# 4. Type checking (optional)
mypy src/
```

## Continuous Integration

Tests run automatically on:
- Every push to main branch
- Every pull request
- Scheduled nightly builds

## Troubleshooting

### "ModuleNotFoundError: No module named 'ehrsequencing'"

**Solution:** Install package in editable mode

```bash
mamba activate ehrsequencing
poetry install
# or
pip install -e .
```

### "pytest: command not found"

**Solution:** Use `python -m pytest` instead of `pytest`

```bash
python -m pytest tests/ -v
```

### "Wrong environment activated"

**Solution:** Check and reactivate

```bash
# Check current environment
echo $CONDA_DEFAULT_ENV

# Deactivate if wrong
mamba deactivate

# Activate correct environment
mamba activate ehrsequencing
```

### "Import errors in tests"

**Solution:** Ensure all dependencies are installed

```bash
mamba activate ehrsequencing
poetry install --with dev
```

## Best Practices

1. **Always activate environment first** - Never skip this step
2. **Run tests before committing** - Catch issues early
3. **Write tests for new features** - Maintain coverage
4. **Use descriptive test names** - Make failures easy to understand
5. **Keep tests fast** - Use mock data when possible
6. **Test edge cases** - Don't just test happy paths

---

**Last Updated:** January 20, 2026
