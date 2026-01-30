# Multi-Project Environment Management

## Workspace Overview

This workspace contains multiple independent projects, each with its own conda environment:

| Project | Environment Name | Location |
|---------|-----------------|----------|
| ehr-sequencing | `ehrsequencing` | `/Users/pleiadian53/work/ehr-sequencing` |
| loinc-predictor | `loincpredictor` | `/Users/pleiadian53/work/loinc-predictor` |
| genai-lab | `genailab` | `/Users/pleiadian53/work/genai-lab` |
| causal-bio-lab | (check environment.yml) | `/Users/pleiadian53/work/causal-bio-lab` |
| biographlab | (check environment.yml) | `/Users/pleiadian53/work/biographlab` |

## Critical Rules

### 1. Never Use Base Environment

**NEVER** run project code in the base environment. Each project has isolated dependencies.

### 2. Always Use Correct Environment

Before running any command, verify you're using the correct project's environment:

```bash
# Check project's environment name
cat environment.yml | grep "name:"

# List all environments
mamba env list
```

### 3. Use Explicit Environment Specification

When running commands programmatically (in scripts, CI/CD, or automation):

```bash
# ✅ CORRECT - Explicit environment
mamba run -n ehrsequencing python -m pytest tests/ -v

# ❌ WRONG - Assumes current environment
pytest tests/ -v
```

## Running Commands in Different Projects

### EHR Sequencing

```bash
# Navigate to project
cd ~/work/ehr-sequencing

# Run tests
mamba run -n ehrsequencing python -m pytest tests/ -v

# Run scripts
mamba run -n ehrsequencing python examples/train_model.py
```

### LOINC Predictor

```bash
# Navigate to project
cd ~/work/loinc-predictor

# Run tests
mamba run -n loincpredictor python -m pytest tests/ -v

# Run scripts
mamba run -n loincpredictor python scripts/train.py
```

### GenAI Lab

```bash
# Navigate to project
cd ~/work/genai-lab

# Run tests
mamba run -n genailab python -m pytest tests/ -v

# Run notebooks (after activating)
mamba activate genailab
jupyter notebook
```

## Interactive Work (Terminal Sessions)

For interactive work in a terminal, activate the appropriate environment:

```bash
# For ehr-sequencing
cd ~/work/ehr-sequencing
mamba activate ehrsequencing

# For loinc-predictor
cd ~/work/loinc-predictor
mamba activate loincpredictor

# For genai-lab
cd ~/work/genai-lab
mamba activate genailab
```

**Always verify activation:**

```bash
echo $CONDA_DEFAULT_ENV
which python
```

## Common Mistakes to Avoid

### ❌ Wrong: Assuming Global Availability

```bash
# This will fail if pytest isn't in base environment
cd ~/work/ehr-sequencing
pytest tests/  # ERROR!
```

### ❌ Wrong: Using Wrong Environment

```bash
# Running ehr-sequencing tests with loinc-predictor environment
cd ~/work/ehr-sequencing
mamba run -n loincpredictor python -m pytest tests/  # WRONG!
```

### ✅ Correct: Explicit Environment

```bash
# Always specify the correct environment
cd ~/work/ehr-sequencing
mamba run -n ehrsequencing python -m pytest tests/  # CORRECT!
```

## Environment Verification Checklist

Before running any command:

- [ ] Identified which project I'm working in
- [ ] Checked the project's `environment.yml` for environment name
- [ ] Used `mamba run -n [ENV_NAME]` or activated correct environment
- [ ] Verified environment with `echo $CONDA_DEFAULT_ENV` (if activated)

## Troubleshooting

### "ModuleNotFoundError" in Tests

**Cause:** Running tests in wrong environment or base environment

**Solution:**
```bash
# Check current environment
echo $CONDA_DEFAULT_ENV

# Use correct environment explicitly
cd ~/work/ehr-sequencing
mamba run -n ehrsequencing python -m pytest tests/ -v
```

### "Command not found: pytest"

**Cause:** pytest not installed in current environment

**Solution:**
```bash
# Use python -m pytest instead
mamba run -n ehrsequencing python -m pytest tests/ -v

# Or install pytest in environment
mamba activate ehrsequencing
mamba install pytest
```

### Import Errors Across Projects

**Cause:** Trying to import one project's code in another project's environment

**Solution:** Each project is independent. Don't mix imports across projects unless explicitly designed for it.

## Best Practices

1. **One terminal per project** - Keep separate terminal windows/tabs for each project
2. **Check environment before every command** - Make it a habit
3. **Use `mamba run -n`** - Safer than relying on activation state
4. **Document environment names** - Keep this file updated as projects are added
5. **Never modify base environment** - Keep it minimal

---

**Last Updated:** January 20, 2026
