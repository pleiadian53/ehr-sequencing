# Notebook Data Setup Guide

This guide explains how to prepare Synthea datasets for running the notebooks in this project, both locally and on RunPods.

## Overview

The notebooks in this repository require Synthea-generated CSV files to run. Currently, the notebooks reference data from:

```python
data_path = Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'all_cohorts'
```

This is a **cross-project data reference** to avoid duplicating large datasets. However, for **pod deployment**, you'll need to make this data accessible.

---

## Local Development Setup

### Option 1: Use Shared Data from loinc-predictor (Current Approach)

If you have the `loinc-predictor` project with Synthea data already generated:

```bash
# Data should exist at:
~/work/loinc-predictor/data/synthea/all_cohorts/

# Verify it contains:
ls ~/work/loinc-predictor/data/synthea/all_cohorts/
# Should show: patients.csv, encounters.csv, conditions.csv, procedures.csv, etc.
```

**Advantages:**
- No data duplication
- Single source of truth for Synthea data
- Notebooks work immediately

**When to use:**
- Local development
- Multiple projects sharing same datasets
- Datasets are large (>500MB)

### Option 2: Copy Data to This Project

For **self-contained deployment** (e.g., RunPods), copy the data into this project:

```bash
# Create data directory
mkdir -p ~/work/ehr-sequencing/data/synthea/all_cohorts

# Copy from loinc-predictor
cp -r ~/work/loinc-predictor/data/synthea/all_cohorts/* \
  ~/work/ehr-sequencing/data/synthea/all_cohorts/

# Verify
ls ~/work/ehr-sequencing/data/synthea/all_cohorts/
# Should show: patients.csv, encounters.csv, conditions.csv, etc.
```

Then **update the notebooks** to use the local path:

```python
# Change from:
data_path = Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'all_cohorts'

# To:
data_path = Path.home() / 'work' / 'ehr-sequencing' / 'data' / 'synthea' / 'all_cohorts'
```

**Advantages:**
- Self-contained project
- No external dependencies
- Portable to pods

**When to use:**
- Pod deployment
- Project needs to be standalone
- Sharing project with others

### Option 3: Generate New Data

Generate fresh Synthea data specifically for this project. See:
- [Data Generation Guide](../data_generation/data_generation_guide.md)
- [Synthea CSV Export Troubleshooting](../data_generation/synthea_csv_export_troubleshooting.md)

**Quick generation (100 patients for development):**

```bash
# Download Synthea
cd ~/work
wget https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar

# Generate 100 patients
java -jar synthea-with-dependencies.jar -p 100

# Copy to project
mkdir -p ~/work/ehr-sequencing/data/synthea/all_cohorts
cp output/csv/*.csv ~/work/ehr-sequencing/data/synthea/all_cohorts/
```

**Production generation (1000+ patients):**

```bash
# Generate larger cohort
java -jar synthea-with-dependencies.jar -p 1000

# Copy to project
cp output/csv/*.csv ~/work/ehr-sequencing/data/synthea/large_cohort_1000/
```

Then update notebooks to point to the new path.

---

## RunPods Deployment Setup

For running notebooks on a pod, you need to make the data available on the pod. Here are your options:

### Option A: rsync Data from Local to Pod (Recommended)

**Step 1**: Ensure data exists locally (use Option 1 or 2 above)

**Step 2**: Upload to pod

```bash
# If using loinc-predictor data structure (cross-project)
# Upload the entire loinc-predictor data directory
rsync -avzP --no-perms --no-owner --no-group \
  ~/work/loinc-predictor/data/synthea/ \
  runpod-main:/workspace/loinc-predictor/data/synthea/

# If using local ehr-sequencing data (self-contained)
rsync -avzP --no-perms --no-owner --no-group \
  ~/work/ehr-sequencing/data/synthea/ \
  runpod-main:/workspace/ehr-sequencing/data/synthea/
```

**Step 3**: Verify on pod

```bash
# SSH to pod
ssh runpod-main

# Check data exists
ls -lh /workspace/loinc-predictor/data/synthea/all_cohorts/
# or
ls -lh /workspace/ehr-sequencing/data/synthea/all_cohorts/

# Count records
wc -l /workspace/*/data/synthea/all_cohorts/patients.csv
```

### Option B: Generate Data Directly on Pod

Generate Synthea data directly on the pod (useful for large cohorts):

```bash
# SSH to pod
ssh runpod-main

# Install Java (if not already installed)
apt-get update && apt-get install -y default-jre wget

# Download Synthea
cd /workspace
wget https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar

# Generate data
java -jar synthea-with-dependencies.jar -p 1000

# Move to project structure
mkdir -p /workspace/ehr-sequencing/data/synthea/all_cohorts
cp output/csv/*.csv /workspace/ehr-sequencing/data/synthea/all_cohorts/
```

**Advantages:**
- No data transfer needed
- Good for very large datasets (10k+ patients)
- Can generate pod-specific cohorts

**Disadvantages:**
- Takes time (1000 patients ~5-10 min)
- Uses pod hours
- Different data than local development

---

## Data Requirements

### Required CSV Files

All notebooks expect these Synthea CSV files:

- `patients.csv` - Patient demographics
- `encounters.csv` - Healthcare visits
- `conditions.csv` - Diagnoses
- `procedures.csv` - Medical procedures
- `medications.csv` - Prescriptions (optional)
- `observations.csv` - Lab results, vitals (optional)
- `immunizations.csv` - Vaccines (optional)

### Minimum Dataset Size

| Use Case | Patients | Disk Space | Generation Time |
|----------|----------|------------|-----------------|
| Development/Testing | 100 | ~10 MB | 1-2 min |
| Notebook Examples | 100-500 | ~50 MB | 3-5 min |
| Training Models | 1,000+ | ~100 MB | 5-10 min |
| Production | 10,000+ | ~1 GB | 1-2 hours |

### Disk Space Calculation

Approximate space per 100 patients:
- CSV files: ~10 MB
- Processed sequences: ~5 MB
- Model checkpoints: ~50-100 MB

---

## Updating Notebook Data Paths

### Single Notebook Update

To change the data path in a specific notebook, edit the cell:

```python
# Find this line in the notebook:
data_path = Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'all_cohorts'

# Change to your desired path:
data_path = Path.home() / 'work' / 'ehr-sequencing' / 'data' / 'synthea' / 'all_cohorts'
# or for pod:
data_path = Path('/workspace/ehr-sequencing/data/synthea/all_cohorts')
```

### Bulk Update Across All Notebooks

To update all notebooks at once:

```bash
# macOS (BSD sed)
find ~/work/ehr-sequencing/notebooks -name "*.ipynb" -type f -exec sed -i '' \
  's|loinc-predictor/data/synthea|ehr-sequencing/data/synthea|g' {} \;

# Linux (GNU sed)
find ~/work/ehr-sequencing/notebooks -name "*.ipynb" -type f -exec sed -i \
  's|loinc-predictor/data/synthea|ehr-sequencing/data/synthea|g' {} \;
```

### Using Environment Variables (Flexible Approach)

Make notebooks portable by using environment variables:

```python
import os
from pathlib import Path

# Flexible data path - works locally and on pod
data_root = os.getenv('SYNTHEA_DATA_PATH', 
                      str(Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea'))
data_path = Path(data_root) / 'all_cohorts'

print(f"Loading data from: {data_path}")
adapter = SyntheaAdapter(data_path=str(data_path))
```

Then set the environment variable:

```bash
# Local (using loinc-predictor data)
export SYNTHEA_DATA_PATH=~/work/loinc-predictor/data/synthea

# Local (using ehr-sequencing data)
export SYNTHEA_DATA_PATH=~/work/ehr-sequencing/data/synthea

# Pod
export SYNTHEA_DATA_PATH=/workspace/ehr-sequencing/data/synthea
```

---

## Troubleshooting

### Issue: Data Not Found

```
FileNotFoundError: [Errno 2] No such file or directory: '.../patients.csv'
```

**Solution**:
1. Verify data path exists:
   ```bash
   ls ~/work/loinc-predictor/data/synthea/all_cohorts/
   ```
2. Check if data was generated/copied correctly
3. Verify notebook is using correct path

### Issue: Empty CSV Files

```
EmptyDataError: No columns to parse from file
```

**Solution**:
1. Check file sizes:
   ```bash
   du -sh ~/work/loinc-predictor/data/synthea/all_cohorts/*.csv
   ```
2. If files are empty (0 bytes), regenerate data
3. See [Synthea CSV Export Troubleshooting](../data_generation/synthea_csv_export_troubleshooting.md)

### Issue: Data on Pod Not Accessible

```bash
# On pod - check if data exists
ls /workspace/ehr-sequencing/data/synthea/all_cohorts/
# or
ls /workspace/loinc-predictor/data/synthea/all_cohorts/
```

**Solution**:
- Re-sync data with `rsync` (see Option A above)
- Verify paths match notebook expectations
- Check directory permissions

### Issue: Data Too Large for rsync

For datasets >1GB, consider:

1. **Compress before transfer**:
   ```bash
   tar -czf synthea-data.tar.gz ~/work/loinc-predictor/data/synthea/all_cohorts/
   rsync -avzP synthea-data.tar.gz runpod-main:/workspace/
   
   # On pod:
   ssh runpod-main
   cd /workspace
   tar -xzf synthea-data.tar.gz
   ```

2. **Generate directly on pod** (Option B above)

3. **Use persistent pod storage** if available

---

## Quick Reference Commands

```bash
# ========================================
# LOCAL SETUP
# ========================================

# Check if loinc-predictor data exists
ls -lh ~/work/loinc-predictor/data/synthea/all_cohorts/

# Copy to ehr-sequencing (if needed)
mkdir -p ~/work/ehr-sequencing/data/synthea/all_cohorts
cp -r ~/work/loinc-predictor/data/synthea/all_cohorts/* \
  ~/work/ehr-sequencing/data/synthea/all_cohorts/

# ========================================
# POD SETUP
# ========================================

# Upload loinc-predictor data to pod
rsync -avzP --no-perms --no-owner --no-group \
  ~/work/loinc-predictor/data/synthea/ \
  runpod-main:/workspace/loinc-predictor/data/synthea/

# Upload ehr-sequencing data to pod
rsync -avzP --no-perms --no-owner --no-group \
  ~/work/ehr-sequencing/data/synthea/ \
  runpod-main:/workspace/ehr-sequencing/data/synthea/

# Verify on pod
ssh runpod-main "ls -lh /workspace/*/data/synthea/all_cohorts/"

# ========================================
# DATA VALIDATION
# ========================================

# Count patients
wc -l ~/work/loinc-predictor/data/synthea/all_cohorts/patients.csv

# Check dataset size
du -sh ~/work/loinc-predictor/data/synthea/all_cohorts/

# List all CSV files
ls -lh ~/work/loinc-predictor/data/synthea/all_cohorts/*.csv
```

---

## Related Documentation

- [Data Generation Guide](../data_generation/data_generation_guide.md) - Generate new Synthea data
- [Synthea CSV Troubleshooting](../data_generation/synthea_csv_export_troubleshooting.md) - Fix generation issues
- [Notebooks README](../../notebooks/README.md) - Notebook usage guide
- [RunPods Local Development Workflow](../../runpods.example/docs/LOCAL_DEVELOPMENT_WORKFLOW.md) - Pod setup and rsync guide

---

## Summary

**For local development:**
- Use existing `~/work/loinc-predictor/data/synthea/all_cohorts/` (no changes needed)

**For pod deployment:**
- Option 1: `rsync` data from local to pod (fastest)
- Option 2: Generate data directly on pod (for large cohorts)

**Current notebook path:**
```python
data_path = Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'all_cohorts'
```

**Recommended pod path:**
```python
data_path = Path('/workspace/loinc-predictor/data/synthea/all_cohorts')
# or if you copied data to ehr-sequencing:
data_path = Path('/workspace/ehr-sequencing/data/synthea/all_cohorts')
```
