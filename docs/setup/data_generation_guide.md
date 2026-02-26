# Generating Synthea Data for EHR Sequence Modeling

## Overview

Deep learning models for EHR sequences require substantial training data to learn meaningful patterns. While small datasets (100-200 patients) are useful for rapid prototyping and development, production models typically need:

- **Minimum**: 1,000+ patients for basic performance
- **Recommended**: 5,000-10,000 patients for robust models
- **Optimal**: 50,000+ patients for state-of-the-art results

This guide shows how to generate synthetic EHR data at scale using Synthea, a realistic patient generator that creates complete medical histories with encounters, conditions, procedures, medications, and observations.

## Option 1: Generate New Synthea Data (Recommended)

### Install Synthea

```bash
# Download Synthea
cd ~/work/data
wget https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar

# Or clone and build from source
git clone https://github.com/synthetichealth/synthea.git
cd synthea
./gradlew build check test
```

### Generate 1000+ Patients

```bash
# Generate 1000 patients
java -jar synthea-with-dependencies.jar -p 1000

# Output will be in ./output/csv/
# Move to your data directory
mv output/csv ~/work/loinc-predictor/data/synthea/large_cohort/
```

### Generate with Specific Conditions

For disease progression modeling, generate patients with specific conditions:

```bash
# CKD patients
java -jar synthea-with-dependencies.jar \
  -p 500 \
  -m chronic_kidney_disease

# Diabetes patients
java -jar synthea-with-dependencies.jar \
  -p 500 \
  -m diabetes

# Combine multiple cohorts
mkdir ~/work/loinc-predictor/data/synthea/combined_1000/
cat large_cohort/patients.csv > combined_1000/patients.csv
cat large_cohort/encounters.csv > combined_1000/encounters.csv
# ... repeat for other files
```

### Configuration Options

Edit `src/main/resources/synthea.properties`:

```properties
# Generate more realistic data
exporter.years_of_history = 10

# Include more conditions
generate.only_alive_patients = false
generate.append_numbers_to_person_names = true

# Increase prevalence of chronic conditions
generate.chronic_kidney_disease.prevalence = 0.15
generate.diabetes.prevalence = 0.20
```

## Option 2: Use Public Synthea Datasets

### SyntheticMass Dataset

Large pre-generated Synthea dataset:

```bash
# Download SyntheticMass (1M+ patients)
wget https://synthea.mitre.org/downloads/synthea_sample_data_csv_apr2020.zip

# Extract specific subset
unzip synthea_sample_data_csv_apr2020.zip
head -n 1001 csv/patients.csv > subset_1000/patients.csv
# Filter other files by patient IDs
```

### MITRE Synthea Downloads

- https://synthea.mitre.org/downloads
- Pre-generated datasets available
- Various sizes and configurations

## Option 3: Use Real De-identified EHR Data

If available, use real de-identified data:

- MIMIC-III/IV (ICU data)
- eICU Collaborative Research Database
- UK Biobank
- All of Us Research Program

**Advantages**:
- Real clinical patterns
- Better generalization
- Meaningful outcomes

**Requirements**:
- IRB approval
- Data use agreements
- Privacy compliance

## Updating the Training Pipeline

Once you have more data, update the data path:

```python
# In notebook or training script
data_dir = Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'large_cohort'

# Or combine multiple cohorts
data_dirs = [
    Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'cohort1',
    Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'cohort2',
    Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'cohort3',
]

all_events = []
for data_dir in data_dirs:
    adapter = SyntheaAdapter(data_dir)
    events = adapter.load_events()
    all_events.extend(events)
```

## Expected Performance by Dataset Size

The relationship between dataset size and model performance for survival analysis:

| Dataset Size | Expected C-index | Training Time | Use Case |
|--------------|------------------|---------------|----------|
| 100-200 patients | 0.45-0.55 | 5-10 min | Development/debugging |
| 500 patients | 0.55-0.65 | 20-30 min | Initial experiments |
| 1,000 patients | 0.60-0.70 | 45-60 min | Baseline models |
| 5,000 patients | 0.65-0.75 | 3-4 hours | Production models |
| 10,000+ patients | 0.70-0.80 | 8-10 hours | State-of-the-art |

**Notes**:
- Performance estimates assume well-defined outcomes and sufficient event rates
- With pretrained embeddings, expect +0.05-0.10 improvement in C-index
- Training time varies by hardware (estimates for single GPU)

## Next Steps

1. **Generate/download more data** (this guide)
2. **Implement pretrained embeddings** (Phase 2)
3. **Retrain with larger dataset**
4. **Evaluate performance improvement**

## Resources

- Synthea GitHub: https://github.com/synthetichealth/synthea
- Synthea Wiki: https://github.com/synthetichealth/synthea/wiki
- SyntheticMass: https://synthea.mitre.org/downloads
- Synthea Module Builder: https://synthetichealth.github.io/module-builder/
