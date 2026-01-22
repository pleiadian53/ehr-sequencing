# Synthea Setup and Data Generation Guide

**Purpose:** Generate synthetic patient data for EHR sequence modeling

---

## Quick Start

⚠️ **IMPORTANT:** Synthea generates FHIR format by default, NOT CSV! You must configure CSV export first.

### 1. Download Synthea

```bash
cd ~/Downloads
curl -LO https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar
```

Or download manually from: https://github.com/synthetichealth/synthea/releases

### 2. Enable CSV Export (CRITICAL STEP)

```bash
# Move JAR to ~/synthea directory
mkdir -p ~/synthea
mv synthea-with-dependencies.jar ~/synthea/
cd ~/synthea

# Create configuration file to enable CSV export
cat > synthea.properties << 'EOF'
exporter.csv.export = true
exporter.fhir.export = false
EOF
```

**Why this is needed:** By default, Synthea only generates FHIR JSON files. The `SyntheaAdapter` requires CSV files, so you must explicitly enable CSV export via the configuration file.

### 3. Generate Data

```bash
# ALWAYS use -c flag to enable CSV export!
# Generate 1000 patients (quick test)
java -jar synthea-with-dependencies.jar -p 1000 -c synthea.properties

# Generate with specific disease module
java -jar synthea-with-dependencies.jar -p 1000 -m diabetes -c synthea.properties

# Generate multiple modules
java -jar synthea-with-dependencies.jar -p 1000 -m "diabetes*,cardiovascular*" -c synthea.properties
```

⚠️ **Common Mistake:** Forgetting the `-c synthea.properties` flag will generate FHIR files instead of CSV!

### 4. Verify CSV Files Created

```bash
# Check that CSV directory exists
ls -lh output/csv/

# Should show:
# observations.csv
# patients.csv
# encounters.csv
# conditions.csv
# medications.csv
# procedures.csv
# etc.
```

If you only see `output/fhir/` directory, you forgot the `-c synthea.properties` flag!

### 5. Copy to Project

```bash
# From ~/synthea directory
# Copy to ehr-sequencing project
cp -r output/csv/* ~/work/ehr-sequencing/data/synthea/

# Or use shared data directory (recommended if you have loinc-predictor)
cp -r output/csv/* ~/work/loinc-predictor/data/synthea/all_cohorts/
```

---

## Shared Data Directory (Recommended)

If you have multiple projects using Synthea data (e.g., `loinc-predictor` and `ehr-sequencing`), you can share the same data directory:

### Option 1: Symlink to Shared Location

```bash
# Create shared data directory in loinc-predictor
mkdir -p ~/work/loinc-predictor/data/synthea/all_cohorts

# Create symlink in ehr-sequencing
cd ~/work/ehr-sequencing
mkdir -p data
ln -s ~/work/loinc-predictor/data/synthea data/synthea

# Verify
ls -la data/synthea
```

### Option 2: Use Absolute Path in Code

```python
# In notebooks or scripts
data_path = Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'all_cohorts'
adapter = SyntheaAdapter(data_path=str(data_path))
```

---

## Detailed Instructions

### Installation Options

#### Option A: Download JAR (Recommended)

```bash
# Create synthea directory
mkdir -p ~/synthea
cd ~/synthea

# Download latest release
curl -LO https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar

# Test installation
java -jar synthea-with-dependencies.jar --help
```

#### Option B: Build from Source

```bash
# Clone repository
git clone https://github.com/synthetichealth/synthea.git
cd synthea

# Build with Gradle
./gradlew build check test

# Run
./run_synthea --help
```

#### Option C: Docker

```bash
# Pull image
docker pull synthetichealth/synthea

# Run
docker run --rm -v $PWD/output:/output synthetichealth/synthea -p 1000
```

---

## Data Generation

### Basic Generation

```bash
# Generate 1000 patients in Massachusetts
java -jar synthea-with-dependencies.jar -p 1000 -c synthea.properties

# Specify state
java -jar synthea-with-dependencies.jar -p 1000 -s California -c synthea.properties

# Specify city
java -jar synthea-with-dependencies.jar -p 1000 -c synthea.properties "San Francisco"
```

### Disease-Specific Cohorts

```bash
# Diabetes
java -jar synthea-with-dependencies.jar -p 5000 -m diabetes -c synthea.properties

# Cardiovascular disease
java -jar synthea-with-dependencies.jar -p 5000 -m cardiovascular_disease -c synthea.properties

# Multiple diseases
java -jar synthea-with-dependencies.jar -p 10000 -m "diabetes*,cardiovascular*,kidney*" -c synthea.properties

# All modules
java -jar synthea-with-dependencies.jar -p 1000 -m "*" -c synthea.properties
```

### Available Disease Modules

Common modules:
- `diabetes` - Type 2 Diabetes
- `cardiovascular_disease` - Heart disease
- `kidney_disease` - Chronic kidney disease
- `metabolic_syndrome` - Metabolic syndrome
- `hepatitis_c` - Hepatitis C
- `hypertension` - High blood pressure
- `copd` - Chronic obstructive pulmonary disease
- `asthma` - Asthma
- `lung_cancer` - Lung cancer
- `colorectal_cancer` - Colorectal cancer

See all modules: https://github.com/synthetichealth/synthea/tree/master/src/main/resources/modules

### Advanced Options

```bash
# Seed for reproducibility
java -jar synthea-with-dependencies.jar -p 1000 -s 12345 -c synthea.properties

# Keep only patients with specific conditions
java -jar synthea-with-dependencies.jar -p 1000 -k "Diabetes" -c synthea.properties

# Age range
java -jar synthea-with-dependencies.jar -p 1000 -a 40-65 -c synthea.properties

# Gender
java -jar synthea-with-dependencies.jar -p 1000 -g M -c synthea.properties  # Male only
```

---

## Configuration

### Enable CSV Export

Edit `synthea.properties`:

```properties
# Enable CSV export
exporter.csv.export = true

# Disable other formats (optional)
exporter.fhir.export = false
exporter.ccda.export = false

# CSV output directory
exporter.baseDirectory = ./output/
```

---

## Output Files

Synthea generates CSV files in `output/csv/`:

### Key Files for EHR Sequencing

1. **patients.csv** - Patient demographics
   - Columns: Id, BIRTHDATE, DEATHDATE, SSN, DRIVERS, PASSPORT, PREFIX, FIRST, LAST, GENDER, RACE, etc.

2. **encounters.csv** - Healthcare encounters
   - Columns: Id, START, STOP, PATIENT, ENCOUNTERCLASS, CODE, DESCRIPTION, REASONCODE, etc.

3. **conditions.csv** - Patient diagnoses
   - Columns: START, STOP, PATIENT, ENCOUNTER, CODE (SNOMED), DESCRIPTION

4. **observations.csv** - Laboratory test results and vital signs
   - Columns: DATE, PATIENT, ENCOUNTER, CODE (LOINC), DESCRIPTION, VALUE, UNITS, TYPE

5. **medications.csv** - Prescribed medications
   - Columns: START, STOP, PATIENT, ENCOUNTER, CODE (RXNORM), DESCRIPTION, REASONCODE

6. **procedures.csv** - Medical procedures
   - Columns: DATE, PATIENT, ENCOUNTER, CODE (SNOMED), DESCRIPTION, REASONCODE

7. **immunizations.csv** - Vaccinations

All files are used by the `SyntheaAdapter` to construct complete patient histories.

---

## Recommended Generation Strategy

### For EHR Sequence Modeling

#### Phase 1: Quick Test (5 minutes)

```bash
# Small dataset for testing pipeline
java -jar synthea-with-dependencies.jar -p 100 -c synthea.properties
```

**Use for:** Testing adapters, visit grouping, sequence building

#### Phase 2: Development Dataset (30 minutes)

```bash
# Medium dataset for model development
java -jar synthea-with-dependencies.jar -p 5000 -c synthea.properties
```

**Use for:** LSTM baseline training, hyperparameter tuning

#### Phase 3: Multi-Cohort (2 hours)

```bash
# Multiple disease cohorts for diverse patterns
java -jar synthea-with-dependencies.jar -p 25000 -m "diabetes*,cardiovascular*,kidney*,metabolic*" -c synthea.properties
```

**Use for:** Med2Vec embedding training, disease trajectory analysis

#### Phase 4: Large-Scale (overnight)

```bash
# Comprehensive dataset for final training
java -jar synthea-with-dependencies.jar -p 100000 -m "*" -c synthea.properties
```

**Use for:** Final model training, benchmarking, publication

---

## Project Integration

### Directory Structure

```
ehr-sequencing/
└── data/
    └── synthea/           # Symlink to shared data or local copy
        ├── test/          # Small test dataset (100 patients)
        ├── dev/           # Development dataset (5K patients)
        └── full/          # Full dataset (100K patients)
```

### Copy Script

Create `scripts/copy_synthea_data.sh`:

```bash
#!/bin/bash
# Copy Synthea output to project

SYNTHEA_DIR=~/synthea/output/csv
PROJECT_DIR=~/work/ehr-sequencing/data/synthea

# Create directories
mkdir -p $PROJECT_DIR/test
mkdir -p $PROJECT_DIR/dev
mkdir -p $PROJECT_DIR/full

# Copy test data
echo "Copying test data..."
cp $SYNTHEA_DIR/*.csv $PROJECT_DIR/test/

echo "✅ Synthea data copied to $PROJECT_DIR"
ls -lh $PROJECT_DIR/test/
```

---

## Validation

### Check Generated Data

```bash
# Count records
wc -l output/csv/patients.csv
wc -l output/csv/encounters.csv
wc -l output/csv/conditions.csv

# Check unique patients
cut -d',' -f2 output/csv/encounters.csv | sort -u | wc -l
```

### Python Validation

```python
import pandas as pd
from pathlib import Path

data_path = Path("output/csv")

# Load key files
patients = pd.read_csv(data_path / "patients.csv")
encounters = pd.read_csv(data_path / "encounters.csv")
conditions = pd.read_csv(data_path / "conditions.csv")
observations = pd.read_csv(data_path / "observations.csv")

print(f"Patients: {len(patients)}")
print(f"Encounters: {len(encounters)}")
print(f"Conditions: {len(conditions)}")
print(f"Observations: {len(observations)}")

print(f"\nEncounters per patient: {len(encounters) / len(patients):.1f}")
print(f"Conditions per patient: {len(conditions) / len(patients):.1f}")

# Check date ranges
encounters['START'] = pd.to_datetime(encounters['START'])
print(f"\nDate range: {encounters['START'].min()} to {encounters['START'].max()}")
```

### Test with SyntheaAdapter

```python
from ehrsequencing.data.adapters import SyntheaAdapter

# Load data
adapter = SyntheaAdapter(data_path="output/csv")

# Load patients
patients = adapter.load_patients()
print(f"Loaded {len(patients)} patients")

# Load encounters
encounters = adapter.load_encounters()
print(f"Loaded {len(encounters)} encounters")

# Test visit grouping
from ehrsequencing.data.visit_grouper import VisitGrouper

grouper = VisitGrouper(adapter=adapter)
patient_id = patients['Id'].iloc[0]
visits = grouper.group_visits(patient_id)
print(f"\nPatient {patient_id[:8]}... has {len(visits)} visits")
```

---

## Troubleshooting

### Java Memory Issues

```bash
# Increase heap size
java -Xmx4g -jar synthea-with-dependencies.jar -p 10000 -c synthea.properties
```

### Slow Generation

```bash
# Reduce history
java -jar synthea-with-dependencies.jar -p 10000 --exporter.years_of_history=5 -c synthea.properties
```

### Problem: Only FHIR files generated, no CSV directory

**Symptom:**
```bash
ls -la output/
# Shows only: fhir/ and metadata/
# Missing: csv/
```

**Solution:**
You forgot to use the `-c synthea.properties` flag! Synthea defaults to FHIR export.

1. Create `synthea.properties` file (see step 2 above)
2. Re-run with `-c` flag:
   ```bash
   java -jar synthea-with-dependencies.jar -p 100 -c synthea.properties
   ```

### Problem: Java not found or version error

**Solution:**
```bash
# macOS
brew install openjdk@21
echo 'export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify
java -version
```

---

## Next Steps

After generating data:

1. **Test the data pipeline:**
   ```bash
   cd ~/work/ehr-sequencing
   mamba activate ehrsequencing
   jupyter notebook notebooks/data-exploration/01_synthea_data_exploration.ipynb
   ```

2. **Train LSTM baseline:**
   ```bash
   python examples/train_lstm_baseline.py \
       --data-path data/synthea/dev \
       --output-dir results/lstm_baseline
   ```

3. **Implement Med2Vec embeddings** (Phase 2)

---

## Resources

- **Synthea Wiki:** https://github.com/synthetichealth/synthea/wiki
- **Module Gallery:** https://synthetichealth.github.io/module-builder/
- **Disease Modules:** https://github.com/synthetichealth/synthea/tree/master/src/main/resources/modules
- **LOINC Browser:** https://loinc.org/
- **SNOMED CT Browser:** https://browser.ihtsdotools.org/

---

## Quick Reference

```bash
# Download
curl -LO https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar

# Setup
mkdir -p ~/synthea && mv synthea-with-dependencies.jar ~/synthea/ && cd ~/synthea
cat > synthea.properties << 'EOF'
exporter.csv.export = true
exporter.fhir.export = false
EOF

# Generate test data
java -jar synthea-with-dependencies.jar -p 100 -c synthea.properties

# Copy to project
cp -r output/csv/* ~/work/ehr-sequencing/data/synthea/test/

# Test
cd ~/work/ehr-sequencing
mamba activate ehrsequencing
jupyter notebook notebooks/data-exploration/01_synthea_data_exploration.ipynb
```

**Happy data generation!** 🎉
