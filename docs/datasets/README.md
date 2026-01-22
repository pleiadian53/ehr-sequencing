# Dataset Documentation

This directory contains guides for obtaining and using datasets with the EHR sequencing project.

## Available Guides

### Synthea Synthetic Data

- **[SYNTHEA_SETUP.md](SYNTHEA_SETUP.md)** - Complete guide for generating synthetic patient data
  - Installation instructions
  - Data generation commands
  - Configuration options
  - Troubleshooting

## Quick Start

### Using Shared Data (Recommended)

If you have the `loinc-predictor` project with Synthea data already generated:

```python
from pathlib import Path
from ehrsequencing.data.adapters import SyntheaAdapter

# Use shared data directory
data_path = Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'all_cohorts'
adapter = SyntheaAdapter(data_path=str(data_path))
```

### Generating New Data

```bash
# See SYNTHEA_SETUP.md for detailed instructions
cd ~/synthea
java -jar synthea-with-dependencies.jar -p 1000 -c synthea.properties
cp -r output/csv/* ~/work/ehr-sequencing/data/synthea/
```

## Data Requirements

The `SyntheaAdapter` requires the following CSV files:

- `patients.csv` - Patient demographics
- `encounters.csv` - Healthcare encounters
- `conditions.csv` - Diagnoses (SNOMED codes)
- `observations.csv` - Lab results and vitals (LOINC codes)
- `medications.csv` - Prescriptions (RxNorm codes)
- `procedures.csv` - Medical procedures (SNOMED codes)

## Dataset Sizes

Recommended dataset sizes for different purposes:

| Purpose | Patients | Generation Time | Disk Space |
|---------|----------|-----------------|------------|
| Testing | 100 | 5 min | ~50 MB |
| Development | 5,000 | 30 min | ~2 GB |
| Training | 25,000 | 2 hours | ~10 GB |
| Large-scale | 100,000 | Overnight | ~40 GB |

## Data Sharing Between Projects

Both `ehr-sequencing` and `loinc-predictor` use the same `SyntheaAdapter` and can share data:

### Option 1: Symlink

```bash
cd ~/work/ehr-sequencing
mkdir -p data
ln -s ~/work/loinc-predictor/data/synthea data/synthea
```

### Option 2: Absolute Path

Use absolute paths in your code to reference the shared location.

### Option 3: Copy

Copy data to each project's `data/synthea/` directory (uses more disk space).

## Related Documentation

- **Data Pipeline:** `../implementation/visit-grouped-sequences.md`
- **Data Exploration:** `../../notebooks/data-exploration/README.md`
- **Variable-Length Sequences:** `../methods/variable-length-sequences.md`

## External Resources

- **Synthea Project:** https://github.com/synthetichealth/synthea
- **Synthea Wiki:** https://github.com/synthetichealth/synthea/wiki
- **Module Gallery:** https://synthetichealth.github.io/module-builder/
- **LOINC Codes:** https://loinc.org/
- **SNOMED CT:** https://browser.ihtsdotools.org/
