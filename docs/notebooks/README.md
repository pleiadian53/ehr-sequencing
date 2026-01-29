# EHR Sequencing Notebooks

Educational notebooks for learning EHR sequence modeling concepts and exploring the `ehrsequencing` package.

## Purpose

These notebooks serve as:
- **Educational tutorials** for understanding EHR data processing and modeling
- **Exploratory analysis** of Synthea synthetic EHR data
- **Step-by-step guides** for the complete modeling pipeline
- **Reference implementations** for common tasks

## Structure

Notebooks are organized by topic into subdirectories:

```
notebooks/
├── README.md (this file)
├── 01_synthea_data_exploration/
│   ├── 01_synthea_data_exploration.ipynb
│   ├── 01a_lstm_data_preparation.ipynb
│   ├── data_shape_transformations.md
│   └── README.md
├── 02_survival_analysis/
│   ├── 01_discrete_time_survival_lstm.ipynb
│   ├── validate_survival_model.py
│   └── README.md
└── (future topics)/
```

## Available Notebooks

### 01. Synthea Data Exploration

**Directory:** `01_synthea_data_exploration/`

Comprehensive exploration of Synthea synthetic EHR data and preparation for modeling.

#### Notebooks:

1. **`01_synthea_data_exploration.ipynb`**
   - Load and explore Synthea CSV files
   - Understand patient demographics and clinical data
   - Group events into clinical visits
   - Build patient sequences
   - Compute data statistics and quality metrics
   - Visualize temporal patterns

2. **`01a_lstm_data_preparation.ipynb`**
   - Prepare visit-grouped sequences for LSTM model
   - Encode sequences to integer IDs
   - Create labels for prediction tasks (diabetes detection)
   - Batch data with proper padding and masking
   - Visualize data shape transformations
   - Test LSTM model forward pass

#### Documentation:

- **`data_shape_transformations.md`**: Comprehensive reference for all data shape transformations from raw CSV to model predictions

### 02. Survival Analysis

**Directory:** `02_survival_analysis/`

Discrete-time survival analysis for disease progression modeling using LSTMs.

#### Notebooks:

1. **`01_discrete_time_survival_lstm.ipynb`**
   - Understanding the C-index (concordance index)
   - Research questions and clinical applications
   - Data labeling strategies for survival outcomes
   - Synthetic survival outcome generation
   - Training discrete-time survival LSTM models
   - Memory estimation and cloud training setup
   - Handling censored data and competing risks

#### Scripts:

- **`validate_survival_model.py`**: Validation script for quick model testing with options for:
  - Patient subsampling for local testing
  - Example patient sequence display
  - Adjustable model complexity
  - Memory estimation
  - Synthetic outcome quality checks

#### Key Features:

- **Memory management**: Subsample patients for local testing (200 patients) or use full dataset on cloud GPUs
- **Cloud training guide**: Instructions for RunPods/Vast.ai setup with GPU recommendations
- **Synthetic outcomes**: Risk-based survival outcome generation with controllable censoring rates
- **C-index evaluation**: Proper concordance index computation for survival models
- **Diagnostic tools**: Correlation checks to validate synthetic outcome quality

## Getting Started

### Prerequisites

1. **Environment setup:**
   ```bash
   cd /path/to/ehr-sequencing
   mamba activate ehrsequencing
   ```

2. **Data setup:**
   - Option A: Use shared Synthea data from `loinc-predictor` project
   - Option B: Generate your own Synthea data (see `docs/datasets/SYNTHEA_SETUP.md`)
   - Option C: Use test fixtures for quick exploration

3. **Launch Jupyter:**
   ```bash
   jupyter lab
   ```

### Recommended Order

For first-time users, follow this sequence:

1. Start with `01_synthea_data_exploration/01_synthea_data_exploration.ipynb`
   - Understand the data structure
   - Learn the core APIs
   - See the complete pipeline

2. Continue with `01_synthea_data_exploration/01a_lstm_data_preparation.ipynb`
   - Learn how to prepare data for models
   - Understand shape transformations
   - See LSTM input format

3. Reference `01_synthea_data_exploration/data_shape_transformations.md`
   - Detailed shape specifications
   - Memory considerations
   - Common pitfalls

4. Explore production code in `examples/`
   - `train_lstm_baseline.py` - Full training script
   - See how notebooks translate to production

## Notebook Philosophy

### Educational Focus

These notebooks prioritize:
- **Clarity over brevity**: Explicit steps with explanations
- **Visualization**: Plots and diagrams to aid understanding
- **Interactivity**: Encourage experimentation
- **Small datasets**: Fast iteration for learning

### Differences from Production Code

| Aspect | Notebooks | Production (`examples/`) |
|--------|-----------|-------------------------|
| **Purpose** | Learning & exploration | Deployment & scale |
| **Data size** | Small subsets (50-100 patients) | Full datasets (1000s of patients) |
| **Code style** | Step-by-step, verbose | Modular, reusable functions |
| **Output** | Rich visualizations | Metrics, checkpoints, logs |
| **Error handling** | Minimal (for clarity) | Comprehensive |
| **Performance** | Not optimized | Optimized for speed/memory |

## Data Requirements

### 📖 Quick Start: Data Setup

**→ See [DATA_SETUP.md](DATA_SETUP.md) for complete data preparation guide**

This comprehensive guide covers:
- Where notebook data comes from (`~/work/loinc-predictor/data/synthea/all_cohorts`)
- How to copy data to this project (for pod deployment)
- How to upload data to RunPods using `rsync`
- How to generate new Synthea data
- Troubleshooting common data issues

### Synthea Data

Most notebooks use Synthea synthetic EHR data. See `docs/datasets/SYNTHEA_SETUP.md` for:
- Installation instructions
- Data generation guide
- Configuration options
- Troubleshooting

### Shared Data (Current Setup)

Notebooks currently reference data from the `loinc-predictor` project:

```python
data_path = Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'all_cohorts'
```

This contains ~100 patients with realistic EHR data.

**For pod deployment**, see [DATA_SETUP.md](DATA_SETUP.md) for instructions on making this data available.

### Test Fixtures

For quick testing without full Synthea setup:

```python
data_path = Path('../../tests/fixtures/mock_synthea_data')
```

Small sample dataset for unit tests and quick exploration.

## Common Tasks

### Loading Data

```python
from ehrsequencing.data.adapters import SyntheaAdapter

adapter = SyntheaAdapter(data_path='/path/to/synthea')
patients = adapter.load_patients(limit=50)
events = adapter.load_events(patient_ids=[p.patient_id for p in patients])
```

### Grouping Visits

```python
from ehrsequencing.data.visit_grouper import VisitGrouper

grouper = VisitGrouper(strategy='hybrid', time_window_hours=24)
patient_visits = {}
for patient_id in patient_ids:
    events = adapter.load_events(patient_ids=[patient_id])
    visits = grouper.group_events(events, patient_id=patient_id)
    patient_visits[patient_id] = visits
```

### Building Sequences

```python
from ehrsequencing.data.sequence_builder import PatientSequenceBuilder

builder = PatientSequenceBuilder(
    max_visits=50,
    max_codes_per_visit=100,
    use_semantic_order=True
)

vocab = builder.build_vocabulary(patient_visits, min_frequency=1)
sequences = builder.build_sequences(patient_visits, min_visits=2)
```

### Encoding for Models

```python
encoded = builder.encode_sequence(sequences[0], return_tensors=True)
# Returns dict with:
#   - visit_codes: [num_visits, max_codes_per_visit]
#   - visit_mask: [num_visits, max_codes_per_visit]
#   - sequence_mask: [num_visits]
#   - time_deltas: [num_visits-1]
```

## Tips for Using Notebooks

### Performance

- **Start small**: Use `limit=50` when loading patients
- **Profile memory**: Monitor memory usage for large datasets
- **Clear outputs**: Clear cell outputs before committing to git

### Experimentation

- **Duplicate cells**: Copy cells to try different parameters
- **Add markdown**: Document your findings inline
- **Save checkpoints**: Save processed data to avoid recomputation

### Debugging

- **Print shapes**: Always print tensor/array shapes
- **Visualize data**: Plot distributions and samples
- **Check masks**: Verify padding masks are correct
- **Use small batches**: Easier to inspect and debug

## Contributing

When adding new notebooks:

1. **Create a topic directory**: `notebooks/XX_topic_name/`
2. **Follow naming convention**: `XX_main_topic.ipynb`, `XXa_subtopic.ipynb`
3. **Add README**: Explain purpose and contents
4. **Include documentation**: Add `.md` files for complex topics
5. **Test thoroughly**: Ensure notebooks run end-to-end
6. **Clear outputs**: Before committing to git

## Related Resources

### Documentation

- `docs/datasets/` - Dataset setup and documentation
- `docs/methods/` - Methodological details
- `README.md` - Project overview

### Source Code

- `src/ehrsequencing/data/` - Data loading and processing
- `src/ehrsequencing/models/` - Model implementations
- `src/ehrsequencing/training/` - Training utilities

### Examples

- `examples/train_lstm_baseline.py` - LSTM training script
- `examples/` - Production-ready scripts

## Questions?

- Check notebook markdown cells for inline documentation
- See `data_shape_transformations.md` for shape reference
- Review source code for implementation details
- Refer to `docs/` for methodological background
