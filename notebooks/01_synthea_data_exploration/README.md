# Synthea Data Exploration

Comprehensive exploration of Synthea synthetic EHR data and preparation for LSTM baseline modeling.

## Overview

This directory contains notebooks and documentation for:
1. **Exploring Synthea data** - Understanding the structure and content of synthetic EHR data
2. **Building patient sequences** - Grouping events into visits and creating temporal sequences
3. **Preparing LSTM inputs** - Encoding sequences for neural network models
4. **Understanding data transformations** - Tracking shape changes through the pipeline

## Contents

### Notebooks

#### `01_synthea_data_exploration.ipynb`

**Purpose:** Comprehensive exploration of Synthea data and the complete data processing pipeline.

**What you'll learn:**
- How to load Synthea CSV files using `SyntheaAdapter`
- Patient demographics and clinical data distributions
- Event types (conditions, observations, medications, procedures)
- Visit grouping strategies and their effects
- Patient sequence construction
- Data quality assessment
- Temporal patterns in EHR data

**Key outputs:**
- Patient demographics summary
- Code frequency distributions
- Visit statistics (visits per patient, codes per visit, time between visits)
- Sequence length distributions
- Data quality metrics

**Runtime:** ~5-10 minutes with 50 patients

---

#### `01a_lstm_data_preparation.ipynb`

**Purpose:** Detailed walkthrough of preparing visit-grouped sequences for the LSTM baseline model.

**What you'll learn:**
- How to encode string codes to integer IDs
- Vocabulary building from patient data
- Padding and masking strategies
- Creating labels for prediction tasks
- Batching sequences with variable lengths
- Data shape transformations at each step
- Running LSTM model forward pass

**Prediction task:** Binary classification - predicting diabetes diagnosis from EHR sequences

**Key outputs:**
- Encoded sequences with proper padding
- Labeled dataset (diabetes vs. no diabetes)
- Batched PyTorch tensors ready for training
- Model predictions on sample batch
- Complete pipeline visualization

**Runtime:** ~3-5 minutes with 50 patients

---

### Documentation

#### `data_shape_transformations.md`

**Purpose:** Comprehensive reference for all data shape transformations in the pipeline.

**Contents:**
- Stage-by-stage transformation details
- Shape specifications for each data structure
- Memory footprint calculations
- Common pitfalls and how to avoid them
- Code examples for each transformation

**Use this when:**
- Debugging shape mismatch errors
- Understanding memory requirements
- Implementing custom data processing
- Optimizing batch sizes

---

## Getting Started

### Prerequisites

1. **Activate environment:**
   ```bash
   mamba activate ehrsequencing
   ```

2. **Set up data:**
   
   **Option A - Use shared data (recommended):**
   ```python
   data_path = Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'all_cohorts'
   ```
   
   **Option B - Generate your own:**
   See `../../docs/datasets/SYNTHEA_SETUP.md` for instructions
   
   **Option C - Use test fixtures:**
   ```python
   data_path = Path('../../tests/fixtures/mock_synthea_data')
   ```

3. **Launch Jupyter:**
   ```bash
   cd notebooks/01_synthea_data_exploration
   jupyter lab
   ```

### Recommended Order

1. **Start here:** `01_synthea_data_exploration.ipynb`
   - Run all cells to see the complete pipeline
   - Experiment with different parameters
   - Understand the data structure

2. **Deep dive:** `01a_lstm_data_preparation.ipynb`
   - Focus on model input preparation
   - Understand encoding and batching
   - See shape transformations in detail

3. **Reference:** `data_shape_transformations.md`
   - Consult when debugging
   - Use as shape specification reference
   - Review memory considerations

## Key Concepts

### Visit Grouping

**Problem:** Raw EHR data consists of individual events (one code at a time). We need to group related events into clinical visits.

**Solution:** `VisitGrouper` with multiple strategies:
- `'encounter'`: Group by encounter ID (when available)
- `'same_day'`: Group events on the same calendar day
- `'time_window'`: Group events within N hours
- `'hybrid'`: Use encounter IDs when available, fallback to time-based grouping

**Example:**
```python
grouper = VisitGrouper(strategy='hybrid', time_window_hours=24)
visits = grouper.group_events(events, patient_id=patient_id)
```

### Patient Sequences

**Problem:** Visits need to be organized into patient-level temporal sequences for modeling.

**Solution:** `PatientSequenceBuilder` creates structured sequences with:
- Chronologically ordered visits
- Vocabulary mapping (code → integer ID)
- Configurable sequence length limits
- Semantic code ordering within visits

**Example:**
```python
builder = PatientSequenceBuilder(max_visits=50, max_codes_per_visit=100)
vocab = builder.build_vocabulary(patient_visits)
sequences = builder.build_sequences(patient_visits, min_visits=2)
```

### Encoding & Padding

**Problem:** Neural networks require fixed-size inputs, but EHR sequences have variable lengths.

**Solution:** Encode codes to IDs and pad to fixed dimensions:
- Convert string codes → integer IDs via vocabulary
- Pad visits to `max_codes_per_visit`
- Pad sequences to `max_visits`
- Create masks to track real vs. padded data

**Example:**
```python
encoded = builder.encode_sequence(sequence, return_tensors=True)
# Returns:
#   visit_codes: [num_visits, max_codes_per_visit]
#   visit_mask: [num_visits, max_codes_per_visit]
#   sequence_mask: [num_visits]
```

### Batching

**Problem:** Training requires batching multiple sequences together.

**Solution:** `collate_fn` dynamically pads batch to max lengths:
- Find max visits and max codes in current batch
- Pad all sequences to these dimensions
- Create batch-level masks
- Return PyTorch tensors

**Example:**
```python
batch = collate_fn(dataset_items[:32])
# Returns:
#   visit_codes: [32, max_visits_in_batch, max_codes_in_batch]
#   visit_mask: [32, max_visits_in_batch, max_codes_in_batch]
#   sequence_mask: [32, max_visits_in_batch]
#   labels: [32, 1]
```

## Data Pipeline Summary

```
Raw Synthea CSV Files
    ↓ SyntheaAdapter.load_events()
List[MedicalEvent] - Individual timestamped codes
    ↓ VisitGrouper.group_events()
Dict[patient_id, List[Visit]] - Events grouped into visits
    ↓ PatientSequenceBuilder.build_sequences()
List[PatientSequence] - Structured patient sequences
    ↓ PatientSequenceBuilder.encode_sequence()
Encoded sequences with padding/masking
    ↓ Add labels + collate_fn()
Batched PyTorch tensors
    ↓ LSTM Model
Predictions
```

## Prediction Task: Diabetes Detection

Both notebooks use **diabetes detection** as the example prediction task:

**Task type:** Binary classification

**Label definition:** 
- `1` if patient has any diabetes diagnosis code in their history
- `0` otherwise

**Diabetes codes (SNOMED-CT):**
- `44054006` - Type 2 diabetes mellitus
- `46635009` - Type 1 diabetes mellitus
- `73211009` - Diabetes mellitus
- `11687002` - Gestational diabetes
- `190330002` - Diabetes mellitus without complication
- `190331003` - Diabetes mellitus with complication

**Why diabetes?**
- Common condition (~10-20% prevalence in typical Synthea data)
- Clear diagnostic codes
- Clinically meaningful
- Good balance for binary classification

**Other possible tasks:**
- Readmission prediction (time-based)
- Mortality prediction (death_date)
- Disease onset prediction (temporal)
- Multi-disease phenotyping (multi-class)
- Risk score prediction (regression)

## Common Parameters

### Data Loading

```python
# Number of patients to load
limit = 50  # Start small for exploration

# Patient filtering
patient_ids = [p.patient_id for p in patients[:10]]
```

### Visit Grouping

```python
strategy = 'hybrid'           # 'encounter', 'same_day', 'time_window', 'hybrid'
time_window_hours = 24        # For time-based grouping
preserve_code_types = True    # Keep semantic structure
```

### Sequence Building

```python
max_visits = 50               # Maximum visits per sequence
max_codes_per_visit = 100     # Maximum codes per visit
min_visits = 2                # Minimum visits to include patient
use_semantic_order = True     # Order codes by type
min_frequency = 1             # Minimum code frequency for vocabulary
```

### Model Configuration

```python
vocab_size = len(vocab)       # Determined by data
embedding_dim = 128           # Code embedding dimension
hidden_dim = 256              # LSTM hidden dimension
num_layers = 1                # Number of LSTM layers
batch_size = 32               # Training batch size
```

## Troubleshooting

### Issue: KeyError when loading data

**Cause:** CSV column names don't match expected format

**Solution:** Check that you're using Synthea data (not MIMIC or other formats)

### Issue: AttributeError on Visit or PatientSequence

**Cause:** Treating dataclass as dictionary

**Solution:** Use attribute access (`.patient_id`) not dict access (`['patient_id']`)

### Issue: Shape mismatch in model forward pass

**Cause:** Incorrect tensor dimensions or missing masks

**Solution:** Check `data_shape_transformations.md` for expected shapes

### Issue: Out of memory

**Cause:** Too many patients or too large dimensions

**Solution:** 
- Reduce `limit` when loading patients
- Decrease `max_visits` or `max_codes_per_visit`
- Use smaller `batch_size`

### Issue: Empty sequences after filtering

**Cause:** `min_visits` threshold too high

**Solution:** Lower `min_visits` or load more patients

## Performance Tips

### For Exploration (Notebooks)

- Use `limit=50` patients for fast iteration
- Clear cell outputs before committing
- Restart kernel if memory grows large

### For Training (Production)

- Use full dataset (no limit)
- Increase batch size for efficiency
- Use GPU if available
- Enable mixed precision (fp16)

## Memory Estimates

### Per Patient (avg)

- Events: ~100 events × 200 bytes = 20 KB
- Visits: ~10 visits × 500 bytes = 5 KB
- Sequence: ~50 KB (encoded)
- **Total: ~75 KB per patient**

### Per Batch (batch_size=32)

- Tensors: ~1.5 MB
- Gradients: ~3 MB
- **Total: ~5 MB per batch**

### Full Dataset

| Patients | Memory | Training Time |
|----------|--------|---------------|
| 100 | ~10 MB | Minutes |
| 1,000 | ~100 MB | Hours |
| 10,000 | ~1 GB | Days |

## Next Steps

After completing these notebooks:

1. **Run production training:**
   ```bash
   python examples/train_lstm_baseline.py \
       --data_dir /path/to/synthea \
       --max_patients 1000 \
       --num_epochs 10 \
       --output_dir ./outputs
   ```

2. **Experiment with models:**
   - Try different LSTM configurations
   - Add attention mechanisms
   - Use bidirectional LSTMs
   - Experiment with different aggregation strategies

3. **Try different tasks:**
   - Readmission prediction
   - Mortality prediction
   - Multi-disease phenotyping
   - Next visit prediction

4. **Explore advanced methods:**
   - Med2Vec embeddings
   - Transformer models
   - Graph neural networks

## Related Resources

- `../../docs/datasets/SYNTHEA_SETUP.md` - Synthea data generation
- `../../examples/train_lstm_baseline.py` - Production training script
- `../../src/ehrsequencing/models/lstm_baseline.py` - Model implementation
- `../../src/ehrsequencing/data/` - Data processing modules
