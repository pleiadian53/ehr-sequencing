# Development Session Notes

**Date:** January 20, 2026  
**Session Focus:** Data Pipeline Implementation with Visit Structuring

---

## Session Summary

Implemented complete data pipeline for EHR sequencing with visit-grouped representation and semantic code ordering.

---

## Completed Work

### 1. Platform-Specific Environments

**Problem:** M1 Mac installation failed due to CUDA dependencies in `environment.yml`

**Solution:** Created three platform-specific environment files:
- `environment-macos.yml` - M1/M2/M3 Mac with MPS support
- `environment-cuda.yml` - Linux/Windows with NVIDIA GPU (CUDA 12.1)
- `environment-cpu.yml` - CPU-only for any platform

**Files Created:**
- `/Users/pleiadian53/work/ehr-sequencing/environment-macos.yml`
- `/Users/pleiadian53/work/ehr-sequencing/environment-cuda.yml`
- `/Users/pleiadian53/work/ehr-sequencing/environment-cpu.yml`
- `/Users/pleiadian53/work/ehr-sequencing/dev/installation/environment-guide.md`

**Updated:**
- `environment.yml` (now macOS-compatible by default)
- `INSTALL.md` (platform-specific instructions)

---

### 2. Data Pipeline Implementation

Implemented complete data pipeline with three main components:

#### A. Data Adapters

**Base Adapter (`src/ehrsequencing/data/adapters/base.py`):**
- `MedicalEvent` dataclass - standardized event representation
- `PatientInfo` dataclass - patient demographics
- `BaseEHRAdapter` abstract class - interface for all adapters
- Methods: `load_patients()`, `load_events()`, `get_statistics()`
- Validation and error handling

**Synthea Adapter (`src/ehrsequencing/data/adapters/synthea.py`):**
- Loads Synthea CSV files (patients, encounters, conditions, observations, medications, procedures)
- Normalizes to `MedicalEvent` format
- Supports filtering by patient IDs, date range, code types
- Provides dataset statistics

#### B. Visit Grouper (`src/ehrsequencing/data/visit_grouper.py`)

**Key Features:**
- Four grouping strategies:
  - `encounter`: Use explicit encounter IDs
  - `same_day`: Group events on same calendar day
  - `time_window`: Group events within time window (e.g., 24 hours)
  - `hybrid`: Use encounter IDs when available, fallback to same_day
- **Semantic code ordering** within visits by type
- `Visit` dataclass with:
  - `codes_by_type`: Dictionary mapping code type to list of codes
  - `codes_flat`: Flat list of all codes
  - `get_ordered_codes()`: Returns codes in semantic order (diagnosis → lab → vital → procedure → medication)

**Design Decision:**
Implemented **Option B (Typed Groups)** from `docs/implementation/visit-grouped-sequences.md`:
- Preserves code types within visits
- Enables semantic ordering without requiring individual timestamps
- Clinically meaningful structure

#### C. Sequence Builder (`src/ehrsequencing/data/sequence_builder.py`)

**Key Features:**
- `PatientSequence` dataclass - complete patient timeline
- Vocabulary building with frequency filtering
- Sequence encoding to integer IDs
- Padding and truncation to fixed lengths
- Time delta calculation between visits
- PyTorch `Dataset` class for training

**Configuration:**
- `max_visits`: Maximum visits per sequence (default: 100)
- `max_codes_per_visit`: Maximum codes per visit (default: 50)
- `use_semantic_order`: Use semantic ordering within visits (default: True)

---

### 3. Unit Tests

**File:** `tests/test_data_pipeline.py`

**Test Coverage:**
- `TestMedicalEvent`: Event creation and validation
- `TestVisitGrouper`: All grouping strategies, semantic ordering
- `TestPatientSequenceBuilder`: Vocabulary, encoding, dataset creation
- `TestSyntheaAdapter`: Loading patients, events, statistics (with mock data)

**Run Tests:**
```bash
pytest tests/test_data_pipeline.py -v
```

---

### 4. Documentation Updates

**Installation Documentation:**
- `INSTALL.md` - Updated with platform-specific instructions
- `dev/installation/local-setup.md` - Detailed local setup (M1 Mac, Windows, Linux)
- `dev/installation/runpod-setup.md` - Cloud GPU training setup
- `dev/installation/pretrained-models.md` - CEHR-BERT integration guide
- `dev/installation/environment-guide.md` - Platform environment comparison

**Project Documentation:**
- `README.md` - Updated features, usage example, installation
- `dev/workflow/ROADMAP.md` - Updated Phase 1 progress (75% complete)

---

## Architecture Decisions

### 1. Visit Grouping Strategy: Hybrid

**Rationale:**
- Most EHR systems have encounter IDs (use when available)
- Fallback to same-day grouping for missing encounter IDs
- Clinically accurate representation

### 2. Within-Visit Structure: Semantic Grouping by Type

**Rationale:**
- Preserves code types without requiring individual timestamps
- Enables semantic ordering: diagnosis → lab → vital → procedure → medication
- Reflects typical clinical workflow
- Compatible with both flat and hierarchical models

**Implementation:**
```python
visit.codes_by_type = {
    'diagnosis': ['E11.9', 'I10'],
    'lab': ['4548-4', '2345-7'],
    'medication': ['860975']
}

# Get semantically ordered codes
ordered_codes = visit.get_ordered_codes()
# ['E11.9', 'I10', '4548-4', '2345-7', '860975']
```

### 3. Sequence Representation: Visit-Grouped

**Format:**
```python
PatientSequence:
  patient_id: 'patient_1'
  visits: [Visit1, Visit2, Visit3, ...]
  sequence_length: 50
  
# Encoded format for models:
visit_codes: [num_visits, max_codes_per_visit]  # Integer IDs
visit_mask: [num_visits, max_codes_per_visit]   # 1=real, 0=padding
sequence_mask: [num_visits]                      # 1=real visit, 0=padding
time_deltas: [num_visits - 1]                    # Days between visits
```

---

## Key Design Patterns

### 1. Adapter Pattern
- `BaseEHRAdapter` defines interface
- `SyntheaAdapter`, `MIMICAdapter` (future) implement interface
- Consistent API across data sources

### 2. Builder Pattern
- `PatientSequenceBuilder` constructs sequences step-by-step
- Vocabulary building → Sequence creation → Dataset creation
- Configurable parameters (max_visits, max_codes_per_visit)

### 3. Strategy Pattern
- `VisitGrouper` supports multiple grouping strategies
- Runtime selection: `strategy='hybrid'`
- Easy to add new strategies

---

## Usage Example

```python
from ehrsequencing.data import SyntheaAdapter, VisitGrouper, PatientSequenceBuilder

# 1. Load data
adapter = SyntheaAdapter('data/synthea/')
patients = adapter.load_patients(limit=100)
events = adapter.load_events(patient_ids=[p.patient_id for p in patients])

# 2. Group into visits (hybrid strategy with semantic ordering)
grouper = VisitGrouper(strategy='hybrid', preserve_code_types=True)
patient_visits = grouper.group_by_patient(events)

# 3. Build sequences
builder = PatientSequenceBuilder(
    max_visits=50,
    max_codes_per_visit=100,
    use_semantic_order=True
)
vocab = builder.build_vocabulary(patient_visits, min_frequency=5)
sequences = builder.build_sequences(patient_visits, min_visits=2)

# 4. Create dataset
dataset = builder.create_dataset(sequences)

# 5. Use with PyTorch DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    visit_codes = batch['visit_codes']      # [32, 50, 100]
    visit_mask = batch['visit_mask']        # [32, 50, 100]
    sequence_mask = batch['sequence_mask']  # [32, 50]
    time_deltas = batch['time_deltas']      # [32, 49]
    # Train model...
```

---

## Next Steps

### Immediate (Phase 1 Completion)
1. **LSTM Baseline Model** - Implement resource-aware LSTM for disease progression
   - Small config for M1 Mac (16GB)
   - Medium config for RunPod (24GB GPU)
   - Large config for cloud (40GB+ GPU)
2. **Data Exploration Notebook** - `notebooks/01_data_exploration.ipynb`
3. **Example Script** - `examples/train_disease_progression.py`

### Phase 2 (Code Embeddings)
1. **CEHR-BERT Integration** - Load pre-trained embeddings
2. **Embedding Wrapper** - `src/ehrsequencing/embeddings/cehrbert_wrapper.py`
3. **Fine-tuning Pipeline** - Fine-tune on disease-specific tasks

### Phase 3 (Advanced Models)
1. **Transformer Model** - Attention-based sequence model
2. **Hierarchical Model** - Code → Visit → Patient hierarchy
3. **Multi-task Learning** - Joint prediction of multiple outcomes

---

## Files Created/Modified

### Created
- `src/ehrsequencing/data/adapters/base.py` (300 lines)
- `src/ehrsequencing/data/adapters/synthea.py` (350 lines)
- `src/ehrsequencing/data/visit_grouper.py` (400 lines)
- `src/ehrsequencing/data/sequence_builder.py` (450 lines)
- `tests/test_data_pipeline.py` (400 lines)
- `environment-macos.yml`
- `environment-cuda.yml`
- `environment-cpu.yml`
- `dev/installation/environment-guide.md`

### Modified
- `src/ehrsequencing/data/adapters/__init__.py`
- `src/ehrsequencing/data/__init__.py`
- `environment.yml`
- `INSTALL.md`
- `README.md`
- `dev/workflow/ROADMAP.md`

---

## Technical Notes

### Memory Considerations
- Sequence padding can be memory-intensive
- Use `max_visits=50` and `max_codes_per_visit=100` for M1 Mac
- Increase for cloud GPUs: `max_visits=100`, `max_codes_per_visit=200`

### Performance
- Vocabulary building: O(n) where n = total codes
- Visit grouping: O(n log n) due to sorting
- Sequence encoding: O(m * v * c) where m=patients, v=visits, c=codes

### Future Optimizations
- Lazy loading for large datasets
- Caching of encoded sequences
- Parallel processing for vocabulary building
- Memory-mapped datasets for very large cohorts

---

**Session Duration:** ~2 hours  
**Lines of Code:** ~2000 (production) + 400 (tests)  
**Documentation:** ~3000 lines across multiple files

---

**Status:** ✅ Data pipeline complete and tested  
**Next Session:** LSTM baseline model implementation
