# Data Shape Transformations: From Raw EHR to LSTM Input

This document provides a comprehensive reference for all data shape transformations in the EHR sequence modeling pipeline, from raw Synthea CSV files to LSTM model predictions.

**Prediction Task:** Binary classification - predicting diabetes diagnosis from patient EHR sequences.

---

## Table of Contents

1. [Overview](#overview)
2. [Stage-by-Stage Transformations](#stage-by-stage-transformations)
3. [Detailed Shape Specifications](#detailed-shape-specifications)
4. [Memory Considerations](#memory-considerations)
5. [Common Pitfalls](#common-pitfalls)

---

## Overview

### Pipeline Summary

```
Raw CSV Files
    ↓ SyntheaAdapter.load_events()
List[MedicalEvent]
    ↓ VisitGrouper.group_events()
Dict[str, List[Visit]]
    ↓ PatientSequenceBuilder.build_sequences()
List[PatientSequence]
    ↓ PatientSequenceBuilder.encode_sequence()
Encoded Sequences (with padding)
    ↓ Add labels + collate_fn()
Batched PyTorch Tensors
    ↓ LSTM Model forward()
Predictions
```

### Key Dimensions

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| `N` | Number of patients | 50-10000 |
| `V` | Number of visits per patient | 2-100 |
| `C` | Number of codes per visit | 1-50 |
| `B` | Batch size | 16-128 |
| `E` | Embedding dimension | 128-512 |
| `H` | Hidden dimension | 256-1024 |
| `vocab_size` | Vocabulary size | 500-5000 |

---

## Stage-by-Stage Transformations

### Stage 1: Raw CSV Files → Medical Events

**Input:** CSV files on disk
- `patients.csv`
- `encounters.csv`
- `conditions.csv`
- `observations.csv`
- `medications.csv`
- `procedures.csv`

**Transformation:** `SyntheaAdapter.load_events(patient_ids)`

**Output:** `List[MedicalEvent]`

```python
# Data structure
MedicalEvent(
    patient_id: str,           # UUID
    timestamp: datetime,       # Timezone-naive
    code: str,                 # Medical code (SNOMED-CT, LOINC, RxNorm, etc.)
    code_type: str,            # 'diagnosis', 'lab', 'medication', 'procedure'
    encounter_id: Optional[str],
    metadata: Optional[Dict]
)

# Shape
List with length = total number of events across all patients
Example: 4,372 events for 50 patients
```

**Key Points:**
- Events are sorted by timestamp
- All timestamps are timezone-naive for consistency
- Each event represents a single medical code occurrence

---

### Stage 2: Medical Events → Visit Groups

**Input:** `List[MedicalEvent]`

**Transformation:** `VisitGrouper.group_events(events, patient_id)`

**Output:** `Dict[str, List[Visit]]`

```python
# Data structure
{
    patient_id_1: [Visit_1, Visit_2, ..., Visit_n1],
    patient_id_2: [Visit_1, Visit_2, ..., Visit_n2],
    ...
}

Visit(
    visit_id: str,
    patient_id: str,
    timestamp: datetime,
    encounter_id: Optional[str],
    codes_by_type: Dict[str, List[str]],  # e.g., {'diagnosis': [...], 'lab': [...]}
    codes_flat: List[str],
    metadata: Optional[Dict]
)

# Shape
Dict with:
  - Keys: N patient IDs (strings)
  - Values: Lists of Visit objects
  - Total visits: sum(len(visits) for visits in dict.values())
  
Example: 
  - 50 patients
  - 421 total visits
  - Average 8.4 visits per patient
```

**Key Points:**
- Visits group events that occur within the same encounter or time window
- `codes_by_type` preserves semantic structure
- `codes_flat` provides simple list for basic models
- Visit timestamps represent the start of the visit

---

### Stage 3: Visit Groups → Patient Sequences

**Input:** `Dict[str, List[Visit]]`

**Transformation:** `PatientSequenceBuilder.build_sequences(patient_visits, min_visits=2)`

**Output:** `List[PatientSequence]`

```python
# Data structure
PatientSequence(
    patient_id: str,
    visits: List[Visit],
    sequence_length: int,
    metadata: Optional[Dict]
)

# Shape
List with length = number of patients with >= min_visits
Example: 42 sequences (filtered from 50 patients)

# Each sequence contains:
  - visits: List[Visit] with length = sequence_length
  - sequence_length: int (number of visits)
```

**Key Points:**
- Filters out patients with insufficient visits
- Visits are chronologically ordered
- Sequences can have variable length
- No padding at this stage

---

### Stage 4: Patient Sequences → Encoded Sequences

**Input:** `PatientSequence`

**Transformation:** `PatientSequenceBuilder.encode_sequence(sequence, return_tensors=False)`

**Output:** `Dict[str, Any]`

```python
# Data structure
{
    'patient_id': str,
    'visit_codes': List[List[int]],      # [num_visits, max_codes_per_visit]
    'visit_mask': List[List[int]],       # [num_visits, max_codes_per_visit]
    'sequence_mask': List[int],          # [num_visits]
    'time_deltas': List[float],          # [num_visits - 1]
    'sequence_length': int
}

# Shape details
visit_codes: [V, C]
  - V = min(sequence.sequence_length, max_visits)
  - C = max_codes_per_visit
  - Values: integer IDs from vocabulary (0 = [PAD], 1 = [UNK], 2+ = codes)
  - Padded with 0s

visit_mask: [V, C]
  - Same shape as visit_codes
  - Values: 1 for real codes, 0 for padding
  - Used to ignore padding in aggregation

sequence_mask: [V]
  - Values: 1 for real visits, 0 for padding visits
  - Used to handle variable-length sequences in LSTM

time_deltas: [V-1]
  - Time between consecutive visits in days
  - Padded with 0.0 for padding visits

# Example with max_visits=50, max_codes_per_visit=100
visit_codes: [50, 100]     # 5,000 integers
visit_mask: [50, 100]      # 5,000 integers
sequence_mask: [50]        # 50 integers
time_deltas: [49]          # 49 floats
```

**Key Points:**
- Codes converted from strings to integer IDs via vocabulary
- Padding applied to standardize dimensions
- Masks track real vs. padded data
- Most recent visits kept if sequence exceeds `max_visits`

---

### Stage 5: Encoded Sequences → Labeled Dataset

**Input:** `List[PatientSequence]`

**Transformation:** Add labels based on prediction task

**Output:** `List[Dict]`

```python
# Data structure
[
    {
        'patient_id': str,
        'visit_codes': List[List[int]],
        'visit_mask': List[List[int]],
        'sequence_mask': List[int],
        'time_deltas': List[float],
        'label': int  # 0 or 1 for binary classification
    },
    ...
]

# Shape
List with length = number of sequences
Each item is a dict with encoded sequence + label

# Label creation (diabetes example)
diabetes_codes = {'44054006', '46635009', '73211009', ...}
label = 1 if any code in diabetes_codes appears in any visit else 0

# Example label distribution
Total sequences: 42
Positive (has diabetes): 8 (19.0%)
Negative (no diabetes): 34 (81.0%)
```

**Key Points:**
- Labels derived from medical codes in the sequence
- For diabetes: check if any visit contains diabetes diagnosis code
- Other tasks: readmission (time-based), mortality (death_date), etc.
- Labels can be binary, multi-class, or continuous

---

### Stage 6: Labeled Dataset → Batched Tensors

**Input:** `List[Dict]` (dataset items)

**Transformation:** `collate_fn(batch)` in DataLoader

**Output:** `Dict[str, torch.Tensor]`

```python
# Data structure
{
    'visit_codes': torch.Tensor,    # [B, V_max, C_max]
    'visit_mask': torch.Tensor,     # [B, V_max, C_max]
    'sequence_mask': torch.Tensor,  # [B, V_max]
    'labels': torch.Tensor          # [B, 1]
}

# Shape details
visit_codes: [B, V_max, C_max]
  - B = batch_size (e.g., 32)
  - V_max = max number of visits in batch
  - C_max = max number of codes per visit in batch
  - dtype: torch.long
  - Values: 0 (padding) or vocabulary IDs

visit_mask: [B, V_max, C_max]
  - Same shape as visit_codes
  - dtype: torch.bool
  - Values: True for real codes, False for padding

sequence_mask: [B, V_max]
  - dtype: torch.bool
  - Values: True for real visits, False for padding

labels: [B, 1]
  - dtype: torch.float32
  - Values: 0.0 or 1.0 for binary classification

# Example with batch_size=32
visit_codes: [32, 45, 87]      # 125,280 values
visit_mask: [32, 45, 87]       # 125,280 values
sequence_mask: [32, 45]        # 1,440 values
labels: [32, 1]                # 32 values

# Memory footprint (batch_size=32)
visit_codes: ~977 KB (int64)
visit_mask: ~122 KB (bool)
sequence_mask: ~1.4 KB (bool)
labels: ~0.1 KB (float32)
Total: ~1.1 MB per batch
```

**Key Points:**
- Dynamic padding: `V_max` and `C_max` determined by batch contents
- Efficient packing: only pad to max in current batch, not global max
- Masks essential for proper gradient computation
- DataLoader handles batching automatically

---

### Stage 7: Batched Tensors → Model Output

**Input:** Batched tensors from Stage 6

**Transformation:** `model.forward(visit_codes, visit_mask, sequence_mask)`

**Output:** `Dict[str, torch.Tensor]`

```python
# Model architecture flow
visit_codes [B, V, C]
    ↓ Embedding layer
code_embeddings [B, V, C, E]
    ↓ Visit encoder (aggregation)
visit_vectors [B, V, E]
    ↓ LSTM
lstm_output [B, V, H]
    ↓ Take final hidden state
final_hidden [B, H]
    ↓ Linear + activation
predictions [B, 1]

# Output structure
{
    'logits': torch.Tensor,         # [B, 1]
    'predictions': torch.Tensor,    # [B, 1]
    'hidden_states': torch.Tensor   # [B, V, H] (if return_hidden=True)
}

# Shape details
logits: [B, 1]
  - dtype: torch.float32
  - Raw predictions before activation
  - Range: (-∞, +∞)

predictions: [B, 1]
  - dtype: torch.float32
  - After sigmoid activation
  - Range: (0, 1) - interpreted as probabilities
  - P(patient has diabetes)

hidden_states: [B, V, H]
  - dtype: torch.float32
  - LSTM hidden states for each visit
  - Can be used for attention, interpretability, etc.

# Example with batch_size=32, hidden_dim=256
logits: [32, 1]           # 32 values
predictions: [32, 1]      # 32 values (probabilities)
hidden_states: [32, 45, 256]  # 368,640 values

# Memory footprint
hidden_states: ~1.4 MB (float32)
```

**Key Points:**
- Embedding layer maps code IDs to dense vectors
- Visit encoder aggregates codes within each visit (mean/sum/attention)
- LSTM captures temporal dependencies across visits
- Final prediction from last hidden state
- Sigmoid activation for binary classification

---

## Detailed Shape Specifications

### Vocabulary

```python
vocab: Dict[str, int]
  - Keys: Medical codes (strings)
  - Values: Integer IDs
  - Special tokens:
    • [PAD]: 0
    • [UNK]: 1
    • [MASK]: 2
    • [CLS]: 3
    • [SEP]: 4
  - Regular codes: 5, 6, 7, ...
  
Example:
{
    '[PAD]': 0,
    '[UNK]': 1,
    '[MASK]': 2,
    '[CLS]': 3,
    '[SEP]': 4,
    '44054006': 5,  # Type 2 diabetes
    '8302-2': 6,    # Body height
    ...
}

Typical size: 500-5000 codes
```

### Visit Object

```python
Visit:
  - visit_id: str (UUID)
  - patient_id: str (UUID)
  - timestamp: datetime (timezone-naive)
  - encounter_id: Optional[str]
  - codes_by_type: Dict[str, List[str]]
    • Keys: 'diagnosis', 'lab', 'medication', 'procedure'
    • Values: Lists of code strings
  - codes_flat: List[str]
    • Flattened list of all codes
  - metadata: Optional[Dict]

Methods:
  - num_codes() -> int
  - get_all_codes() -> List[str]
  - get_ordered_codes(type_order) -> List[str]

Example:
Visit(
    visit_id='abc-123',
    patient_id='patient-456',
    timestamp=datetime(2024, 5, 5),
    codes_by_type={
        'diagnosis': ['44054006', '73211009'],
        'lab': ['8302-2', '29463-7', '8867-4'],
        'medication': ['197361']
    },
    codes_flat=['44054006', '73211009', '8302-2', '29463-7', '8867-4', '197361']
)
```

### PatientSequence Object

```python
PatientSequence:
  - patient_id: str
  - visits: List[Visit]
  - sequence_length: int (len(visits))
  - metadata: Optional[Dict]

Methods:
  - get_code_sequence(use_semantic_order: bool) -> List[List[str]]
  - get_flat_code_sequence() -> List[str]
  - get_time_deltas() -> List[float]

Example:
PatientSequence(
    patient_id='patient-456',
    visits=[Visit_1, Visit_2, ..., Visit_10],
    sequence_length=10,
    metadata={'age': 45, 'gender': 'M'}
)
```

---

## Memory Considerations

### Per-Sequence Memory

For a single encoded sequence with `max_visits=50`, `max_codes_per_visit=100`:

```
visit_codes:    50 × 100 × 8 bytes (int64)   = 40 KB
visit_mask:     50 × 100 × 1 byte (bool)     = 5 KB
sequence_mask:  50 × 1 byte (bool)           = 50 bytes
time_deltas:    49 × 4 bytes (float32)       = 196 bytes
Total:                                       ≈ 45 KB per sequence
```

### Per-Batch Memory

For a batch of 32 sequences:

```
visit_codes:    32 × 50 × 100 × 8 bytes     = 1.28 MB
visit_mask:     32 × 50 × 100 × 1 byte      = 160 KB
sequence_mask:  32 × 50 × 1 byte            = 1.6 KB
labels:         32 × 1 × 4 bytes            = 128 bytes
Total:                                      ≈ 1.44 MB per batch
```

### Model Memory

For LSTM baseline (small) with `vocab_size=1000`, `embedding_dim=128`, `hidden_dim=256`:

```
Embedding:      1000 × 128 × 4 bytes        = 512 KB
LSTM:           ~500K parameters            = 2 MB
Linear:         256 × 1 × 4 bytes           = 1 KB
Total parameters:                           ≈ 2.5 MB

Forward pass (batch_size=32):
  - Embeddings:     32 × 50 × 100 × 128     = 20.48 MB
  - Visit vectors:  32 × 50 × 128           = 819 KB
  - LSTM hidden:    32 × 50 × 256           = 1.64 MB
  - Gradients:      ~2× forward pass        = 45 MB
Total:                                      ≈ 70 MB per batch
```

### Scaling Considerations

| Dataset Size | Sequences | Batches (B=32) | Memory | Training Time |
|--------------|-----------|----------------|--------|---------------|
| Small | 100 | 4 | ~6 MB | Minutes |
| Medium | 1,000 | 32 | ~50 MB | Hours |
| Large | 10,000 | 313 | ~450 MB | Days |
| Very Large | 100,000 | 3,125 | ~4.5 GB | Weeks |

**Recommendations:**
- Use `max_visits=50` and `max_codes_per_visit=100` for most tasks
- Reduce dimensions if memory-constrained
- Use gradient accumulation for larger effective batch sizes
- Consider mixed precision training (fp16) to reduce memory by 50%

---

## Common Pitfalls

### 1. Forgetting Masks

**Problem:** Not using masks leads to incorrect aggregation and gradient computation.

```python
# ❌ Wrong - includes padding in mean
visit_vector = code_embeddings.mean(dim=1)

# ✅ Correct - masks out padding
masked_embeddings = code_embeddings * visit_mask.unsqueeze(-1)
visit_vector = masked_embeddings.sum(dim=1) / visit_mask.sum(dim=1, keepdim=True).clamp(min=1)
```

### 2. Incorrect Padding Direction

**Problem:** Padding at the beginning instead of the end.

```python
# ❌ Wrong - pads at start (shifts temporal order)
padded = [PAD, PAD, code1, code2, code3]

# ✅ Correct - pads at end (preserves temporal order)
padded = [code1, code2, code3, PAD, PAD]
```

### 3. Timezone Issues

**Problem:** Mixing timezone-aware and timezone-naive timestamps.

```python
# ❌ Wrong - causes comparison errors
timestamp1 = pd.to_datetime('2024-01-01')  # naive
timestamp2 = pd.to_datetime('2024-01-01').tz_localize('UTC')  # aware

# ✅ Correct - all timestamps naive
timestamp = pd.to_datetime(row['DATE']).tz_localize(None)
```

### 4. Dictionary vs. Dataclass Access

**Problem:** Treating dataclasses as dictionaries.

```python
# ❌ Wrong - PatientSequence is a dataclass
patient_id = sequence['patient_id']

# ✅ Correct - use attribute access
patient_id = sequence.patient_id
```

### 5. Batch Dimension Confusion

**Problem:** Forgetting batch dimension in reshaping.

```python
# ❌ Wrong - loses batch structure
embeddings = embeddings.view(-1, embedding_dim)

# ✅ Correct - preserves batch
embeddings = embeddings.view(batch_size, num_visits, -1, embedding_dim)
```

### 6. Variable Length Handling

**Problem:** Not using packed sequences for efficiency.

```python
# ❌ Inefficient - processes padding
lstm_output, _ = lstm(visit_vectors)

# ✅ Efficient - skips padding
lengths = sequence_mask.sum(dim=1).cpu()
packed = nn.utils.rnn.pack_padded_sequence(visit_vectors, lengths, batch_first=True, enforce_sorted=False)
packed_output, _ = lstm(packed)
lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
```

---

## Summary

This document provides a complete reference for data shapes throughout the EHR sequence modeling pipeline. Key points:

1. **Consistent shapes**: All transformations maintain clear input/output contracts
2. **Proper masking**: Essential for handling variable-length sequences
3. **Memory efficiency**: Dynamic padding and packed sequences reduce waste
4. **Type safety**: Clear distinction between lists, dicts, dataclasses, and tensors
5. **Scalability**: Pipeline handles datasets from 100 to 100,000+ patients

For implementation details, see:
- `01_synthea_data_exploration.ipynb` - Data loading and exploration
- `01a_lstm_data_preparation.ipynb` - LSTM input preparation
- `examples/train_lstm_baseline.py` - Full training pipeline
- `src/ehrsequencing/models/lstm_baseline.py` - Model architecture
