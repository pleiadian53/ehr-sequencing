# EHR Data Processing Module

This module provides tools for loading and processing **real EHR data** from various sources (Synthea, MIMIC, etc.) into a standardized format for EHR sequence modeling.

## Purpose

The `ehrsequencing.data` package is designed for:
- **Real EHR data adapters** (Synthea, MIMIC, etc.)
- **Visit grouping** with semantic code ordering
- **Patient sequence building** for temporal modeling
- **Systematic processing** for EHR-seq workflow and disease phenotyping

**Note:** Synthetic data generation has been moved to `ehrsequencing.synthetic` package.
For synthetic data (realistic patterns, domain shift, survival analysis), see `ehrsequencing.synthetic`.

## Components

### 1. Data Adapters (`adapters/`)

Adapters for loading EHR data from various sources:

```python
from ehrsequencing.data import SyntheaAdapter, MedicalEvent, PatientInfo

# Load Synthea data
adapter = SyntheaAdapter(data_dir='path/to/synthea/output')
patients = adapter.load_patients(max_patients=1000)

# Access patient information
for patient in patients:
    print(f"Patient {patient.patient_id}: {len(patient.events)} events")
    for event in patient.events:
        print(f"  {event.timestamp}: {event.code} - {event.description}")
```

### 2. Visit Grouper (`visit_grouper.py`)

Groups medical events into visits with semantic ordering:

```python
from ehrsequencing.data import VisitGrouper

grouper = VisitGrouper(time_window_hours=24)
visits = grouper.group_events(patient.events)

for visit in visits:
    print(f"Visit {visit.visit_id}: {len(visit.codes)} codes")
```

### 3. Sequence Builder (`sequence_builder.py`)

Builds patient sequences for temporal modeling:

```python
from ehrsequencing.data import PatientSequenceBuilder

builder = PatientSequenceBuilder(max_seq_length=512, vocab_size=1000)
sequences = builder.build_sequences(patients)

# Use with PyTorch
from ehrsequencing.data import PatientSequenceDataset
dataset = PatientSequenceDataset(sequences)
```

## Usage Example

Complete pipeline from raw EHR data to model-ready sequences:

```python
from ehrsequencing.data import (
    SyntheaAdapter,
    VisitGrouper,
    PatientSequenceBuilder,
    PatientSequenceDataset
)
from torch.utils.data import DataLoader

# 1. Load data
adapter = SyntheaAdapter('data/synthea')
patients = adapter.load_patients(max_patients=5000)

# 2. Group into visits
grouper = VisitGrouper(time_window_hours=24)
for patient in patients:
    patient.visits = grouper.group_events(patient.events)

# 3. Build sequences
builder = PatientSequenceBuilder(max_seq_length=512)
sequences = builder.build_sequences(patients)

# 4. Create dataset and dataloader
dataset = PatientSequenceDataset(sequences)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 5. Train model
for batch in dataloader:
    codes, ages, visit_ids, attention_mask = batch
    # Train your model...
```

## Supported Data Sources

- **Synthea**: Synthetic patient data generator
- **MIMIC-III**: (Coming soon) Critical care database
- **Custom**: Extend `BaseEHRAdapter` for your own data source

## For Synthetic Data

Synthetic data generation has been moved to `ehrsequencing.synthetic`:

```python
# Realistic synthetic data for medical LLM training
from ehrsequencing.synthetic import generate_realistic_dataset

# Domain-shifted datasets for transfer learning
from ehrsequencing.synthetic import generate_domain_shifted_datasets

# Survival analysis synthetic outcomes
from ehrsequencing.synthetic import DiscreteTimeSurvivalGenerator
```

See `ehrsequencing.synthetic` documentation for details.
