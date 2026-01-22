# EHR Sequencing

**Biological Language Model for Electronic Health Records**

> Treating medical codes as "words" and patient histories as "documents" to model disease progression and discover temporal phenotypes.

---

## Overview

EHR Sequencing applies sequence modeling techniques from genomics and NLP to Electronic Health Records, enabling:

- **Disease Progression Modeling** - Predict future diagnoses and outcomes
- **Temporal Phenotyping** - Discover disease subtypes from patient trajectories  
- **Patient Segmentation** - Cluster patients by clinical similarity
- **Clinical Trajectory Analysis** - Understand disease evolution patterns

### The Analogy

```
DNA Sequences (ATCG...)  →  Genomic Language Models
    ↓                              ↓
Medical Code Sequences   →  EHR Sequencing Models
(LOINC, SNOMED, ICD...)      (This Project)
```

---

## Project Status

**Phase:** Foundation & Modernization  
**Version:** 0.1.0 (Alpha)  
**Status:** Active Development

This project is being modernized from a legacy codebase (`temporal-phenotyping`) with updated architecture, modern dependencies, and comprehensive documentation.

---

## Quick Start

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd ehr-sequencing

# 2. Create conda environment (choose your platform)
# macOS (M1/M2/M3):
mamba env create -f environment-macos.yml
# Linux/Windows with NVIDIA GPU:
mamba env create -f environment-cuda.yml
# CPU-only:
mamba env create -f environment-cpu.yml

# 3. Activate environment
mamba activate ehrsequencing

# 4. Install package with poetry
poetry install

# 5. Verify installation
python -c "import ehrsequencing; print(f'✅ EHR Sequencing v{ehrsequencing.__version__} ready!')"
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

### Basic Usage

```python
from ehrsequencing.data import SyntheaAdapter, VisitGrouper, PatientSequenceBuilder

# 1. Load EHR data
adapter = SyntheaAdapter('data/synthea/')
patients = adapter.load_patients(limit=100)
events = adapter.load_events(patient_ids=[p.patient_id for p in patients])

# 2. Group events into visits (with semantic code ordering)
grouper = VisitGrouper(strategy='hybrid', preserve_code_types=True)
patient_visits = grouper.group_by_patient(events)

# 3. Build patient sequences
builder = PatientSequenceBuilder(max_visits=50, max_codes_per_visit=100)
vocab = builder.build_vocabulary(patient_visits, min_frequency=5)
sequences = builder.build_sequences(patient_visits, min_visits=2)

# 4. Create PyTorch dataset
dataset = builder.create_dataset(sequences)
print(f"Created dataset with {len(dataset)} patients")
print(f"Vocabulary size: {len(vocab)}")

# 5. Train model (coming soon)
# from ehrsequencing.models import LSTMProgressionModel
# model = LSTMProgressionModel(vocab_size=len(vocab), ...)
# model.train(dataset)
```

---

## Features

### Current (Phase 1) - ✅ Complete
- ✅ Modern project structure
- ✅ Poetry + Conda dependency management
- ✅ Platform-specific environments (macOS, CUDA, CPU)
- ✅ Comprehensive documentation
- ✅ Data adapters (Synthea implemented)
- ✅ Visit grouping with semantic code ordering
- ✅ Patient sequence builder
- ✅ Unit tests for data pipeline
- ✅ LSTM baseline model with training utilities
- ✅ Data exploration notebook

### Planned (Phase 2-5)
- ⬜ Med2Vec code embeddings
- ⬜ LSTM and Transformer patient encoders
- ⬜ BEHRT (BERT for EHR) implementation
- ⬜ Disease trajectory prediction
- ⬜ Phenotype discovery algorithms
- ⬜ Interactive visualizations

---

## Project Structure

```
ehr-sequencing/
├── src/ehrsequencing/          # Main package
│   ├── data/                   # Data loading & preprocessing
│   ├── embeddings/             # Code embeddings (Med2Vec, etc.)
│   ├── models/                 # Sequence models (LSTM, Transformer, BEHRT)
│   ├── clustering/             # Disease subtyping
│   ├── evaluation/             # Metrics & visualization
│   └── utils/                  # Utilities
│
├── notebooks/                  # Jupyter notebooks
├── examples/                   # Production scripts
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── legacy/                     # Original codebase (preserved)
│
├── pyproject.toml              # Poetry configuration
├── environment.yml             # Conda environment
└── README.md                   # This file
```

---

## Methodology

### Sequence Construction

Patient histories are converted into sequences of medical codes:

```
Patient Timeline:
  2020-01-15: [LOINC:4548-4, SNOMED:44054006, RXNORM:860975]
  2020-06-15: [LOINC:4548-4, LOINC:2339-0]
  2020-12-15: [SNOMED:44054006, RXNORM:860975]
       ↓
Sequence: [V1, V2, V3, ...]
```

### Code Embeddings

Medical codes are embedded into continuous vector space using:
- **Med2Vec** - Skip-gram model for code co-occurrence
- **Graph embeddings** - Leveraging medical ontologies
- **Pre-trained models** - BioBERT, ClinicalBERT

### Sequence Models

Patient sequences are encoded using:
- **LSTM** - Recurrent models for temporal dependencies
- **Transformers** - Attention-based models
- **BEHRT** - BERT adapted for EHR with age/visit embeddings

### Applications

- **Diagnosis Prediction** - Predict future conditions
- **Mortality Risk** - Estimate survival probability
- **Disease Subtyping** - Discover phenotypes via clustering
- **Trajectory Analysis** - Understand disease progression patterns

---

## Documentation

- **[Project Setup](dev/workflow/PROJECT_SETUP.md)** - Complete setup guide
- **[Roadmap](dev/workflow/ROADMAP.md)** - Development plan
- **[Methods](docs/methods/)** - Methodology documentation
- **[Tutorials](docs/tutorials/)** - Getting started guides

---

## Related Projects

- **[loinc-predictor](https://github.com/yourusername/loinc-predictor)** - LOINC code prediction and error correction
- **PyHealth** - Healthcare AI toolkit
- **MIMIC-III Benchmarks** - Standard evaluation tasks

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ehr_sequencing_2026,
  title = {EHR Sequencing: Biological Language Model for Electronic Health Records},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/ehr-sequencing}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

## Contact

- **Author:** Your Name
- **Email:** your.email@example.com
- **GitHub:** [@yourusername](https://github.com/yourusername)

---

**Status:** 🚧 Under Active Development | **Version:** 0.1.0 | **Updated:** January 2026
