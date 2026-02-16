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

**Phase:** BEHRT Implementation & Survival Analysis  
**Version:** 0.2.0 (Beta)  
**Status:** Active Development

**Recent Milestones (February 2026):**
- ✅ Phase 3 Complete: BEHRT (Transformer for EHR) with MLM pre-training
- ✅ Benchmarking infrastructure for model comparison
- ✅ LoRA fine-tuning support for efficient training
- ✅ Transfer learning benchmarking with domain shift scenarios
- ✅ LSTM baseline for survival analysis (C-index 0.53 on 1151 patients)
- 🔄 **Phase 4 Implemented:** BEHRTForSurvival with three loss functions (NLL, Ranking, Hybrid)
- 🎯 **Current Focus:** Validating implementations through systematic end-to-end testing

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

# 5. Train model
from ehrsequencing.models.behrt import BEHRTForMLM, BEHRTConfig
from ehrsequencing.models.survival_lstm import DiscreteTimeSurvivalLSTM

# Option A: BEHRT for MLM pre-training
config = BEHRTConfig(vocab_size=len(vocab))
model = BEHRTForMLM(config)

# Option B: LSTM for survival analysis
from ehrsequencing.models import DiscreteTimeSurvivalLSTM
model = DiscreteTimeSurvivalLSTM(vocab_size=len(vocab))

# Option C: BEHRT for survival analysis
from ehrsequencing.models import BEHRTForSurvival, BEHRTSurvivalConfig
config = BEHRTSurvivalConfig.from_pretrained_small(vocab_size=len(vocab))
model = BEHRTForSurvival(config)
```

---

## Features

### Phase 1: Foundation - ✅ Complete
- ✅ Modern project structure
- ✅ Poetry + Conda dependency management
- ✅ Platform-specific environments (macOS, CUDA, CPU)
- ✅ Comprehensive documentation
- ✅ Data adapters (Synthea implemented)
- ✅ Visit grouping with semantic code ordering
- ✅ Patient sequence builder
- ✅ Unit tests for data pipeline
- ✅ Data exploration notebook

### Phase 1.5: Survival Analysis - ✅ Complete
- ✅ Discrete-time survival LSTM model
- ✅ Survival loss and C-index metrics
- ✅ Synthetic outcome generator with validation
- ✅ Production training pipeline with early stopping
- ✅ RunPods A40 GPU training guide

### Phase 2: Code Embeddings - ⏸️ Deferred
- ⏸️ Med2Vec training implementation (skip-gram embeddings)
- ⏸️ Standalone code embedding training pipeline
- ✅ **External embeddings supported:** Can load pre-trained Med2Vec from PyHealth/HuggingFace
- ✅ **Benchmarking support:** `examples/pretrain_finetune/benchmark_pretrained_embeddings.py`
- **Note:** Training deferred in favor of Phase 3 (BEHRT with learned embeddings)
- **Rationale:** Modern transformers learn embeddings end-to-end; pre-trained embeddings available externally
- **Status:** Can use external embeddings for comparison; training implementation optional

### Phase 3: BEHRT & Benchmarking - ✅ Complete
- ✅ BEHRT (Transformer for EHR) with age/visit/segment embeddings
- ✅ Masked Language Modeling (MLM) pre-training
- ✅ LoRA fine-tuning for efficient training
- ✅ Benchmarking infrastructure (tracker, visualizer, metrics)
- ✅ PyHealth adapter for model comparison
- ✅ 3 model sizes (small/medium/large) for different hardware
- ✅ Comprehensive training and benchmarking examples

### Phase 4: BEHRT Survival Analysis - 🔄 Implemented (Validation In Progress)
- ✅ BEHRTForSurvival model with visit aggregation
- ✅ Three loss functions: NLL (calibration), Pairwise Ranking (C-index), Hybrid
- ✅ BEHRT survival dataset adapter (flattened sequences with visit boundaries)
- ✅ Training pipeline with early stopping and C-index evaluation
- ✅ Comprehensive benchmarking script comparing all loss functions
- ✅ Documentation: Loss functions and optimization tutorial
- 🔄 **Validation:** Systematic end-to-end testing via examples/ and notebooks/
- 🎯 **Next:** Complete validation, then run BEHRT vs LSTM comparison experiments

### Future Phases
- ⬜ Med2Vec code embeddings (optional baseline)
- ⬜ Disease trajectory prediction with multi-horizon forecasting
- ⬜ Phenotype discovery via clustering
- ⬜ Interactive visualizations
- ⬜ Real-world data validation (MIMIC-III/IV)

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

#### LSTM Baseline

- Recurrent models for temporal dependencies
- Visit-level encoding with mean pooling
- Discrete-time survival analysis
- Baseline for comparison

#### BEHRT (Transformer for EHR)

- Bidirectional Transformer with EHR-specific embeddings:
  - Code embeddings (medical codes)
  - Age embeddings (patient age at each visit)
  - Visit embeddings (visit sequence position)
  - Segment embeddings (visit boundaries)
- **Pre-training:** Masked Language Modeling (MLM) for self-supervised learning
- **Fine-tuning:** LoRA for efficient adaptation to downstream tasks
- **Survival Analysis:** BEHRTForSurvival with visit aggregation and hazard prediction

### Applications

- **Diagnosis Prediction** - Predict future medical codes (MLM pre-training)
- **Survival Analysis** - Discrete-time hazard prediction with multiple loss functions:
  - **NLL Loss** - Standard negative log-likelihood (optimizes calibration)
  - **Pairwise Ranking Loss** - Directly optimizes C-index (discrimination)
  - **Hybrid Loss** - Combines NLL + Ranking (best of both worlds)
  - Applications: Hospital readmission, mortality risk, disease onset
- **Model Comparison** - Benchmark BEHRT vs LSTM vs PyHealth
- **Transfer Learning** - Domain shift scenarios and embedding fine-tuning
- **Disease Subtyping** - Discover phenotypes via clustering (planned)
- **Trajectory Analysis** - Understand disease progression patterns (planned)

---

## Documentation

### Setup & Workflow
- **[Project Setup](dev/workflow/PROJECT_SETUP.md)** - Complete setup guide
- **[Roadmap](dev/workflow/ROADMAP.md)** - Development plan

### Data Generation
- **[Data Generation Guide](docs/data_generation/)** - Generating synthetic patient data with Synthea
- **[CSV Export Troubleshooting](docs/data_generation/synthea_csv_export_troubleshooting.md)** - Solving Synthea CSV export issues
- **[RunPods Training Guide](docs/runpods_training_guide.md)** - Cloud GPU training for large datasets

### Methods & Theory
- **[Methods](docs/methods/)** - Methodology documentation
- **[Discrete-Time Survival Analysis](docs/methods/discrete_time_survival_analysis/)** - Tutorial on survival analysis for EHR
- **[Loss Functions and Optimization](docs/methods/discrete_time_survival_analysis/loss_functions_and_optimization.md)** - NLL vs C-index, ranking losses, hybrid approaches
- **[Pretrained Embeddings](docs/pretrained_embeddings_guide.md)** - Using Med2Vec, CUI2Vec, Clinical BERT
- **[Benchmarking](src/ehrsequencing/benchmarks/README.md)** - Model comparison framework

### Applications

- **[Survival Analysis](docs/applications/survival_analysis.md)** - Readmission and mortality prediction
- **[Benchmarking](docs/applications/benchmarking.md)** - Comparing BEHRT vs LSTM vs PyHealth

### Examples

- **[Train BEHRT Survival](examples/survival_analysis/train_behrt_survival.py)** - Training script with NLL/Ranking/Hybrid losses
- **[Benchmark Loss Functions](examples/survival_analysis/benchmark_loss_functions.py)** - Compare loss functions for C-index optimization
- **[Transfer Learning](examples/pretrain_finetune/)** - Embedding fine-tuning and domain shift experiments

### Tutorials
- **[Tutorials](docs/tutorials/)** - Getting started guides
- **[Notebooks](notebooks/)** - Interactive demonstrations

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

**Status:** 🚧 Under Active Development | **Version:** 0.2.0 | **Updated:** February 2026
