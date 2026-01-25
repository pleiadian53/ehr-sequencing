# EHR Sequencing

**Research Framework for Longitudinal EHR Sequence Modeling**

> A comprehensive toolkit for exploring temporal representations, learning objectives, and model architectures for disease progression, survival analysis, and temporal phenotyping under censoring and irregular follow-up.

---

## Overview

EHR Sequencing applies sequence modeling techniques from genomics and NLP to Electronic Health Records, treating medical codes as "words" and patient histories as "documents" to enable:

- **Disease Progression Modeling** - Predict future diagnoses and outcomes
- **Survival Analysis** - Time-to-event modeling with proper censoring handling
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

## Key Features

### 🏥 **Comprehensive Data Pipeline**
- **Multi-source adapters**: Synthea, MIMIC-III support
- **Visit grouping**: Semantic code ordering (diagnoses → procedures → medications)
- **Flexible tokenization**: Visit-based, flat, or hierarchical sequences
- **PyTorch integration**: Ready-to-use datasets and dataloaders

### 🧬 **Survival Analysis**
- **Discrete-time survival models**: LSTM-based hazard prediction
- **Synthetic outcome generation**: Validated correlation (r = -0.5)
- **Proper censoring handling**: Negative log-likelihood loss
- **C-index evaluation**: Fixed-horizon risk scores to avoid length bias

### 🤖 **Model Architectures**
- **LSTM baseline**: Visit-level sequence encoding
- **Discrete-time survival LSTM**: Hazard prediction at each visit
- **Extensible framework**: Easy to add Transformers, BEHRT, etc.

### 📊 **Evaluation & Validation**
- **Concordance index (C-index)**: Survival model ranking quality
- **Synthetic data validation**: Fast iteration with pre-validated outcomes
- **Memory estimation**: Plan GPU requirements before training

---

## Quick Start

### Installation

```bash
# 1. Clone repository
git clone https://github.com/pleiadian53/ehr-sequencing.git
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

See [Installation Guide](INSTALL.md) for detailed instructions.

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
```

### Survival Analysis Example

```python
from ehrsequencing.models import DiscreteTimeSurvivalLSTM
from ehrsequencing.synthetic.survival import DiscreteTimeSurvivalGenerator

# Generate synthetic outcomes
generator = DiscreteTimeSurvivalGenerator(censoring_rate=0.3)
outcome = generator.generate(sequences)

# Train survival model
model = DiscreteTimeSurvivalLSTM(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=256
)

# Evaluate with C-index
# See notebooks/02_survival_analysis/ for complete workflow
```

---

## Documentation

### 📚 **Tutorials**
- [Survival Analysis: Prediction Problem](notebooks/02_survival_analysis/TUTORIAL_01_prediction_problem.md)
- [Synthetic Data Design & Labeling](notebooks/02_survival_analysis/TUTORIAL_02_synthetic_data_design.md)
- [Loss Function Formulation](notebooks/02_survival_analysis/TUTORIAL_03_loss_function.md)

### 📓 **Notebooks**
- [Discrete-Time Survival LSTM](notebooks/02_survival_analysis/01_discrete_time_survival_lstm.ipynb)

### 📖 **Methods**
- [Causal Survival Analysis](methods/causal-survival-analysis-1.md)
- [Modern Code Embeddings](methods/modern-code-embeddings.md)

### 🛠️ **Guides**
- [Data Generation with Synthea](data_generation/data_generation_guide.md)
- [Pretrained Embeddings](pretrained_embeddings_guide.md)
- [RunPods Training](runpods_training_guide.md)

---

## Project Status

**Phase:** 1.5 - Survival Analysis (80% Complete)  
**Version:** 0.1.0 (Alpha)  
**Status:** Active Development

### Recent Updates (January 2026)
- ✅ Implemented discrete-time survival LSTM model
- ✅ Created synthetic outcome generator with validated correlation
- ✅ Resolved C-index calculation issues (achieved 0.65-0.70)
- ✅ Comprehensive survival analysis tutorials
- 🔄 Next: Code embeddings (Med2Vec, BEHRT)

See the project repository for detailed development plan.

---

## Research Focus

This project explores multiple dimensions of EHR sequence modeling:

### **Temporal Representations**
- Visit-based sequences
- Flat event streams
- Hierarchical code structures
- Time-aware embeddings

### **Learning Objectives**
- Supervised prediction (disease onset, mortality)
- Self-supervised pre-training (masked language modeling)
- Survival analysis (time-to-event with censoring)
- Representation learning (patient embeddings)

### **Model Architectures**
- LSTMs (baseline and survival variants)
- Transformers (BEHRT-style)
- Graph neural networks (code relationships)
- Hybrid architectures

### **Real-World Challenges**
- Censoring (patients lost to follow-up)
- Irregular sampling (variable visit frequencies)
- Missing data (incomplete records)
- Length bias (variable sequence lengths)

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ehr_sequencing_2026,
  title = {EHR Sequencing: Research Framework for Longitudinal EHR Modeling},
  author = {EHR Sequencing Research Team},
  year = {2026},
  url = {https://github.com/pleiadian53/ehr-sequencing}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **Synthea**: Synthetic patient data generation
- **PyHealth**: Reference implementations for EHR modeling
- **Material for MkDocs**: Documentation framework with LaTeX support

---

## Contact

- **GitHub**: [pleiadian53/ehr-sequencing](https://github.com/pleiadian53/ehr-sequencing)
- **Issues**: [Report bugs or request features](https://github.com/pleiadian53/ehr-sequencing/issues)

---

**Built with ❤️ for advancing healthcare AI research**
