# EHR Sequencing Project Setup

**Date:** January 19, 2026  
**Status:** Modernization in Progress

---

## Project Overview

**EHR Sequencing** is a biological language model approach for Electronic Health Records, treating medical codes as "words" and patient histories as "documents" to model disease progression and perform temporal phenotyping.

### Core Concept

```
Biological Sequences (DNA/Protein)  →  Language Models (NLP)
              ↓                                    ↓
Medical Code Sequences (EHR)        →  EHR Sequencing Models
```

---

## Legacy Structure (Preserved)

The original `temporal-phenotyping` project has been renamed to `ehr-sequencing` and contains:

```
ehr-sequencing/ (legacy)
├── seqmaker/          # Sequence construction (114 files)
├── cluster/           # Disease subtyping (46 files)
├── batchpheno/        # Batch phenotyping (17 files)
├── classifier/        # Temporal classifiers (5 files)
├── cohort_selection/  # Cohort definition (2 files)
├── pattern/           # Pattern mining (7 files)
├── demo/              # Demos and examples (108 files)
└── ...
```

**This legacy code is preserved in place** - we're building a modern structure alongside it.

---

## Modern Structure (New)

Following the same pattern as `loinc-predictor`:

```
ehr-sequencing/
├── src/ehrsequencing/          # Modern package
│   ├── __init__.py
│   ├── data/                   # Data adapters (duplicated from loinc-predictor)
│   │   ├── adapters/
│   │   │   ├── base.py
│   │   │   ├── synthea.py
│   │   │   └── mimic3.py
│   │   ├── schema.py           # ClinicalEvent, PatientSequence
│   │   └── sequences.py        # Sequence builder
│   │
│   ├── embeddings/             # Code embeddings
│   │   ├── med2vec.py          # Skip-gram embeddings
│   │   ├── graph.py            # Graph-based embeddings
│   │   └── pretrained.py       # Load pretrained embeddings
│   │
│   ├── models/                 # Sequence models
│   │   ├── lstm.py             # PatientLSTM
│   │   ├── transformer.py      # PatientTransformer, BEHRT
│   │   ├── hierarchical.py     # Hierarchical encoders
│   │   └── trajectory.py       # Disease progression models
│   │
│   ├── clustering/             # Disease subtyping
│   │   ├── phenotypes.py       # Phenotype discovery
│   │   └── trajectories.py     # Trajectory clustering
│   │
│   ├── evaluation/             # Metrics and evaluation
│   │   ├── metrics.py
│   │   └── visualization.py
│   │
│   └── utils/                  # Utilities
│       ├── tokenization.py
│       └── temporal.py
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_sequence_construction.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_models.ipynb
│   └── 05_disease_subtyping.ipynb
│
├── examples/                   # Production scripts
│   ├── train_med2vec.py
│   ├── train_patient_transformer.py
│   └── discover_phenotypes.py
│
├── docs/                       # Public documentation
│   ├── README.md
│   ├── methods/
│   │   ├── sequence-construction.md
│   │   ├── embeddings.md
│   │   └── disease-progression.md
│   ├── datasets/
│   │   └── synthea-setup.md
│   └── tutorials/
│       └── getting-started.md
│
├── dev/                        # Private documentation (gitignored)
│   ├── workflow/
│   │   ├── PROJECT_SETUP.md    # This file
│   │   ├── ROADMAP.md
│   │   └── PHASE1_PLAN.md
│   └── notes/
│       └── research-ideas.md
│
├── tests/                      # Unit tests
│   ├── test_data/
│   ├── test_embeddings/
│   └── test_models/
│
├── scripts/                    # Utility scripts
│   └── generate_synthea_sequences.sh
│
├── data/                       # Data directory (gitignored)
├── checkpoints/                # Model checkpoints (gitignored)
├── results/                    # Experiment results (gitignored)
│
├── legacy/                     # Legacy code (preserved)
│   ├── seqmaker/
│   ├── cluster/
│   └── ...
│
├── pyproject.toml              # Poetry configuration
├── environment.yml             # Conda/Mamba environment
├── .gitignore
└── README.md
```

---

## Technology Stack

### Core Dependencies

**Data & ML:**
- `pandas`, `numpy` - Data manipulation
- `torch` - Deep learning framework
- `scikit-learn` - Classical ML
- `gensim` - Word2Vec-style embeddings

**Medical/Bio:**
- `biopython` - Sequence analysis utilities
- `lifelines` - Survival analysis

**Visualization:**
- `matplotlib`, `seaborn` - Plotting
- `plotly` - Interactive visualizations
- `umap-learn` - Dimensionality reduction

**Development:**
- `pytest` - Testing
- `black`, `ruff` - Code formatting
- `mypy` - Type checking
- `jupyter` - Notebooks

---

## Environment Setup

### Conda Environment

```yaml
# environment.yml
name: ehrsequencing
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - pytorch>=2.0
  - pandas
  - numpy
  - scikit-learn
  - jupyter
  - pip
  - pip:
    - poetry
```

### Poetry Dependencies

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.10"
torch = "^2.0.0"
pandas = "^2.0.0"
numpy = "^1.24.0"
scikit-learn = "^1.3.0"
gensim = "^4.3.0"
lifelines = "^0.27.0"
umap-learn = "^0.5.0"
plotly = "^5.17.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.7.0"
ruff = "^0.0.285"
mypy = "^1.5.0"
jupyter = "^1.0.0"
```

---

## Installation

```bash
# 1. Create conda environment
cd ~/work/ehr-sequencing
mamba env create -f environment.yml
mamba activate ehrsequencing

# 2. Install dependencies with poetry
poetry install

# 3. Install package in development mode
pip install -e .

# 4. Verify installation
python -c "import ehrsequencing; print('✅ EHR Sequencing ready!')"
```

---

## Project Phases

### Phase 1: Foundation (Weeks 1-2)
- ✅ Project structure setup
- ✅ Environment configuration
- ⬜ Data adapters (Synthea, MIMIC-III)
- ⬜ Sequence construction pipeline
- ⬜ Basic visualization tools

### Phase 2: Embeddings (Weeks 3-4)
- ⬜ Med2Vec implementation
- ⬜ Code embedding training
- ⬜ Embedding evaluation
- ⬜ Visualization (t-SNE, UMAP)

### Phase 3: Sequence Models (Weeks 5-6)
- ⬜ PatientLSTM implementation
- ⬜ PatientTransformer implementation
- ⬜ BEHRT adaptation
- ⬜ Pre-training objectives (MLM, next visit)

### Phase 4: Disease Progression (Weeks 7-8)
- ⬜ Trajectory prediction models
- ⬜ Multi-horizon forecasting
- ⬜ Survival analysis integration
- ⬜ Temporal phenotyping

### Phase 5: Disease Subtyping (Weeks 9-10)
- ⬜ Clustering algorithms
- ⬜ Phenotype discovery
- ⬜ Trajectory analysis
- ⬜ Clinical validation

---

## Key Differences from loinc-predictor

| Aspect | loinc-predictor | ehr-sequencing |
|--------|----------------|----------------|
| **Goal** | LOINC code prediction | Disease progression modeling |
| **Data View** | Cross-sectional (test → code) | Longitudinal (patient timeline) |
| **Primary Task** | Classification/matching | Sequence modeling |
| **Models** | Random Forest, Siamese networks | LSTM, Transformers, BEHRT |
| **Output** | Predicted LOINC code | Patient embedding, trajectory |
| **Evaluation** | Accuracy, F1, MRR | AUC, time-to-event, clustering |

---

## Relationship to loinc-predictor

### Shared Components (Duplicated)
- Data adapters (Synthea, MIMIC-III)
- Clinical event schema
- Basic preprocessing utilities

### Independent Components
- **ehr-sequencing specific:**
  - Sequence construction strategies
  - Code embeddings (Med2Vec)
  - Temporal encoders (LSTM, Transformer)
  - Disease subtyping algorithms

- **loinc-predictor specific:**
  - LOINC-specific features (TF-IDF, string distance)
  - Classifier Array
  - Matchmaker approach
  - Siamese networks for code matching

### Potential Integration
- Use corrected LOINC codes from `loinc-predictor` in sequences
- Share evaluation metrics and visualization tools
- Cross-project benchmarking on same datasets

---

## Development Workflow

### Parallel Development with loinc-predictor

Both projects will be developed simultaneously:

**Week 1-2:**
- **ehr-sequencing:** Setup + data adapters
- **loinc-predictor:** Phase 3 feature engineering

**Week 3-4:**
- **ehr-sequencing:** Med2Vec embeddings
- **loinc-predictor:** Classifier Array implementation

**Week 5-6:**
- **ehr-sequencing:** Sequence models (LSTM, Transformer)
- **loinc-predictor:** Siamese networks

**Week 7-8:**
- **ehr-sequencing:** Disease progression models
- **loinc-predictor:** Hybrid ensemble

---

## Git Strategy

### Branch Structure
```
main                    # Stable releases
├── dev                 # Active development
├── feature/embeddings  # Feature branches
└── feature/models
```

### Commit Convention
```
feat: Add Med2Vec embedding model
fix: Correct sequence padding logic
docs: Update embedding tutorial
test: Add tests for PatientLSTM
refactor: Simplify tokenization pipeline
```

---

## Documentation Strategy

### Public Documentation (docs/)
- Method descriptions
- API reference
- Tutorials and examples
- Dataset guides

### Private Documentation (dev/)
- Planning documents
- Research notes
- Experiment logs
- Internal roadmaps

### Code Documentation
- Docstrings for all public functions
- Type hints throughout
- Inline comments for complex logic
- README in each major directory

---

## Testing Strategy

### Unit Tests
```python
# tests/test_embeddings/test_med2vec.py
def test_med2vec_training():
    model = Med2Vec(vocab_size=100, embed_dim=128)
    # ... test training logic

def test_embedding_similarity():
    # Test that similar codes have similar embeddings
    pass
```

### Integration Tests
```python
# tests/test_integration/test_pipeline.py
def test_end_to_end_pipeline():
    # Load data → Build sequences → Train model → Predict
    pass
```

### Notebook Tests
- Ensure all notebooks run without errors
- Check outputs are reasonable

---

## Success Metrics

### Technical Metrics
- **Embedding quality:** Nearest neighbor accuracy > 80%
- **Prediction accuracy:** AUC > 0.85 for diagnosis prediction
- **Clustering quality:** Silhouette score > 0.5
- **Training time:** < 1 hour on GPU for 10K patients

### Research Metrics
- **Novel phenotypes discovered:** > 5 clinically meaningful subtypes
- **Trajectory prediction:** Concordance index > 0.75
- **Interpretability:** Attention patterns align with clinical knowledge

---

## Next Steps

1. **Immediate (Today):**
   - Create directory structure
   - Set up pyproject.toml and environment.yml
   - Create .gitignore
   - Initialize git (if not already)

2. **This Week:**
   - Implement data adapters
   - Create sequence builder
   - Write first notebook (data exploration)
   - Set up basic tests

3. **Next Week:**
   - Implement Med2Vec
   - Train embeddings on Synthea data
   - Visualize embeddings
   - Document embedding approach

---

## References

### Key Papers
1. **Med2Vec** - Choi et al., "Multi-layer Representation Learning for Medical Concepts" (2016)
2. **BEHRT** - Li et al., "BEHRT: Transformer for Electronic Health Records" (2020)
3. **RETAIN** - Choi et al., "RETAIN: An Interpretable Predictive Model" (2016)

### Related Projects
- **loinc-predictor** - Sibling project for LOINC code prediction
- **PyHealth** - EHR modeling toolkit
- **MIMIC-III Benchmarks** - Standard evaluation tasks

---

**Document Version:** 1.0  
**Last Updated:** January 19, 2026  
**Maintained By:** EHR Sequencing Team
