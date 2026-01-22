# Legacy Code Archive

**Date Archived:** January 19, 2026  
**Reason:** Project modernization and restructuring

---

## Overview

This directory contains the original implementation of the EHR Sequencing (formerly Temporal Phenotyping) project. The code was developed prior to the emergence of modern machine learning and deep learning techniques such as transformers, and represents a mature, functional but outdated system for temporal phenotyping, disease subtyping, and patient clustering from EHR data.

**Important:** This code is preserved for reference and historical purposes. New development should use the modernized structure under `src/ehrsequencing/`.

---

## Legacy Directory Structure

### Core Modules

#### `seqmaker/` - Sequence Generation & Analysis (114 files)
**Purpose:** Core sequence generation, classification, and clustering

**Key Files:**
- `seqMaker.py`, `seqMaker2.py`, `seqMaker3.py` - Sequence generation from EHR data
- `seqReader.py` - Read and parse generated sequences
- `seqClassify.py` - Sequence classification (multiple variants)
- `seqCluster.py` - Sequence clustering algorithms
- `seqAnalyzer.py` - Sequence analysis and visualization
- `cohort.py` - Cohort selection and management (178KB - major module)
- `pathAnalyzer.py` - Disease pathway analysis (372KB - major module)
- `vector.py` - Vector representations and embeddings (131KB)
- `evaluate.py` - Model evaluation metrics
- `algorithms.py` - Core algorithms for sequence processing
- `word2vec.py` - Word2Vec embeddings for medical codes

**Technologies:**
- Gensim (Word2Vec)
- Scikit-learn (classification, clustering)
- Deep learning (Keras/TensorFlow)
- UMAP, t-SNE (dimensionality reduction)

#### `cluster/` - Clustering Algorithms (46 files)
**Purpose:** Patient clustering and phenotype discovery

**Key Files:**
- `cluster.py` - Main clustering implementation
- `analyzer.py` - Cluster analysis and interpretation
- `learn_manifold.py` - Manifold learning (UMAP, t-SNE)
- `gap_stats.py` - Gap statistics for optimal cluster number
- `sampling.py` - Sampling strategies for large datasets

**Methods:**
- K-means, hierarchical clustering
- DBSCAN, spectral clustering
- Gap statistics for cluster validation

#### `classifier/` - Classification Models (5 files)
**Purpose:** Supervised classification of patient sequences

**Key Files:**
- `utils.py` - Classification utilities
- `evaluate.py` - Evaluation metrics
- `plot_classifier_comparison.py` - Compare classifier performance

**Models:**
- Random Forest, Gradient Boosting
- Logistic Regression, SVM
- Deep neural networks

#### `batchpheno/` - Batch Phenotyping (17 files)
**Purpose:** Large-scale phenotyping and cohort analysis

**Key Files:**
- `analyzer.py` - Phenotype analysis (52KB)
- `icd9utils.py` - ICD-9 code utilities and hierarchy
- `qrymed2.py`, `qrymed3.py` - Medical code querying
- `sampling.py` - Sampling strategies
- `dfUtils.py` - DataFrame utilities for EHR data

**Data:**
- `ICD9_descriptions` - ICD-9 code descriptions (1.2MB)
- `ICD9_parent_child_relations` - ICD-9 hierarchy (248KB)

#### `pattern/` - Pattern Recognition (7 files)
**Purpose:** Temporal pattern discovery in sequences

**Key Files:**
- Pattern mining algorithms
- Motif discovery
- Sequence alignment

#### `sampler/` - Sampling Strategies (4 files)
**Purpose:** Data sampling for large EHR datasets

**Methods:**
- Stratified sampling
- Temporal sampling
- Balanced sampling for imbalanced datasets

### Supporting Modules

#### `ontology/` - Medical Ontologies (2 files)
**Purpose:** Medical code hierarchies and relationships
- ICD-9/10 hierarchies
- SNOMED CT relationships
- LOINC structure

#### `pyumls/` - UMLS Integration (25 files)
**Purpose:** Unified Medical Language System integration
- Concept mapping
- Semantic type extraction
- Cross-terminology mapping

#### `cohort_selection/` - Cohort Selection (2 files)
**Purpose:** Define and extract patient cohorts
- Inclusion/exclusion criteria
- Temporal constraints
- Comorbidity filtering

#### `system/` - System Utilities (2 files)
**Purpose:** System-level utilities
- File I/O
- Configuration management
- Logging

### Configuration & Scripts

#### `config/` - Configuration Files (6 files)
**Purpose:** Project configuration
- Database connections
- File paths
- Model hyperparameters

#### `bin/` - Executable Scripts (1 file)
**Purpose:** Command-line tools

#### `set_env.sh`, `set_env_template.sh`
**Purpose:** Environment setup scripts
- Python path configuration
- Environment variables

### Analysis & Experiments

#### `demo/` - Demonstrations (109 files)
**Purpose:** Example analyses and experiments
- Proof-of-concept implementations
- Experimental features
- Analysis notebooks

#### `sprint/` - Sprint Deliverables (28 files)
**Purpose:** Development sprint outputs
- Feature implementations
- Experiment results

#### `diabetes/` - Diabetes-Specific Analysis (1 file)
**Purpose:** Disease-specific implementations

#### `ref/` - Reference Materials (9 files)
**Purpose:** Reference implementations and documentation

#### `qrymed/` - Medical Query Tools (1 file)
**Purpose:** Query medical databases

---

## Key Technologies Used

### Machine Learning
- **Scikit-learn** - Classification, clustering, preprocessing
- **Gensim** - Word2Vec embeddings for medical codes
- **Keras/TensorFlow** - Deep learning models
- **XGBoost/LightGBM** - Gradient boosting

### Dimensionality Reduction
- **UMAP** - Uniform Manifold Approximation and Projection
- **t-SNE** - t-Distributed Stochastic Neighbor Embedding
- **PCA** - Principal Component Analysis

### Data Processing
- **Pandas** - DataFrame operations
- **NumPy** - Numerical computations
- **SciPy** - Scientific computing

### Visualization
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive visualizations

---

## Migration Notes

### What Was Preserved
1. **All original code** - Complete implementation preserved
2. **Data files** - ICD-9 descriptions and hierarchies
3. **Configuration** - Original config files
4. **Documentation** - Inline comments and docstrings

### What Changed in Modern Implementation

#### From Legacy → Modern

**Sequence Generation:**
```python
# Legacy (seqmaker/seqMaker.py)
from seqmaker.seqMaker import SequenceMaker
seqmaker = SequenceMaker(config)
sequences = seqmaker.generate()

# Modern (src/ehrsequencing/data/sequences/)
from ehrsequencing.data.sequences import PatientSequenceBuilder
from ehrsequencing.data.adapters import SyntheaAdapter

adapter = SyntheaAdapter('data/')
events = adapter.load_events()
builder = PatientSequenceBuilder(vocab)
sequences = builder.build_sequences(events)
```

**Embeddings:**
```python
# Legacy (seqmaker/word2vec.py)
from gensim.models import Word2Vec
model = Word2Vec(sequences, size=128, window=5)

# Modern (src/ehrsequencing/embeddings/)
from ehrsequencing.embeddings import CEHRBERTWrapper
model = CEHRBERTWrapper.from_pretrained('cehrbert-base')
embeddings = model.encode_codes(code_ids)
```

**Clustering:**
```python
# Legacy (cluster/cluster.py)
from cluster.cluster import ClusterAnalyzer
analyzer = ClusterAnalyzer()
clusters = analyzer.fit(sequences)

# Modern (src/ehrsequencing/clustering/)
from ehrsequencing.clustering import TemporalPhenotyper
phenotyper = TemporalPhenotyper(visit_encoder, n_phenotypes=5)
phenotypes = phenotyper.fit(patient_sequences)
```

---

## Key Differences: Legacy vs Modern

| Aspect | Legacy | Modern |
|--------|--------|--------|
| **Structure** | Flat modules | Hierarchical packages |
| **Embeddings** | Word2Vec (gensim) | Pre-trained CEHR-BERT |
| **Sequences** | Flat code sequences | Visit-grouped hierarchical |
| **Data Loading** | Custom parsers | Standardized adapters |
| **Configuration** | Shell scripts | pyproject.toml + environment.yml |
| **Dependencies** | requirements.txt | Poetry + Conda |
| **Testing** | Ad-hoc scripts | pytest framework |
| **Documentation** | Inline comments | Sphinx + Markdown docs |

---

## How to Use Legacy Code

### If You Need to Reference Legacy Implementation

1. **Read the code** - All original implementations are preserved
2. **Extract algorithms** - Core algorithms can be adapted to modern structure
3. **Compare approaches** - Understand design decisions

### Running Legacy Code (Not Recommended)

If you absolutely need to run legacy code:

```bash
# Set up legacy environment
cd /Users/pleiadian53/work/ehr-sequencing/legacy
source set_env.sh

# Legacy Python path includes legacy modules
export PYTHONPATH="${PYTHONPATH}:/Users/pleiadian53/work/ehr-sequencing/legacy"

# Run legacy scripts
python seqmaker/seqMaker.py
```

**Warning:** Legacy code may have dependencies on old library versions and may not work with modern Python environments.

---

## Migration Checklist

### Completed ✅
- [x] Move all legacy code to `legacy/` directory
- [x] Create modern package structure under `src/ehrsequencing/`
- [x] Document legacy code organization
- [x] Set up modern dependency management (Poetry + Conda)

### In Progress 🚧
- [ ] Implement modern data adapters
- [ ] Implement visit-grouped sequence builder
- [ ] Integrate pre-trained CEHR-BERT
- [ ] Implement disease progression model

### Planned 📋
- [ ] Migrate key algorithms to modern structure
- [ ] Create comprehensive test suite
- [ ] Benchmark modern vs legacy performance
- [ ] Archive legacy code (optional)

---

## Contact & Questions

For questions about legacy code or migration:
- **Author:** Barnett Chiu
- **Email:** barnettchiu@gmail.com
- **Project:** EHR Sequencing (formerly Temporal Phenotyping)

---

## References

### Legacy Publications & Documentation
- Original temporal phenotyping methodology
- Disease subtyping algorithms
- Patient clustering approaches

### Modern Approach Documentation
- `docs/methods/modern-code-embeddings.md` - Modern embedding approaches
- `docs/methods/pretrained-models-and-disease-progression.md` - Pre-trained models
- `docs/implementation/visit-grouped-sequences.md` - Implementation plan

---

**Last Updated:** January 19, 2026  
**Status:** Archived - Reference Only
