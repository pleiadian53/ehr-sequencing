# EHR Sequencing Project Roadmap

**Last Updated:** January 24, 2026

---

## Vision

Build a biological language model for Electronic Health Records that treats medical codes as "words" and patient histories as "documents" to enable:
- Disease progression modeling
- Temporal phenotyping
- Disease subtype discovery
- Clinical trajectory prediction

---

## Current Status: Phase 1.5 - Survival Analysis

**Progress:** Phase 1 Complete (100%) | Phase 1.5 In Progress (80%)

**Recent Updates (January 24, 2026):**
- ✅ Implemented discrete-time survival LSTM model
- ✅ Created synthetic survival outcome generator with validated correlation
- ✅ Developed fast validation script for synthetic data quality
- ✅ Implemented save/load functionality for pre-generated outcomes
- ✅ Resolved C-index calculation issues (achieved 0.65-0.70)
- ✅ Created comprehensive survival analysis notebook
- 🔄 Next: Documentation and tutorial materials

---

## Phase 1: Foundation & Data Pipeline (Weeks 1-2)

### Objectives
- Modernize project structure
- Implement data adapters for Synthea and MIMIC-III
- Build sequence construction pipeline
- Create initial notebooks and documentation

### Tasks

#### Week 1: Project Structure
- [x] Create `dev/workflow/PROJECT_SETUP.md`
- [ ] Create directory structure (`src/`, `docs/`, `notebooks/`, `tests/`)
- [ ] Set up `pyproject.toml` and `environment.yml`
- [ ] Configure `.gitignore`
- [ ] Move legacy code to `legacy/` directory
- [ ] Create initial README.md

#### Week 2: Data Pipeline
- [ ] Implement `src/ehrsequencing/data/schema.py`
  - `ClinicalEvent` dataclass
  - `PatientSequence` dataclass
  - `Visit` dataclass
- [ ] Implement `src/ehrsequencing/data/adapters/base.py`
  - Abstract `DataSourceAdapter` class
- [ ] Implement `src/ehrsequencing/data/adapters/synthea.py`
  - Load observations, conditions, medications
  - Merge into unified event stream
- [ ] Implement `src/ehrsequencing/data/sequences.py`
  - Sequence construction strategies (flat, visit-grouped, hierarchical)
  - Tokenization utilities
  - Vocabulary building
- [ ] Create `notebooks/01_data_exploration.ipynb`
- [ ] Write unit tests for data pipeline

### Deliverables
- ✅ Modern project structure
- ✅ Platform-specific environments (macOS/CUDA/CPU)
- ✅ Comprehensive installation documentation
- ✅ Working data adapters (Synthea)
- ✅ Visit grouping with semantic code ordering
- ✅ Patient sequence builder with PyTorch dataset
- ✅ Unit tests for data pipeline
- ✅ LSTM baseline model with training utilities
- ✅ Data exploration notebook

**Phase 1 Status: ✅ COMPLETE (100%)**

---

## Phase 1.5: Survival Analysis (Week 2.5)

### Objectives
- Implement discrete-time survival analysis for EHR sequences
- Create synthetic outcome generator with realistic risk-time correlation
- Develop validation tools for synthetic data quality
- Build complete training pipeline with proper evaluation metrics

### Tasks Completed
- [x] Implement `DiscreteTimeSurvivalLSTM` model
  - Visit-level hazard prediction
  - Proper handling of variable-length sequences
  - Integration with PyTorch DataLoader
- [x] Implement `DiscreteTimeSurvivalLoss`
  - Negative log-likelihood for discrete-time survival
  - Proper masking for censored observations
  - Numerical stability (epsilon clamping)
- [x] Create `DiscreteTimeSurvivalGenerator`
  - Risk factor computation (comorbidity, frequency, diversity)
  - Controlled noise for realistic correlation (r = -0.5)
  - Configurable censoring rate and time scale
- [x] Develop `test_synthetic_outcomes.py`
  - Fast validation without full notebook execution
  - Correlation diagnostics and distribution checks
  - Save/load functionality for pre-validated data
- [x] Resolve C-index calculation issues
  - Identified length bias in cumulative hazard approach
  - Implemented fixed-horizon risk score (mean of first 10 visits)
  - Achieved C-index 0.65-0.70 (aligned with synthetic correlation)
- [x] Create comprehensive notebook `01_discrete_time_survival_lstm.ipynb`
  - Educational content on C-index and survival analysis
  - Complete workflow from data loading to model evaluation
  - Visualization of training progress and outcomes

### Key Learnings
1. **Length bias in survival models**: Summing hazards across all visits creates bias where longer sequences get higher cumulative risk
2. **Risk score formulation**: Fixed-horizon approach (first N visits) removes length bias while capturing baseline risk
3. **Synthetic data validation**: Strong correlation (r < -0.5) is essential for meaningful model training
4. **C-index interpretation**: With r=-0.5 synthetic correlation, C-index of 0.65-0.70 is realistic and appropriate

### Deliverables
- ✅ Discrete-time survival LSTM model
- ✅ Synthetic outcome generator with validation
- ✅ Fast validation script with save/load
- ✅ Complete survival analysis notebook
- 🔄 Tutorial documentation (in progress)

**Phase 1.5 Status: 🔄 IN PROGRESS (80%)**

---

## Phase 2: Code Embeddings (Weeks 3-4)

### Objectives
- Implement Med2Vec (skip-gram) embeddings
- Train embeddings on Synthea data
- Evaluate and visualize embeddings
- Create embedding utilities

### Tasks

#### Week 3: Med2Vec Implementation
- [ ] Implement `src/ehrsequencing/embeddings/med2vec.py`
  - Skip-gram architecture
  - Negative sampling
  - Training loop
- [ ] Implement `src/ehrsequencing/embeddings/utils.py`
  - Context window extraction
  - Vocabulary management
  - Embedding persistence
- [ ] Create `examples/train_med2vec.py`
- [ ] Create `notebooks/02_embeddings.ipynb`

#### Week 4: Evaluation & Visualization
- [ ] Implement embedding evaluation metrics
  - Nearest neighbor accuracy
  - Analogy tasks
  - Clustering quality
- [ ] Implement visualization tools
  - t-SNE projection
  - UMAP projection
  - Interactive plots (Plotly)
- [ ] Train embeddings on full Synthea dataset
- [ ] Document embedding approach in `docs/methods/embeddings.md`

### Deliverables
- ⬜ Med2Vec implementation
- ⬜ Trained embeddings (128-dim)
- ⬜ Embedding evaluation results
- ⬜ Visualization notebook
- ⬜ Documentation

---

## Phase 3: Sequence Encoders (Weeks 5-6)

### Objectives
- Implement LSTM and Transformer patient encoders
- Add temporal encoding (age, time deltas)
- Pre-train with self-supervised objectives
- Evaluate patient representations

### Tasks

#### Week 5: LSTM & Transformer
- [ ] Implement `src/ehrsequencing/models/lstm.py`
  - `PatientLSTM` class
  - Bidirectional encoding
  - Variable-length sequence handling
- [ ] Implement `src/ehrsequencing/models/transformer.py`
  - `PatientTransformer` class
  - Positional encoding
  - Attention pooling
- [ ] Implement `src/ehrsequencing/models/behrt.py`
  - BEHRT architecture
  - Age + visit + position embeddings
- [ ] Create `notebooks/03_sequence_models.ipynb`

#### Week 6: Pre-training
- [ ] Implement pre-training objectives
  - Masked language modeling (MLM)
  - Next visit prediction
  - Contrastive learning
- [ ] Train models on Synthea data
- [ ] Evaluate patient representations
  - Clustering by disease
  - Similarity retrieval
- [ ] Create `examples/train_patient_transformer.py`

### Deliverables
- ⬜ LSTM encoder
- ⬜ Transformer encoder
- ⬜ BEHRT implementation
- ⬜ Pre-trained models
- ⬜ Evaluation results

---

## Phase 4: Disease Progression (Weeks 7-8)

### Objectives
- Implement trajectory prediction models
- Multi-horizon forecasting (30d, 90d, 365d)
- Survival analysis integration
- Clinical validation

### Tasks

#### Week 7: Trajectory Models
- [ ] Implement `src/ehrsequencing/models/trajectory.py`
  - `DiseaseTrajectoryModel` class
  - Multi-horizon prediction heads
  - Time-to-event modeling
- [ ] Implement evaluation metrics
  - AUC for disease prediction
  - Concordance index for survival
  - Calibration metrics
- [ ] Create `notebooks/04_disease_progression.ipynb`

#### Week 8: Clinical Validation
- [ ] Define clinical prediction tasks
  - Diabetes onset
  - Heart failure
  - Mortality
- [ ] Train and evaluate models
- [ ] Compare with baselines
  - Logistic regression
  - Random forest
  - Simple LSTM
- [ ] Document results in `docs/methods/disease-progression.md`

### Deliverables
- ⬜ Trajectory prediction model
- ⬜ Multi-horizon forecasting
- ⬜ Clinical validation results
- ⬜ Comparison with baselines

---

## Phase 5: Disease Subtyping (Weeks 9-10)

### Objectives
- Discover disease subtypes via clustering
- Analyze temporal trajectories
- Validate clinical meaningfulness
- Create visualization tools

### Tasks

#### Week 9: Clustering & Phenotyping
- [ ] Implement `src/ehrsequencing/clustering/phenotypes.py`
  - K-means on patient embeddings
  - Hierarchical clustering
  - DBSCAN for outlier detection
- [ ] Implement `src/ehrsequencing/clustering/trajectories.py`
  - Trajectory clustering
  - Temporal pattern mining
- [ ] Create `notebooks/05_disease_subtyping.ipynb`

#### Week 10: Validation & Visualization
- [ ] Evaluate clustering quality
  - Silhouette score
  - Davies-Bouldin index
  - Clinical coherence
- [ ] Characterize discovered subtypes
  - Demographics
  - Code distributions
  - Outcomes
- [ ] Create interactive visualizations
- [ ] Document findings in `docs/methods/disease-subtyping.md`

### Deliverables
- ⬜ Clustering algorithms
- ⬜ Discovered phenotypes
- ⬜ Clinical validation
- ⬜ Visualization tools

---

## Future Phases

### Phase 6: Advanced Models (Weeks 11-12)
- Graph neural networks for code relationships
- Hierarchical attention networks
- Multi-task learning
- Transfer learning from clinical BERT

### Phase 7: Production Deployment (Weeks 13-14)
- Model serving API
- Real-time inference
- Monitoring and logging
- Documentation and tutorials

### Phase 8: Research Extensions (Ongoing)
- Novel pre-training objectives
- Cross-dataset evaluation (MIMIC-III, MIMIC-IV)
- Interpretability methods
- Clinical collaborations

---

## Success Metrics

### Technical Metrics
- **Embedding quality:** Nearest neighbor accuracy > 80%
- **Prediction accuracy:** AUC > 0.85 for diagnosis prediction
- **Clustering quality:** Silhouette score > 0.5
- **Training efficiency:** < 1 hour on GPU for 10K patients
- **Inference speed:** < 100ms per patient

### Research Metrics
- **Novel phenotypes:** > 5 clinically meaningful subtypes discovered
- **Trajectory prediction:** Concordance index > 0.75
- **Interpretability:** Attention patterns align with clinical knowledge
- **Generalization:** Performance within 5% across datasets

### Code Quality
- **Test coverage:** > 80%
- **Documentation:** All public APIs documented
- **Type hints:** 100% coverage
- **Code style:** Black + Ruff compliant

---

## Risk Management

### Technical Risks
- **Computational resources:** Mitigate with efficient implementations, mixed precision
- **Data quality:** Validate Synthea realism, plan MIMIC-III validation
- **Model complexity:** Start simple (LSTM), iterate to complex (BEHRT)

### Research Risks
- **Overfitting:** Use cross-validation, regularization, early stopping
- **Interpretability:** Implement attention visualization, SHAP values
- **Clinical validity:** Collaborate with domain experts, literature review

---

## Dependencies

### Data Sources
- **Synthea:** Synthetic patient data (no barriers)
- **MIMIC-III:** Real ICU data (requires credentialing)
- **MIMIC-IV:** Newer version (optional)

### Related Projects
- **loinc-predictor:** Provides corrected LOINC codes
- **PyHealth:** Reference implementations
- **MIMIC-III Benchmarks:** Standard evaluation tasks

---

## Documentation Plan

### Public Documentation (docs/)
- [ ] `docs/README.md` - Overview and navigation
- [ ] `docs/methods/sequence-construction.md`
- [ ] `docs/methods/embeddings.md`
- [ ] `docs/methods/disease-progression.md`
- [ ] `docs/methods/disease-subtyping.md`
- [ ] `docs/tutorials/getting-started.md`
- [ ] `docs/tutorials/custom-models.md`
- [ ] `docs/api/reference.md`

### Private Documentation (dev/)
- [x] `dev/workflow/PROJECT_SETUP.md`
- [x] `dev/workflow/ROADMAP.md` (this file)
- [ ] `dev/workflow/PHASE1_PLAN.md`
- [ ] `dev/workflow/PHASE2_PLAN.md`
- [ ] `dev/notes/research-ideas.md`
- [ ] `dev/notes/experiment-log.md`

---

## Parallel Development with loinc-predictor

Both projects will be developed simultaneously:

| Week | ehr-sequencing | loinc-predictor |
|------|----------------|-----------------|
| 1-2  | Foundation & data pipeline | Phase 3: Feature engineering |
| 3-4  | Code embeddings (Med2Vec) | Classifier Array |
| 5-6  | Sequence models (LSTM, Transformer) | Siamese networks |
| 7-8  | Disease progression | Matchmaker completion |
| 9-10 | Disease subtyping | Hybrid ensemble |

---

## Next Actions

**Immediate (Today):**
1. Create directory structure
2. Set up `pyproject.toml` and `environment.yml`
3. Create `.gitignore`
4. Initialize package structure

**This Week:**
1. Implement data schema
2. Implement Synthea adapter
3. Build sequence construction pipeline
4. Create first notebook

**Next Week:**
1. Start Med2Vec implementation
2. Set up training pipeline
3. Begin embedding evaluation

---

**Roadmap Version:** 1.0  
**Maintained By:** EHR Sequencing Team  
**Review Frequency:** Weekly
