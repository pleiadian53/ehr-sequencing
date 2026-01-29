# Model Training Plan & GPU Resource Strategy

**Last Updated:** January 29, 2026  
**Purpose:** Define cost-effective development workflow separating local development from cloud GPU usage

---

## Core Philosophy

**Develop locally, train at scale on cloud GPUs**

- **Local (MPS/CPU)**: Model development, debugging, small-scale validation
- **RunPods (A40/A100)**: Production training, benchmarking, hyperparameter sweeps
- **Cost target**: < $20/month for GPU usage (assuming ~20-40 hours total)

---

## Development Workflow

### Stage 1: Local Development (90% of work)

**Environment**: MacBook Pro with MPS (Metal Performance Shaders)

**Activities**:
1. Model architecture implementation
2. Loss function development
3. Data pipeline debugging
4. Small-scale training (100-200 patients)
5. Unit tests and integration tests
6. Visualization and analysis tools
7. Documentation and notebooks

**Dataset**: Small Synthea subset (`all_cohorts/` - 106 patients)

**Success criteria**:
- Model trains without errors
- Loss decreases on small dataset
- Code is clean and tested
- Ready for scale-up

### Stage 2: Cloud GPU Training (10% of work, 90% of compute)

**Environment**: RunPods A40 (40GB VRAM, $0.40/hr) or A100 (80GB, $1.00/hr)

**Activities**:
1. Full-scale training (1000+ patients)
2. Hyperparameter sweeps
3. Model comparison and benchmarking
4. Production model training
5. Final evaluation and validation

**Dataset**: Large Synthea cohort (`large_cohort_1000/` - 1000 patients)

**Success criteria**:
- Achieve target metrics (C-index > 0.70, AUC > 0.85)
- Models converge within budget
- Results reproducible

---

## Phase-by-Phase Training Strategy

### Phase 1.5: Survival Analysis (Current)

#### Local Development ✅
- [x] Implement `DiscreteTimeSurvivalLSTM`
- [x] Implement `DiscreteTimeSurvivalLoss`
- [x] Create synthetic outcome generator
- [x] Train on 106 patients (local validation)
- [x] Create training script with early stopping
- [x] Develop comprehensive notebook

**Local results**: C-index 0.50-0.60 (limited by small dataset)

#### RunPods Training 🎯
- [ ] Train on 1000 patients with optimized hyperparameters
- [ ] Hyperparameter sweep (learning rate, dropout, hidden dims)
- [ ] Compare model architectures (1-layer vs 2-layer LSTM)
- [ ] Final model selection and evaluation

**Target**: C-index 0.65-0.75, training time 1-2 hours, cost $0.40-$0.80

**Scripts**:
- Local: `examples/survival_analysis/train_lstm_basic.py`
- RunPods: `examples/survival_analysis/train_lstm_runpods.py`

---

### Phase 2: Code Embeddings (Med2Vec)

#### Local Development (Weeks 3-4)
- [ ] Implement Med2Vec skip-gram architecture
- [ ] Implement negative sampling
- [ ] Create training loop with progress tracking
- [ ] Train on 106 patients (proof of concept)
- [ ] Implement embedding evaluation (nearest neighbors)
- [ ] Create visualization tools (t-SNE, UMAP)
- [ ] Write unit tests

**Local dataset**: 106 patients, ~5K unique codes
**Local training time**: 10-30 minutes
**Expected quality**: Embeddings show some structure but limited by data

#### RunPods Training (1 session, 2-4 hours)
- [ ] Train on full Synthea dataset (1000 patients, ~20K codes)
- [ ] Hyperparameter sweep:
  - Embedding dimensions: [64, 128, 256]
  - Context window: [3, 5, 10]
  - Negative samples: [5, 10, 20]
  - Learning rate: [1e-3, 5e-4, 1e-4]
- [ ] Evaluate embedding quality
- [ ] Save best embeddings for downstream tasks

**Target**: Nearest neighbor accuracy > 80%, training time 2-4 hours, cost $0.80-$1.60

**Scripts to create**:
- Local: `examples/embeddings/train_med2vec_local.py`
- RunPods: `examples/embeddings/train_med2vec_runpods.py`

---

### Phase 3: Sequence Encoders (Weeks 5-6)

#### Local Development
- [ ] Implement `PatientLSTM` encoder
- [ ] Implement `PatientTransformer` encoder
- [ ] Implement `BEHRT` (age + visit + position embeddings)
- [ ] Implement pre-training objectives:
  - Masked language modeling (MLM)
  - Next visit prediction
  - Contrastive learning
- [ ] Train on 106 patients (architecture validation)
- [ ] Debug attention mechanisms
- [ ] Create evaluation metrics
- [ ] Develop visualization tools

**Local training**: 30-60 minutes per model
**Focus**: Architecture correctness, not performance

#### RunPods Training (2-3 sessions, 6-12 hours total)

**Session 1: LSTM Pre-training** (2-3 hours)
- [ ] Train bidirectional LSTM on 1000 patients
- [ ] Pre-training objective: Next visit prediction
- [ ] Hyperparameter sweep (hidden dims, layers, dropout)
- [ ] Save best encoder

**Session 2: Transformer Pre-training** (3-4 hours)
- [ ] Train Transformer on 1000 patients
- [ ] Pre-training objective: Masked language modeling
- [ ] Hyperparameter sweep (attention heads, layers, hidden dims)
- [ ] Compare with LSTM

**Session 3: BEHRT Pre-training** (3-4 hours)
- [ ] Train BEHRT on 1000 patients
- [ ] Pre-training objective: MLM + next visit
- [ ] Fine-tune on downstream tasks
- [ ] Final model selection

**Target**: Patient embeddings cluster by disease, training time 6-12 hours, cost $2.40-$4.80

**Scripts to create**:
- Local: `examples/encoders/train_patient_lstm_local.py`
- RunPods: `examples/encoders/train_patient_lstm_runpods.py`
- RunPods: `examples/encoders/train_patient_transformer_runpods.py`
- RunPods: `examples/encoders/train_behrt_runpods.py`

---

### Phase 4: Disease Progression (Weeks 7-8)

#### Local Development
- [ ] Implement `DiseaseTrajectoryModel`
- [ ] Implement multi-horizon prediction heads (30d, 90d, 365d)
- [ ] Implement evaluation metrics (AUC, calibration)
- [ ] Train on 106 patients (sanity check)
- [ ] Create prediction visualization tools

#### RunPods Training (2-3 sessions, 8-12 hours total)

**Session 1: Baseline Models** (2-3 hours)
- [ ] Train logistic regression baselines
- [ ] Train random forest baselines
- [ ] Train simple LSTM baseline
- [ ] Establish performance benchmarks

**Session 2: Trajectory Models** (3-4 hours)
- [ ] Train trajectory model with LSTM encoder
- [ ] Train trajectory model with Transformer encoder
- [ ] Multi-horizon forecasting
- [ ] Compare with baselines

**Session 3: Clinical Validation** (3-4 hours)
- [ ] Define clinical tasks (diabetes, heart failure, mortality)
- [ ] Train task-specific models
- [ ] Evaluate on held-out test set
- [ ] Calibration analysis

**Target**: AUC > 0.85, C-index > 0.75, training time 8-12 hours, cost $3.20-$4.80

---

### Phase 5: Disease Subtyping (Weeks 9-10)

#### Local Development
- [ ] Implement clustering algorithms (K-means, hierarchical, DBSCAN)
- [ ] Implement trajectory clustering
- [ ] Create visualization tools (cluster plots, trajectory plots)
- [ ] Test on 106 patients (algorithm validation)

#### RunPods Training (2 sessions, 6-8 hours total)

**Session 1: Patient Clustering** (3-4 hours)
- [ ] Extract patient embeddings from pre-trained encoder
- [ ] Cluster 1000 patients
- [ ] Evaluate clustering quality (silhouette, Davies-Bouldin)
- [ ] Characterize discovered phenotypes

**Session 2: Trajectory Mining** (3-4 hours)
- [ ] Cluster temporal trajectories
- [ ] Identify progression patterns
- [ ] Validate clinical meaningfulness
- [ ] Create interactive visualizations

**Target**: Discover 5+ meaningful subtypes, training time 6-8 hours, cost $2.40-$3.20

---

## Cost Estimation

### Per-Phase GPU Costs (RunPods A40 @ $0.40/hr)

| Phase | Sessions | Hours | Cost | Purpose |
|-------|----------|-------|------|---------|
| 1.5 Survival | 1 | 1-2 | $0.40-$0.80 | Final benchmarking |
| 2 Embeddings | 1 | 2-4 | $0.80-$1.60 | Production embeddings |
| 3 Encoders | 3 | 6-12 | $2.40-$4.80 | Pre-training |
| 4 Progression | 3 | 8-12 | $3.20-$4.80 | Clinical validation |
| 5 Subtyping | 2 | 6-8 | $2.40-$3.20 | Phenotype discovery |
| **Total** | **10** | **23-38** | **$9.20-$15.20** | **Phases 1.5-5** |

### Cost Optimization Strategies

1. **Use early stopping**: Prevent wasted GPU time on converged models
2. **Pre-validate locally**: Only run on GPU when local results look good
3. **Batch experiments**: Run multiple hyperparameter configs in one session
4. **Use spot instances**: 50% cheaper but can be interrupted
5. **Monitor actively**: Stop runs that aren't progressing
6. **Save checkpoints**: Resume interrupted runs without starting over

### When to Use A100 vs. A40

**Use A40 ($0.40/hr)** for:
- Single model training
- Small-to-medium models (< 10M parameters)
- Batch size < 128
- Most of your work

**Use A100 ($1.00/hr)** for:
- Very large models (> 50M parameters)
- Large batch sizes (> 256)
- Multi-GPU training
- Time-critical experiments

**Recommendation**: Stick with A40 for Phases 1.5-5. Only consider A100 for Phase 6+ (advanced models).

---

## RunPods Session Checklist

### Before Starting Session

- [ ] Code tested locally and working
- [ ] Training script has early stopping
- [ ] Hyperparameters documented
- [ ] Expected runtime estimated
- [ ] Budget allocated
- [ ] Results directory created
- [ ] Git repo up to date

### During Session

- [ ] Start in tmux (survives SSH disconnects)
- [ ] Monitor GPU usage (`watch -n 1 nvidia-smi`)
- [ ] Check training progress regularly
- [ ] Save checkpoints every N epochs
- [ ] Log metrics to file
- [ ] Take notes on observations

### After Session

- [ ] Download trained models
- [ ] Download training logs
- [ ] Download visualizations
- [ ] Terminate pod immediately
- [ ] Document results in `dev/workflow/experiment-log.md`
- [ ] Update roadmap with findings
- [ ] Commit and push to GitHub

---

## Local Development Best Practices

### Dataset Strategy

**Small dataset** (`all_cohorts/` - 106 patients):
- Fast iteration (< 5 min training)
- Good for debugging
- Limited performance

**Medium dataset** (subsample 500 patients):
- Reasonable training time (10-30 min)
- Better performance estimates
- Good for hyperparameter exploration

**Large dataset** (`large_cohort_1000/` - 1000 patients):
- **RunPods only** - too slow locally
- Production training
- Final evaluation

### Model Size Guidelines

**Local (MPS)**:
- Embedding dim: 64-128
- Hidden dim: 128-256
- Layers: 1-2
- Batch size: 16-32
- Parameters: < 5M

**RunPods (A40)**:
- Embedding dim: 128-256
- Hidden dim: 256-512
- Layers: 2-4
- Batch size: 64-128
- Parameters: 5-50M

### Training Time Targets

**Local**:
- Per epoch: < 30 seconds
- Full training: < 30 minutes
- If slower: reduce model size or dataset

**RunPods**:
- Per epoch: 1-5 minutes
- Full training: 1-4 hours
- If slower: check GPU utilization

---

## Script Naming Convention

### Local Scripts
```
examples/{topic}/train_{model}_local.py
```

**Characteristics**:
- Small dataset by default
- No early stopping (fast enough to run to completion)
- Minimal logging
- Focus on correctness

**Example**:
```python
# examples/survival_analysis/train_lstm_local.py
MAX_PATIENTS = 106  # Small dataset
EPOCHS = 50         # Fixed epochs
BATCH_SIZE = 16     # Small batches
```

### RunPods Scripts
```
examples/{topic}/train_{model}_runpods.py
```

**Characteristics**:
- Large dataset by default
- Early stopping enabled
- Comprehensive logging
- Hyperparameter arguments
- Checkpoint saving
- Training history export

**Example**:
```python
# examples/survival_analysis/train_lstm_runpods.py
MAX_PATIENTS = None  # Use all data
EPOCHS = 100         # Max epochs (early stopping)
BATCH_SIZE = 64      # Large batches
```

---

## Experiment Tracking

### Local Experiments

Track in notebooks or quick scripts. No need for formal tracking.

### RunPods Experiments

**Required documentation** in `dev/workflow/experiment-log.md`:

```markdown
## Experiment: Survival LSTM - Large Scale Training
**Date:** 2026-01-29
**Phase:** 1.5
**Pod:** A40 40GB
**Cost:** $0.80
**Duration:** 2 hours

### Configuration
- Dataset: large_cohort_1000 (1000 patients)
- Model: DiscreteTimeSurvivalLSTM
- Hyperparameters:
  - embedding_dim: 128
  - hidden_dim: 256
  - num_layers: 2
  - dropout: 0.3
  - batch_size: 64
  - learning_rate: 0.001

### Results
- Best val loss: 2.34
- Best C-index: 0.68
- Training time: 1.5 hours
- Early stopping: epoch 35/100

### Observations
- Model converged smoothly
- No overfitting with early stopping
- C-index aligned with synthetic correlation (r=-0.5)

### Next Steps
- Try larger hidden_dim (512)
- Experiment with learning rate scheduling
```

---

## Phase 1.5 Status & Next Steps

### Completed (Local) ✅
- Model implementation
- Loss function
- Synthetic data generator
- Training pipeline
- Evaluation metrics
- Comprehensive notebook
- Documentation

### Ready for RunPods 🎯
- [x] Training script optimized for GPU
- [x] Early stopping implemented
- [x] Hyperparameters documented
- [x] Expected results defined
- [ ] **Action**: Schedule RunPods session for final benchmarking

### After RunPods Session
- [ ] Document results in experiment log
- [ ] Update roadmap with findings
- [ ] Create production model artifact
- [ ] Move to Phase 2 (Med2Vec)

---

## Success Criteria by Phase

### Phase 1.5: Survival Analysis
- **Local**: Model trains without errors, C-index > 0.50
- **RunPods**: C-index 0.65-0.75, cost < $1.00

### Phase 2: Code Embeddings
- **Local**: Embeddings show some structure
- **RunPods**: Nearest neighbor accuracy > 80%, cost < $2.00

### Phase 3: Sequence Encoders
- **Local**: All architectures implemented and tested
- **RunPods**: Patient embeddings cluster by disease, cost < $5.00

### Phase 4: Disease Progression
- **Local**: Multi-horizon prediction working
- **RunPods**: AUC > 0.85, C-index > 0.75, cost < $5.00

### Phase 5: Disease Subtyping
- **Local**: Clustering algorithms validated
- **RunPods**: 5+ meaningful phenotypes, cost < $4.00

---

## Recommended Workflow

### Week 1-2: Local Development
1. Implement model architecture
2. Create training script (local version)
3. Test on small dataset (106 patients)
4. Debug and iterate
5. Create notebook with examples
6. Write unit tests
7. Document approach

### Week 3: RunPods Preparation
1. Create RunPods training script
2. Add early stopping and checkpointing
3. Test locally with small dataset
4. Document hyperparameters
5. Estimate runtime and cost
6. Prepare experiment log template

### Week 4: RunPods Training
1. Acquire pod (A40)
2. Set up environment
3. Run training session(s)
4. Monitor progress
5. Download results
6. Terminate pod
7. Document findings

### Week 5: Analysis & Iteration
1. Analyze RunPods results
2. Compare with local results
3. Identify improvements
4. Update models if needed
5. Prepare for next phase

---

## Tools & Scripts to Create

### Local Development
- [ ] `scripts/create_small_dataset.py` - Subsample Synthea data
- [ ] `scripts/estimate_training_time.py` - Predict runtime
- [ ] `scripts/profile_model.py` - Memory and speed profiling

### RunPods Management
- [ ] `scripts/setup_runpod.sh` - Automated pod setup
- [ ] `scripts/sync_to_pod.sh` - Sync code to pod
- [ ] `scripts/sync_from_pod.sh` - Download results
- [ ] `scripts/monitor_training.py` - Remote monitoring

### Experiment Tracking
- [ ] `dev/workflow/experiment-log.md` - Experiment journal
- [ ] `scripts/log_experiment.py` - Automated logging
- [ ] `scripts/compare_experiments.py` - Result comparison

---

## Summary

**Key Principles**:
1. **Develop locally** - 90% of work, 10% of cost
2. **Train on cloud** - 10% of work, 90% of compute
3. **Validate before scaling** - Don't waste GPU time on broken code
4. **Document everything** - Experiments, costs, learnings
5. **Optimize for cost** - Early stopping, monitoring, spot instances

**Expected Total Cost** (Phases 1.5-5): **$10-$15**

**Timeline**:
- Local development: 8-10 weeks
- RunPods sessions: 10 sessions, 25-40 hours total
- Total project: 10-12 weeks

**This approach is highly feasible and cost-effective for completing the roadmap.**

---

**Document Version:** 1.0  
**Next Review:** After Phase 1.5 RunPods session  
**Maintained By:** EHR Sequencing Team
