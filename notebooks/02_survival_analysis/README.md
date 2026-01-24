# Survival Analysis Notebooks

This directory contains educational notebooks demonstrating survival analysis methods for EHR sequence modeling.

## Overview

Survival analysis extends traditional prediction tasks by modeling **when** events occur, not just **if** they occur. This temporal dimension is crucial for clinical decision-making, risk stratification, and resource planning.

## Contents

### Notebooks

#### `01_discrete_time_survival_lstm.ipynb`

**Comprehensive introduction to discrete-time survival analysis with LSTMs.**

**Topics Covered**:

1. **Understanding the C-index**
   - Mathematical definition and intuition
   - How it handles censoring
   - Interpretation in clinical context
   - Comparison with AUC and other metrics

2. **Research Questions & Clinical Applications**
   - Disease progression modeling (CKD, cancer, heart failure)
   - Treatment response prediction
   - Adverse event forecasting
   - Competing risks analysis
   - Resource utilization planning

3. **Data Labeling Strategies**
   - Translating clinical questions into survival labels
   - Defining events, time origins, and censoring
   - Avoiding temporal leakage
   - Handling different censoring types
   - Real-world example: CKD progression (Stage 3 → 4)

4. **Complete Workflow**
   - Data loading and preprocessing
   - Synthetic outcome generation
   - Model training with discrete-time survival LSTM
   - Evaluation with C-index
   - Visualization and interpretation

**Key Concepts**:
- Discrete-time hazard functions
- Visit-level survival modeling
- Concordance index (C-index)
- Temporal leakage prevention
- Synthetic data generation for testing

**Prerequisites**: Basic understanding of LSTMs and EHR data structures (see `../01_synthea_data_exploration/`)

### Scripts

#### `validate_survival_model.py`

**Quick validation script for testing survival models with flexible configurations.**

**Features**:
- **Patient subsampling**: Test with small datasets locally (e.g., 200 patients) or full datasets on cloud GPUs
- **Example display**: Show patient sequences with their survival outcomes
- **Model complexity control**: Choose from small/medium/large model sizes
- **Memory estimation**: Estimate GPU memory requirements before training
- **Outcome quality checks**: Validate synthetic outcomes have correct risk-time correlation

**Usage Examples**:

```bash
# Quick local validation with 200 patients
python validate_survival_model.py --max-patients 200 --show-examples 5

# Full dataset on cloud GPU with large model
python validate_survival_model.py --max-patients None --model-size large

# Memory estimation only (no training)
python validate_survival_model.py --estimate-memory-only

# Check synthetic outcome quality
python validate_survival_model.py --max-patients 200 --check-outcomes

# Small model for fast iteration
python validate_survival_model.py --max-patients 100 --model-size small --epochs 5
```

**Command-Line Options**:
- `--max-patients`: Number of patients (or "None" for all)
- `--model-size`: Model complexity (small/medium/large)
- `--show-examples`: Number of example sequences to display
- `--check-outcomes`: Run diagnostic checks on synthetic outcomes
- `--estimate-memory-only`: Only estimate memory (skip training)
- `--epochs`, `--batch-size`, `--lr`: Training hyperparameters
- `--device`: Device to use (auto/cpu/mps/cuda)

**When to Use**:
- **Local testing**: Use `--max-patients 200` with `--model-size small` for quick iteration
- **Cloud training**: Use `--max-patients None` with `--model-size large` for best performance
- **Debugging**: Use `--show-examples` and `--check-outcomes` to validate data quality
- **Planning**: Use `--estimate-memory-only` to check if your system can handle the model

---

## Why Survival Analysis?

### Traditional Classification vs. Survival Analysis

**Binary Classification**:
```
Question: "Will patient develop disease X?"
Answer: Yes/No
Problem: Ignores timing, treats all events as equal
```

**Survival Analysis**:
```
Question: "When will patient develop disease X?"
Answer: Time-to-event + risk trajectory
Advantages: 
  • Captures temporal dynamics
  • Handles censoring naturally
  • Enables risk stratification over time
  • Supports causal inference
```

### Clinical Impact

1. **Early Intervention**: Identify high-risk patients before events occur
2. **Resource Planning**: Predict when patients will need specific treatments
3. **Personalized Medicine**: Tailor interventions based on individual risk trajectories
4. **Clinical Trials**: Account for dropout and variable follow-up times

---

## Survival Model Types

### Discrete-Time Models
- **When to use**: Events occur at visits (discrete time points)
- **Examples**: Disease progression at clinic visits, treatment response at follow-ups
- **Model**: LSTM predicting hazard at each visit
- **Loss**: Discrete-time survival loss (negative log-likelihood)
- **Notebook**: `01_discrete_time_survival_lstm.ipynb`

### Continuous-Time Models
- **When to use**: Events can occur at any time
- **Examples**: Time to death, time to hospital admission
- **Model**: Cox proportional hazards with neural networks
- **Loss**: Partial likelihood or ranking loss
- **Notebook**: Coming soon

### Competing Risks Models
- **When to use**: Multiple event types, occurrence of one precludes others
- **Examples**: Death from different causes, disease vs. dropout
- **Model**: Multi-output survival model
- **Loss**: Cause-specific hazards
- **Notebook**: Coming soon

### Multi-State Models
- **When to use**: Complex disease trajectories with multiple states
- **Examples**: CKD stages, cancer progression, treatment pathways
- **Model**: Transition-based survival model
- **Loss**: State-specific hazards
- **Notebook**: Coming soon

---

## Key Evaluation Metrics

### Concordance Index (C-index)
- **What**: Probability model correctly ranks pairs by risk
- **Range**: 0 to 1 (0.5 = random, 1.0 = perfect)
- **Advantages**: Handles censoring, interpretable, standard metric
- **Use**: Primary metric for survival models

### Brier Score
- **What**: Mean squared error between predicted and observed survival
- **Range**: 0 to 1 (lower is better)
- **Advantages**: Calibration-focused, time-specific
- **Use**: Assess prediction accuracy at specific time points

### Integrated Brier Score (IBS)
- **What**: Average Brier score over time
- **Advantages**: Single summary metric, accounts for entire follow-up
- **Use**: Compare models across full time range

### Time-Dependent AUC
- **What**: AUC for binary outcome at specific time point
- **Advantages**: Familiar interpretation, time-specific discrimination
- **Use**: Assess discrimination at clinically relevant time points

---

## Common Pitfalls and Solutions

### Pitfall 1: Temporal Leakage
**Problem**: Using future information to predict the past

**Example**:
```python
# ✗ WRONG: Using all visit codes to predict event at visit 5
features = all_codes_in_sequence

# ✓ CORRECT: Only use codes up to current visit
features = codes_up_to_visit_t
```

**Solution**: Respect temporal ordering, truncate sequences at prediction time

### Pitfall 2: Ignoring Censoring
**Problem**: Treating censored patients as non-events

**Example**:
```python
# ✗ WRONG: Binary classification (ignores censoring)
label = 1 if event_occurred else 0

# ✓ CORRECT: Survival label (includes censoring)
label = (event_time, event_indicator)
```

**Solution**: Use survival-specific losses that handle censoring

### Pitfall 3: Informative Censoring
**Problem**: Censoring is related to outcome risk

**Example**:
```python
# ✗ WRONG: Censoring sicker patients (informative)
if patient_very_sick:
    censored = True

# ✓ CORRECT: Administrative censoring (independent)
if end_of_study:
    censored = True
```

**Solution**: Use administrative censoring or model censoring mechanism

### Pitfall 4: Wrong Time Origin
**Problem**: Starting clock at wrong time point

**Example**:
```python
# ✗ WRONG: Starting at birth for adult-onset disease
time_origin = birth_date

# ✓ CORRECT: Starting at disease diagnosis
time_origin = diagnosis_date
```

**Solution**: Define clinically meaningful time origin

---

## Data Requirements

### Minimum Requirements
- **Longitudinal data**: Multiple observations per patient over time
- **Event definition**: Clear criteria for outcome of interest
- **Time information**: Timestamps for events and censoring
- **Censoring indicators**: Flag for observed vs. censored events

### Recommended Data Elements
- **Demographics**: Age, sex, race/ethnicity
- **Diagnoses**: ICD codes with timestamps
- **Procedures**: CPT codes with timestamps
- **Medications**: Drug codes with start/stop dates
- **Lab values**: Results with timestamps
- **Vital signs**: Measurements with timestamps

### Data Quality Considerations
- **Completeness**: Sufficient follow-up time for events to occur
- **Missingness**: Handle missing data appropriately
- **Coding accuracy**: Validate event definitions
- **Temporal resolution**: Adequate granularity for research question

---

## Getting Started

### 1. Set Up Environment
```bash
# Activate conda environment
mamba activate ehrsequencing

# Navigate to notebooks directory
cd notebooks/02_survival_analysis/

# Launch Jupyter
jupyter lab
```

### 2. Run First Notebook
Open `01_discrete_time_survival_lstm.ipynb` and run cells sequentially.

### 3. Experiment
- Modify synthetic outcome parameters
- Try different model architectures
- Visualize survival curves
- Compare with baseline models

### 4. Apply to Real Data
- Define your clinical question
- Create appropriate labels
- Train and evaluate model
- Interpret results in clinical context

---

## References

### Foundational Papers
- Harrell et al. (1982): "Evaluating the Yield of Medical Tests" - Original C-index
- Cox (1972): "Regression Models and Life-Tables" - Cox proportional hazards
- Kalbfleisch & Prentice (2002): "The Statistical Analysis of Failure Time Data" - Survival analysis textbook

### Deep Learning for Survival
- Lee et al. (2018): "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks"
- Katzman et al. (2018): "DeepSurv: Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network"
- Kvamme et al. (2019): "Time-to-Event Prediction with Neural Networks and Cox Regression"

### EHR-Specific Applications
- Rajkomar et al. (2018): "Scalable and accurate deep learning with electronic health records"
- Choi et al. (2016): "RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism"

### Our Documentation
- `../../docs/methods/causal-survival-analysis-1.md` - Temporal leakage and causal labels
- `../../docs/methods/causal-survival-analysis-2.md` - Discrete-time survival derivation

---

## Next Steps

After completing these notebooks, you'll be ready to:

1. **Apply to real clinical questions**: Use your own EHR data
2. **Explore advanced models**: Competing risks, multi-state models
3. **Add interpretability**: Attention mechanisms, feature importance
4. **Integrate pretrained embeddings**: Med2Vec, BEHRT (Phase 2)
5. **Deploy models**: Production-ready survival prediction systems

---

## Questions or Issues?

- Check `../../docs/methods/` for detailed methodology
- Review `../01_synthea_data_exploration/` for data pipeline basics
- See `../../examples/train_survival_lstm.py` for production training script
- Consult survival analysis textbooks for statistical foundations
