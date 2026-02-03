# Survival Analysis with EHR Sequencing

Discrete-time survival analysis for predicting time-to-event outcomes from Electronic Health Records.

---

## Overview

EHR Sequencing provides tools for survival analysis on patient sequences, enabling prediction of:

- **Hospital Readmission** - 30-day readmission risk
- **Mortality** - In-hospital, 30-day, and 1-year mortality
- **Disease Onset** - Time to diabetes, heart failure, stroke, etc.
- **Treatment Response** - Time to treatment efficacy or failure

---

## Models

### LSTM Baseline

Recurrent neural network for survival analysis.

**Architecture:**
```
Visit Codes → Embeddings → LSTM → Hazard Prediction
```

**Features:**
- Learns embeddings from scratch
- Unidirectional (causal) processing
- Fast training, good baseline performance

**Usage:**
```python
from ehrsequencing.models.survival_lstm import DiscreteTimeSurvivalLSTM

model = DiscreteTimeSurvivalLSTM(
    vocab_size=5000,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.3
)
```

**Training:**
```bash
python examples/survival_analysis/train_lstm.py \
    --data-dir data/synthea/ \
    --outcome synthetic \
    --epochs 100 \
    --batch-size 64
```

---

### BEHRT for Survival (Coming Soon)

Transformer-based model with pre-trained representations.

**Architecture:**
```
Visit Codes → BEHRT Encoder (pre-trained) → Visit Aggregation → Hazard Prediction
```

**Features:**
- Pre-trained with Masked Language Modeling (MLM)
- Bidirectional context within visits
- EHR-specific embeddings (age, visit, segment)
- Transfer learning from large unlabeled data

**Expected Advantages:**
- 10-20% improvement in C-index over LSTM
- Faster convergence with pre-training
- Better generalization (smaller train-val gap)
- Richer representations for downstream tasks

**Usage (Planned):**
```python
from ehrsequencing.models.behrt_survival import BEHRTForSurvival
from ehrsequencing.models.behrt import BEHRTForMLM

# Load pre-trained BEHRT
pretrained = BEHRTForMLM.from_pretrained('checkpoints/behrt_mlm/')

# Create survival model
model = BEHRTForSurvival(
    behrt_encoder=pretrained.behrt,
    hidden_dim=256,
    dropout=0.1
)

# Fine-tune for survival
# Option 1: Freeze encoder, train only head
for param in model.behrt.parameters():
    param.requires_grad = False

# Option 2: LoRA fine-tuning (efficient)
from ehrsequencing.models.lora import apply_lora_to_behrt
model.behrt = apply_lora_to_behrt(model.behrt, rank=16)

# Option 3: Full fine-tuning
# All parameters trainable (slower, potentially better)
```

---

## Discrete-Time Survival Framework

### Hazard Function

At each visit $t$, the model predicts a hazard:

$$h_t = P(T = t | T \geq t)$$

The hazard represents the probability of an event occurring at time $t$, given survival to time $t$.

### Survival Function

The survival probability is computed from hazards:

$$S(t) = \prod_{i=1}^{t} (1 - h_i)$$

### Loss Function

Negative log-likelihood for discrete-time survival:

$$\mathcal{L} = -\sum_{i=1}^{N} \left[ \delta_i \log h_{t_i} + (1-\delta_i) \log S(t_i) \right]$$

where $\delta_i$ is the event indicator (1 if event occurred, 0 if censored).

---

## Evaluation Metrics

### Concordance Index (C-index)

Measures discrimination - the probability that a model correctly orders pairs of patients by risk.

- **C-index = 0.5**: Random predictions
- **C-index = 1.0**: Perfect discrimination
- **C-index > 0.7**: Good performance

**Interpretation:**
- C-index of 0.75 means the model correctly ranks 75% of patient pairs

### Calibration

Measures agreement between predicted and observed event rates.

- **Calibration plot**: Predicted risk vs observed frequency
- **Brier score**: Mean squared error of predictions
- **Well-calibrated**: Predictions match reality

### Time-Dependent AUC

AUC at specific time horizons (e.g., 7, 14, 30 days).

Useful for evaluating early vs late prediction performance.

---

## Clinical Use Cases

### 1. Hospital Readmission Prediction

**Goal:** Identify patients at high risk of 30-day readmission

**Clinical Value:**
- Reduce readmission rates (CMS penalty avoidance)
- Target interventions (follow-up calls, home health)
- Optimize discharge planning

**Metrics:**
- C-index for discrimination
- Calibration at 7, 14, 30 days
- Cost-benefit analysis (prevented readmissions)

**Example:**
```bash
python examples/survival_analysis/train_lstm.py \
    --data-dir data/synthea/ \
    --outcome readmission \
    --time-horizon 30 \
    --epochs 100
```

---

### 2. Mortality Risk Prediction

**Goal:** Estimate survival probability for ICU/ED triage

**Clinical Value:**
- ICU resource allocation
- Goals-of-care discussions
- Clinical trial enrollment criteria

**Time Horizons:**
- In-hospital mortality
- 30-day mortality
- 1-year mortality

**Example:**
```bash
python examples/survival_analysis/train_lstm.py \
    --data-dir data/synthea/ \
    --outcome mortality \
    --time-horizon 365 \
    --epochs 100
```

---

### 3. Disease Onset Prediction

**Goal:** Predict time to disease development for preventive care

**Clinical Value:**
- Identify pre-diabetic patients
- Predict cardiovascular events
- Screen for cancer risk

**Target Diseases:**
- Diabetes (high prevalence, preventable)
- Heart failure (high cost, manageable)
- Stroke (severe outcome, preventable)

**Example:**
```bash
python examples/survival_analysis/train_lstm.py \
    --data-dir data/synthea/ \
    --outcome disease_onset \
    --target-disease diabetes \
    --time-horizon 365 \
    --epochs 100
```

---

## Data Requirements

### Input Format

Patient sequences with visit-level structure:

```python
{
    'patient_id': '12345',
    'visits': [
        {
            'visit_date': '2020-01-15',
            'codes': ['LOINC:4548-4', 'SNOMED:44054006'],
            'age': 45.2
        },
        {
            'visit_date': '2020-06-15',
            'codes': ['LOINC:2339-0', 'RXNORM:860975'],
            'age': 45.6
        }
    ],
    'outcome': {
        'event_time': 3,  # Event at visit 3
        'event_indicator': 1,  # 1 = event occurred, 0 = censored
        'event_type': 'readmission'
    }
}
```

### Synthetic Data Generation

For development and testing:

```python
from ehrsequencing.synthetic.survival import DiscreteTimeSurvivalGenerator

generator = DiscreteTimeSurvivalGenerator(
    risk_correlation=-0.5,  # Negative correlation (higher risk → earlier event)
    censoring_rate=0.3,
    time_scale=10
)

# Generate outcomes for patient sequences
outcomes = generator.generate_outcomes(patient_sequences)
```

---

## Training Pipeline

### 1. Data Preparation

```python
from ehrsequencing.data import SyntheaAdapter, VisitGrouper, PatientSequenceBuilder

# Load data
adapter = SyntheaAdapter('data/synthea/')
patients = adapter.load_patients()
events = adapter.load_events()

# Group into visits
grouper = VisitGrouper(strategy='hybrid')
visits = grouper.group_by_patient(events)

# Build sequences
builder = PatientSequenceBuilder()
sequences = builder.build_sequences(visits)
```

### 2. Model Training

```python
from ehrsequencing.models.survival_lstm import DiscreteTimeSurvivalLSTM
from ehrsequencing.models.losses import DiscreteTimeSurvivalLoss

model = DiscreteTimeSurvivalLSTM(vocab_size=5000)
loss_fn = DiscreteTimeSurvivalLoss()

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        hazards = model(batch['codes'], batch['visit_mask'], batch['sequence_mask'])
        loss = loss_fn(hazards, batch['event_time'], batch['event_indicator'])
        loss.backward()
        optimizer.step()
```

### 3. Evaluation

```python
from ehrsequencing.models.losses import concordance_index

# Compute C-index
c_index = concordance_index(
    risk_scores=predicted_risks,
    event_times=true_event_times,
    event_indicators=true_event_indicators
)

print(f"C-index: {c_index:.4f}")
```

---

## Best Practices

### Model Selection

1. **Start with LSTM baseline** - Fast, interpretable, good performance
2. **Try BEHRT if available** - Pre-training may improve performance
3. **Compare multiple approaches** - Use benchmarking framework

### Hyperparameter Tuning

- **Embedding dimension**: 128-256 for LSTM, pre-trained for BEHRT
- **Hidden dimension**: 256-512
- **Dropout**: 0.2-0.4 (higher for small datasets)
- **Learning rate**: 1e-3 for LSTM, 1e-5 for BEHRT fine-tuning
- **Early stopping**: Patience 10-20 epochs

### Avoiding Overfitting

- Use dropout and weight decay
- Early stopping on validation C-index
- Cross-validation for small datasets
- Regularization (L2, LoRA for BEHRT)

### Clinical Validation

- Calibration plots (predicted vs observed)
- Subgroup analysis (age, gender, comorbidities)
- Temporal validation (train on old data, test on new)
- External validation (different hospital/dataset)

---

## References

### Papers

- **Discrete-Time Survival**: Tutz & Schmid (2016). "Modeling Discrete Time-to-Event Data"
- **C-index**: Harrell et al. (1982). "Evaluating the Yield of Medical Tests"
- **BEHRT**: Li et al. (2020). "BEHRT: Transformer for Electronic Health Records"

### Code Examples

- `examples/survival_analysis/train_lstm.py` - LSTM training script
- `examples/survival_analysis/train_lstm_demo.py` - Quick demo
- `notebooks/01_discrete_time_survival_lstm.ipynb` - Interactive tutorial

---

## Support

For questions or issues:
- Check the [tutorials](../tutorials/)
- Review [example scripts](../../examples/survival_analysis/)
- See [benchmarking guide](benchmarking.md) for model comparison

---

**Status:** LSTM baseline complete | BEHRT survival in development  
**Updated:** February 2026
