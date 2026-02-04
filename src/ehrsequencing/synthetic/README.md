# Synthetic Data Generation for EHR Sequence Modeling

This package provides synthetic data generators for various EHR sequence modeling tasks, including medical LLM training, transfer learning, and survival analysis.

## Purpose

The `ehrsequencing.synthetic` package is designed for:
- **Medical LLM training data** with realistic disease patterns
- **Transfer learning datasets** with domain shift
- **Survival analysis** synthetic outcomes
- **Benchmarking and testing** without requiring real patient data

## Modules

### 1. Realistic Synthetic Data (`realistic_synthetic.py`)

Generates EHR sequences with learnable medical patterns for training BEHRT and other sequence models.

**Features:**
- Disease clusters (diabetes → insulin, metformin, glucose monitoring)
- Temporal progression (diagnosis → treatment → follow-up)
- Co-morbidities (realistic disease co-occurrence)
- Age-related patterns

**Usage:**
```python
from ehrsequencing.synthetic import generate_realistic_dataset, print_dataset_statistics

codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_realistic_dataset(
    num_patients=5000,
    vocab_size=1000,
    max_seq_length=512,
    seed=42
)

print_dataset_statistics(codes, ages, visit_ids)
```

**Disease Patterns:**
- Type 2 Diabetes (10% prevalence)
- Hypertension (15% prevalence)
- Asthma (8% prevalence)
- Depression (12% prevalence)
- COPD (6% prevalence)
- Heart Failure (5% prevalence)
- Chronic Kidney Disease (7% prevalence)
- Rheumatoid Arthritis (9% prevalence)

### 2. Domain-Shifted Datasets (`domain_shift.py`)

Pre-configured domain shift scenarios for transfer learning evaluation.

**Features:**
- Clean API - no manual pattern modification needed
- Pre-configured scenarios (general→elderly, hospital A→B, historical→recent)
- Automatic pattern restoration (no side effects)

**Usage:**
```python
from ehrsequencing.synthetic import generate_domain_shifted_datasets

source_data, target_data = generate_domain_shifted_datasets(
    source_patients=10000,
    target_patients=5000,
    scenario='general_to_elderly'  # Pre-configured scenario
)
```

**Available Scenarios:**
1. **`general_to_elderly`** (default)
   - Source: Younger population (20-60 yrs), 40% lower disease rates
   - Target: Older population (50-90 yrs), 80% higher disease rates

2. **`hospital_a_to_b`**
   - Source: Urban hospital, diverse population
   - Target: Rural hospital, 30% higher disease rates

3. **`historical_to_recent`**
   - Source: 2010-2015 data, older treatment patterns
   - Target: 2016-2020 data, modern patterns

### 3. Survival Analysis (`survival.py`)

Synthetic outcome generators for survival analysis tasks.

**Features:**
- Discrete-time survival (visit-based hazards)
- Continuous-time survival (Cox proportional hazards)
- Competing risks

**Usage:**
```python
from ehrsequencing.synthetic import DiscreteTimeSurvivalGenerator

generator = DiscreteTimeSurvivalGenerator(
    baseline_hazard=0.01,
    covariate_effects={'age': 0.05, 'disease_severity': 0.1}
)

outcomes = generator.generate_outcomes(patient_sequences)
```

### 4. Demo Data (`demo_synthetic.py`)

Quick synthetic datasets for testing and demos.

**Usage:**
```python
from ehrsequencing.synthetic import generate_demo_dataset

codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_demo_dataset(
    num_patients=100,
    vocab_size=500,
    max_seq_length=50
)
```

### 5. Random Data (`random_synthetic.py`)

Random synthetic data for baseline comparison.

**Usage:**
```python
from ehrsequencing.synthetic import generate_random_dataset

codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_random_dataset(
    num_patients=1000,
    vocab_size=1000,
    max_seq_length=512
)
```

## Comparison: Realistic vs Random Data

| Aspect | Random Data | Realistic Data |
|--------|-------------|----------------|
| **Patterns** | None | Disease clusters, temporal sequences |
| **Generalization** | ❌ Overfits | ✅ Generalizes |
| **Val Loss** | Increases | Decreases |
| **Accuracy** | ~5% train, ~0.1% val | ~40% train, ~30% val |
| **Use Case** | Baseline only | Training & showcasing |

## Complete Example

Training BEHRT with realistic synthetic data:

```python
from ehrsequencing.synthetic import generate_realistic_dataset
from ehrsequencing.models.behrt import BEHRTForMLM, BEHRTConfig
from torch.utils.data import DataLoader, TensorDataset

# Generate data
codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_realistic_dataset(
    num_patients=5000,
    vocab_size=1000,
    max_seq_length=512,
    seed=42
)

# Create dataset
dataset = TensorDataset(masked_codes, ages, visit_ids, attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
config = BEHRTConfig.large(vocab_size=1000)
model = BEHRTForMLM(config)

# Train
for batch in dataloader:
    masked_codes, ages, visit_ids, attention_mask, labels = batch
    outputs = model(masked_codes, ages, visit_ids, attention_mask, labels)
    loss = outputs['loss']
    # Backprop...
```

## Transfer Learning Example

```python
from ehrsequencing.synthetic import generate_domain_shifted_datasets

# Generate domain-shifted datasets
source_data, target_data = generate_domain_shifted_datasets(
    source_patients=10000,
    target_patients=5000,
    scenario='general_to_elderly'
)

# Train on source
model = train_on_source(source_data)

# Evaluate transfer learning
zero_shot_performance = evaluate(model, target_data)
fine_tuned_performance = finetune_and_evaluate(model, target_data)
```

## Package Organization

```
ehrsequencing/
├── data/              # Real EHR data adapters (Synthea, MIMIC, etc.)
│   ├── adapters/
│   ├── visit_grouper.py
│   └── sequence_builder.py
│
├── synthetic/         # All synthetic data generation (THIS PACKAGE)
│   ├── survival.py              # Survival analysis
│   ├── realistic_synthetic.py   # Medical LLM training
│   ├── domain_shift.py          # Transfer learning
│   ├── demo_synthetic.py        # Quick demos
│   └── random_synthetic.py      # Baseline comparison
```

## Future Enhancements

Potential subpackages for specialized synthetic data:
- `synthetic/medical_llm/` - Medical LLM-specific generators
- `synthetic/survival/` - Expanded survival analysis tools
- `synthetic/phenotyping/` - Disease phenotyping datasets
- `synthetic/fairness/` - Bias and fairness evaluation datasets

## Citation

If you use this synthetic data generator in your research, please cite:

```bibtex
@software{ehr_sequencing_synthetic,
  title = {EHR Sequencing: Synthetic Data Generation},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/ehr-sequencing}
}
```
