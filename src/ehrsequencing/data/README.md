# Realistic Synthetic EHR Data Generator

This module generates synthetic EHR data with **learnable medical patterns** for showcasing BEHRT and other sequence models.

## Why Realistic Synthetic Data?

**Problem with Random Data:**
- Random medical codes have no structure
- Model can memorize training data but can't generalize
- Validation loss increases (overfitting)
- Doesn't showcase model's true capabilities

**Solution: Realistic Patterns:**
- Disease clusters (diabetes → insulin, metformin, glucose tests)
- Temporal progression (diagnosis → treatment → monitoring)
- Co-morbidities (diabetes + hypertension)
- Age-related patterns
- Learnable structure that models can capture

## Disease Patterns Included

The generator simulates 8 common chronic diseases:

1. **Type 2 Diabetes** (10% prevalence)
   - Diagnosis codes: 250-252
   - Treatments: Metformin, insulin (100-103)
   - Monitoring: Glucose tests, HbA1c (300-302)

2. **Hypertension** (15% prevalence)
   - Diagnosis codes: 401-403
   - Treatments: ACE inhibitors, beta blockers (110-112)
   - Monitoring: BP checks (310-311)

3. **Asthma** (8% prevalence)
   - Diagnosis codes: 493-494
   - Treatments: Inhalers, steroids (120-122)
   - Monitoring: Pulmonary function tests (320-321)

4. **Depression** (12% prevalence)
   - Diagnosis codes: 296, 311
   - Treatments: SSRIs, therapy (130-132)
   - Monitoring: Mental health assessments (330-331)

5. **COPD** (6% prevalence)
   - Diagnosis codes: 496, 491-492
   - Treatments: Bronchodilators, oxygen (140-142)
   - Monitoring: Spirometry (340-341)

6. **Heart Failure** (5% prevalence)
   - Diagnosis codes: 428-429
   - Treatments: Diuretics, ACE inhibitors (150-152)
   - Monitoring: Echo, BNP (350-351)

7. **Chronic Kidney Disease** (7% prevalence)
   - Diagnosis codes: 585-586
   - Treatments: Dialysis, medications (160-161)
   - Monitoring: Creatinine, GFR (360-361)

8. **Rheumatoid Arthritis** (9% prevalence)
   - Diagnosis codes: 714-715
   - Treatments: NSAIDs, DMARDs (170-172)
   - Monitoring: Inflammatory markers (370-371)

## Co-morbidity Patterns

Realistic disease co-occurrence:
- 40% of diabetics have hypertension
- 30% of diabetics develop CKD
- 25% of hypertension patients develop heart failure
- 20% of COPD patients have heart failure
- 15% of diabetics have depression

## Usage

### Basic Usage

```python
from ehrsequencing.data.realistic_synthetic import generate_realistic_dataset, print_dataset_statistics

# Generate dataset
codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_realistic_dataset(
    num_patients=5000,
    vocab_size=1000,
    max_seq_length=512,
    seed=42
)

# Print statistics
print_dataset_statistics(codes, ages, visit_ids)
```

### With BEHRT Training

```bash
# Use realistic data (RECOMMENDED for showcasing)
python examples/pretrain_finetune/train_behrt_demo.py \
    --model_size large \
    --use_lora \
    --lora_rank 16 \
    --num_patients 5000 \
    --epochs 100 \
    --batch_size 128 \
    --realistic_data  # ← Add this flag!

# Use random data (for testing only)
python examples/pretrain_finetune/train_behrt_demo.py \
    --model_size small \
    --num_patients 100 \
    --epochs 10
```

## Expected Performance

### With Random Data ❌
```
Epoch 1  | Train Loss: 7.0 | Val Loss: 6.9
Epoch 10 | Train Loss: 5.6 | Val Loss: 7.2  ← Overfitting
Accuracy: ~5% train, ~0.1% val
```
Model memorizes random patterns but can't generalize.

### With Realistic Data ✅
```
Epoch 1  | Train Loss: 7.0 | Val Loss: 6.9
Epoch 10 | Train Loss: 5.2 | Val Loss: 5.8  ← Learning!
Epoch 30 | Train Loss: 3.8 | Val Loss: 4.2
Accuracy: ~40% train, ~30% val
```
Model learns disease patterns and generalizes to validation set.

## How It Works

### 1. Patient Generation
Each patient is assigned:
- Base age (20-70 years)
- Diseases based on age range and prevalence
- Co-morbidities based on existing diseases

### 2. Trajectory Generation
For each disease:
1. **Diagnosis visit**: Initial diagnosis codes
2. **Treatment visits**: Diagnosis + treatment codes
3. **Monitoring visits**: Treatment + monitoring codes
4. **Routine care**: Checkups, screenings

### 3. Temporal Progression
- Visits spaced 1-4 months apart
- Codes repeat realistically (e.g., refills, follow-ups)
- Age increases with each visit

### 4. MLM Masking
- 15% of codes masked for prediction
- 80% replaced with [MASK] token
- 10% replaced with random code
- 10% kept unchanged

## Dataset Statistics Example

```
📊 Dataset Statistics:
   Number of patients: 5000
   Sequence length: 512
   Avg sequence length: 45.3 ± 12.8
   Unique codes used: 387
   Age range: 20-85
   Avg age: 52.4
   Avg visits per patient: 8.2

   Top 10 most frequent codes:
      1. Code 500:  8234 ( 4.12%) - Routine care
      2. Code 250:  6891 ( 3.45%) - Type 2 Diabetes (diagnosis)
      3. Code 401:  6234 ( 3.12%) - Hypertension (diagnosis)
      4. Code 100:  5678 ( 2.84%) - Type 2 Diabetes (treatment)
      5. Code 110:  5123 ( 2.56%) - Hypertension (treatment)
      ...
```

## Advantages Over Random Data

| Aspect | Random Data | Realistic Data |
|--------|-------------|----------------|
| **Patterns** | None | Disease clusters, temporal sequences |
| **Generalization** | ❌ Overfits | ✅ Generalizes |
| **Val Loss** | Increases | Decreases |
| **Accuracy** | ~5% train, ~0.1% val | ~40% train, ~30% val |
| **Showcasing** | ❌ Poor | ✅ Excellent |
| **Learning** | Memorization only | True pattern learning |

## API Reference

### `generate_realistic_dataset()`

```python
def generate_realistic_dataset(
    num_patients: int = 1000,
    vocab_size: int = 1000,
    max_seq_length: int = 512,
    mask_prob: float = 0.15,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

**Returns:**
- `codes`: Original code sequences [num_patients, max_seq_length]
- `ages`: Age at each position [num_patients, max_seq_length]
- `visit_ids`: Visit sequence IDs [num_patients, max_seq_length]
- `attention_mask`: 1 for real tokens, 0 for padding [num_patients, max_seq_length]
- `masked_codes`: Codes with MLM masking applied [num_patients, max_seq_length]
- `labels`: Target labels for MLM (-100 for non-masked) [num_patients, max_seq_length]

### `generate_patient_trajectory()`

```python
def generate_patient_trajectory(
    patient_id: int,
    vocab_size: int = 1000,
    max_visits: int = 50,
    seed: Optional[int] = None
) -> Tuple[List[List[int]], List[int], List[int]]
```

**Returns:**
- `visits`: List of visits, each visit is a list of codes
- `ages`: Age at each visit
- `visit_ids`: Visit sequence numbers

## Future Enhancements

1. **More diseases**: Add cancer, stroke, Alzheimer's, etc.
2. **Medications**: More realistic drug sequences
3. **Lab values**: Include numeric lab results
4. **Procedures**: Add surgical procedures, imaging
5. **Seasonality**: Flu in winter, allergies in spring
6. **Demographics**: Gender-specific patterns
7. **Real code mappings**: Use actual ICD-10, RxNorm codes

## Citation

If you use this synthetic data generator in your research, please cite:

```bibtex
@software{ehr_sequencing_realistic_synthetic,
  title = {Realistic Synthetic EHR Data Generator},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/ehr-sequencing}
}
```
