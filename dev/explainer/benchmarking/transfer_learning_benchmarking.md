# Transfer Learning Benchmarking for EHR Sequence Models

## Overview

This document provides a comprehensive guide to benchmarking transfer learning capabilities in EHR sequence models, specifically focusing on how learned representations (embeddings) transfer across different patient populations or healthcare settings.

## Table of Contents

1. [Introduction](#introduction)
2. [Why Transfer Learning Matters](#why-transfer-learning-matters)
3. [The 4-Way Comparison Framework](#the-4-way-comparison-framework)
4. [Domain Shift Scenarios](#domain-shift-scenarios)
5. [Implementation Guide](#implementation-guide)
6. [Interpreting Results](#interpreting-results)
7. [Common Pitfalls](#common-pitfalls)
8. [Best Practices](#best-practices)

---

## Introduction

Transfer learning is critical for real-world EHR applications because:
- **Data scarcity**: Target domains often have limited labeled data
- **Distribution shift**: Patient populations vary across hospitals, regions, and time periods
- **Cost efficiency**: Pre-training on large datasets can reduce training time on target tasks

This benchmarking approach systematically evaluates how well a model trained on one distribution (source) performs on another (target).

---

## Why Transfer Learning Matters

### Real-World Scenarios

1. **Hospital System Transfer**
   - Train on Urban Hospital A (diverse, younger population)
   - Deploy to Rural Hospital B (older, sicker population)
   - Challenge: Different disease prevalence, demographics

2. **Temporal Transfer**
   - Train on historical data (2010-2015)
   - Deploy on recent data (2016-2020)
   - Challenge: Treatment patterns evolve, population ages

3. **Population Transfer**
   - Train on general population (primary care)
   - Deploy to specialized care (elderly, ICU)
   - Challenge: Higher disease burden, different code distributions

### Key Questions

- **Zero-shot transfer**: How well does the model work without any target domain training?
- **Fine-tuning benefit**: How much does adaptation to target domain help?
- **Upper bound**: What's the best we can achieve with target-only training?
- **Transfer efficiency**: Is transfer learning better than training from scratch?

---

## The 4-Way Comparison Framework

Our benchmarking framework compares four training strategies:

### 1. Source→Source (Baseline)

**Setup:**
- Train on source domain
- Test on source domain

**Purpose:**
- Establish baseline performance
- Verify model can learn source distribution

**Expected Result:**
- High performance (ROC-AUC: 0.95-1.00)
- This is the "easy" case - same distribution

### 2. Source→Target (Zero-Shot Transfer)

**Setup:**
- Train on source domain
- Test on target domain (no target training)

**Purpose:**
- Measure how well learned representations generalize
- Quantify domain shift impact

**Expected Result:**
- Degraded performance (ROC-AUC: 0.65-0.80)
- Performance drop indicates domain shift severity

**Key Insight:**
If zero-shot performance is high (>0.90), there's likely **no real domain shift** - this is a bug, not a feature!

### 3. Source→Target (Fine-Tuned)

**Setup:**
- Train on source domain
- Fine-tune on target domain
- Test on target domain

**Purpose:**
- Evaluate transfer learning effectiveness
- Measure adaptation capability

**Expected Result:**
- Improved performance (ROC-AUC: 0.85-0.95)
- Should be better than zero-shot
- May approach target-only performance

**Key Insight:**
The gap between zero-shot and fine-tuned shows the **benefit of adaptation**.

### 4. Target (From Scratch)

**Setup:**
- Train on target domain only (no source data)
- Test on target domain

**Purpose:**
- Establish upper bound
- Compare against transfer learning

**Expected Result:**
- Best performance (ROC-AUC: 0.95-1.00)
- This is the "oracle" - what we could achieve with unlimited target data

**Key Insight:**
If fine-tuned performance matches this, transfer learning is **highly effective**.

---

## Domain Shift Scenarios

We provide three pre-configured scenarios that simulate real-world distribution shifts:

### Scenario 1: General Population → Elderly Care (Default)

**Source Domain: General Population**
- Age range: 20-60 years
- Disease prevalence: 0.6x baseline (40% lower)
- Setting: Primary care, younger, healthier

**Target Domain: Elderly Care**
- Age range: 50-90 years
- Disease prevalence: 1.8x baseline (80% higher)
- Setting: Specialized care, older, sicker

**Expected Challenge:** Medium-High
- Large age gap (30 years)
- 3x difference in disease rates
- Different code distributions

**Example Differences:**
```
Source:
- Diabetes: 6% prevalence
- Hypertension: 9% prevalence
- Avg age: 47 years

Target:
- Diabetes: 18% prevalence (3x higher)
- Hypertension: 27% prevalence (3x higher)
- Avg age: 57 years (10 years older)
```

### Scenario 2: Hospital A → Hospital B

**Source Domain: Urban Hospital**
- Age range: 20-85 years
- Disease prevalence: 1.0x baseline
- Setting: Urban, diverse population

**Target Domain: Rural Hospital**
- Age range: 30-90 years
- Disease prevalence: 1.3x baseline (30% higher)
- Setting: Rural, older, limited resources

**Expected Challenge:** Medium
- Moderate age shift
- 30% higher disease burden
- Different healthcare access patterns

### Scenario 3: Historical → Recent

**Source Domain: 2010-2015 Data**
- Age range: 20-80 years
- Disease prevalence: 0.9x baseline
- Setting: Older treatment protocols

**Target Domain: 2016-2020 Data**
- Age range: 25-85 years
- Disease prevalence: 1.2x baseline (20% higher)
- Setting: Modern treatment protocols, aging population

**Expected Challenge:** Low-Medium
- Subtle temporal shift
- Treatment pattern evolution
- Population aging

---

## Implementation Guide

### Quick Start

```python
from ehrsequencing.synthetic import generate_domain_shifted_datasets

# Generate domain-shifted datasets
source_data, target_data = generate_domain_shifted_datasets(
    source_patients=10000,
    target_patients=5000,
    scenario='general_to_elderly',  # Pre-configured scenario
    vocab_size=1000,
    max_seq_length=256,
    source_seed=42,
    target_seed=123
)

# Each dataset contains:
# - codes: Code sequences
# - ages: Age sequences
# - visit_ids: Visit IDs
# - attention_mask: Attention mask
# - labels: MLM labels
```

### Running the Benchmark

**Local Testing (Small Model):**
```bash
cd examples/pretrain_finetune

python benchmark_transfer_learning.py \
    --model-size small \
    --source-patients 1000 \
    --target-patients 500 \
    --epochs 10 \
    --finetune-epochs 5 \
    --batch-size 32 \
    --output-dir experiments/transfer_learning_test
```

**Production Testing (Large Model on GPU):**
```bash
# On A40 pod
cd /workspace/ehr-sequencing/examples/pretrain_finetune

nohup python -u benchmark_transfer_learning.py \
    --model-size large \
    --source-patients 10000 \
    --target-patients 5000 \
    --epochs 100 \
    --finetune-epochs 20 \
    --batch-size 128 \
    --output-dir /workspace/ehr-sequencing/experiments/transfer_learning \
    > /workspace/ehr-sequencing/experiments/sessions/transfer_learning_large.out 2>&1 &
```

### Custom Scenarios

You can create custom domain shift scenarios:

```python
from ehrsequencing.synthetic.domain_shift import DomainConfig, DOMAIN_SCENARIOS

# Define custom scenario
DOMAIN_SCENARIOS['custom_scenario'] = {
    'source': DomainConfig(
        name='Custom Source',
        description='Your source description',
        prevalence_multiplier=0.8,  # 20% lower disease rates
        age_shift=-10,  # 10 years younger
        age_min=25,
        age_max=70
    ),
    'target': DomainConfig(
        name='Custom Target',
        description='Your target description',
        prevalence_multiplier=1.5,  # 50% higher disease rates
        age_shift=15,  # 15 years older
        age_min=40,
        age_max=90
    )
}

# Use custom scenario
source_data, target_data = generate_domain_shifted_datasets(
    source_patients=10000,
    target_patients=5000,
    scenario='custom_scenario'
)
```

---

## Interpreting Results

### Healthy Transfer Learning Pattern

```
Source→Source:           ROC-AUC: 0.98  (baseline)
Source→Target (zero):    ROC-AUC: 0.72  (degraded - domain shift)
Source→Target (tuned):   ROC-AUC: 0.91  (improved - adaptation works)
Target (scratch):        ROC-AUC: 0.97  (upper bound)
```

**Interpretation:**
✅ **Good transfer learning!**
- Zero-shot shows clear domain shift (0.72 vs 0.98)
- Fine-tuning provides substantial improvement (+0.19)
- Fine-tuned approaches target-only performance (0.91 vs 0.97)
- Transfer learning is effective and worthwhile

### Warning Signs

#### Pattern 1: No Domain Shift

```
Source→Source:           ROC-AUC: 0.98
Source→Target (zero):    ROC-AUC: 0.97  ← Too high!
Source→Target (tuned):   ROC-AUC: 0.98
Target (scratch):        ROC-AUC: 0.98
```

**Problem:** Zero-shot performance is too high - no real domain shift
**Cause:** Source and target have identical distributions (bug)
**Fix:** Verify domain shift implementation, check disease prevalence and age distributions

#### Pattern 2: No Transfer Benefit

```
Source→Source:           ROC-AUC: 0.98
Source→Target (zero):    ROC-AUC: 0.65
Source→Target (tuned):   ROC-AUC: 0.67  ← Barely improved!
Target (scratch):        ROC-AUC: 0.96
```

**Problem:** Fine-tuning doesn't help
**Causes:**
- Source domain too different (negative transfer)
- Fine-tuning hyperparameters wrong (learning rate too high/low)
- Not enough fine-tuning epochs
- Model capacity issues

#### Pattern 3: Catastrophic Forgetting

```
Source→Source:           ROC-AUC: 0.98
Source→Target (zero):    ROC-AUC: 0.72
Source→Target (tuned):   ROC-AUC: 0.88
Target (scratch):        ROC-AUC: 0.70  ← Worse than fine-tuned!
```

**Problem:** Target-only training underperforms
**Causes:**
- Target dataset too small
- Overfitting on target
- Need more regularization

### Performance Metrics

**Primary Metric: ROC-AUC**
- Measures discrimination ability
- Range: 0.5 (random) to 1.0 (perfect)
- Robust to class imbalance

**Secondary Metrics:**
- **PR-AUC**: Precision-Recall curve area (good for imbalanced data)
- **Average Precision**: Summary of precision-recall curve
- **Training Time**: Efficiency comparison

**Expected Ranges:**

| Scenario | Zero-Shot | Fine-Tuned | Target-Only |
|----------|-----------|------------|-------------|
| Low shift | 0.85-0.92 | 0.92-0.98 | 0.95-1.00 |
| Medium shift | 0.70-0.85 | 0.85-0.95 | 0.95-1.00 |
| High shift | 0.55-0.70 | 0.75-0.90 | 0.95-1.00 |

---

## Common Pitfalls

### 1. Identical Source and Target Distributions

**Symptom:** Perfect zero-shot transfer (ROC-AUC > 0.95)

**Root Cause:**
- Domain shift configuration not applied
- Global pattern modification not working
- Patterns restored before generation

**Fix:**
```python
# WRONG: Modifying global patterns
DISEASE_PATTERNS['diabetes'].prevalence *= 0.6  # Gets restored!

# RIGHT: Pass modified patterns as parameters
source_patterns = copy.deepcopy(DISEASE_PATTERNS)
for pattern in source_patterns.values():
    pattern.prevalence *= 0.6
generate_realistic_dataset(disease_patterns=source_patterns)
```

### 2. Insufficient Domain Shift

**Symptom:** Zero-shot performance only slightly degraded

**Causes:**
- Prevalence multipliers too close to 1.0 (e.g., 0.9x vs 1.1x)
- Age shifts too small (e.g., ±5 years)
- Random seed differences dominate over systematic shift

**Fix:**
- Use larger multipliers (0.6x vs 1.8x)
- Increase age shifts (±15-20 years)
- Verify distributions are actually different

### 3. Overfitting on Small Datasets

**Symptom:** Perfect training accuracy but poor validation

**Causes:**
- Too few patients (<1000)
- Too many epochs
- Model too large for dataset

**Fix:**
- Use at least 5000 source patients, 2000 target patients
- Add early stopping
- Use smaller model or add regularization

### 4. Wrong Fine-Tuning Strategy

**Symptom:** Fine-tuning doesn't improve or makes things worse

**Common Mistakes:**
- Learning rate too high (catastrophic forgetting)
- Learning rate too low (no adaptation)
- Fine-tuning all layers (should fine-tune embeddings + top layers)
- Too many/few epochs

**Best Practices:**
- Use lower learning rate than pre-training (1e-5 vs 1e-4)
- Fine-tune embeddings + classifier, freeze middle layers
- Use 10-20% of pre-training epochs
- Monitor validation loss carefully

### 5. Evaluation on Wrong Split

**Symptom:** Inconsistent results across runs

**Causes:**
- Testing on training data
- Data leakage between source and target
- Inconsistent train/val splits

**Fix:**
- Always use separate validation sets
- Ensure source and target are completely disjoint
- Use fixed random seeds for reproducibility

---

## Best Practices

### Dataset Size Guidelines

**Minimum (for testing):**
- Source: 1,000 patients
- Target: 500 patients
- Epochs: 10 (source), 5 (fine-tune)

**Recommended (for reliable results):**
- Source: 10,000 patients
- Target: 5,000 patients
- Epochs: 100 (source), 20 (fine-tune)

**Large-scale (for publication):**
- Source: 50,000+ patients
- Target: 10,000+ patients
- Multiple random seeds (5-10 runs)

### Hyperparameter Recommendations

**Pre-training on Source:**
```python
learning_rate = 1e-4
batch_size = 128
epochs = 100
patience = 10  # Early stopping
weight_decay = 0.01
```

**Fine-tuning on Target:**
```python
learning_rate = 1e-5  # 10x lower
batch_size = 64  # Can be smaller
epochs = 20  # 20% of pre-training
patience = 5  # More aggressive early stopping
weight_decay = 0.01
```

### Reproducibility Checklist

- ✅ Fixed random seeds (source_seed, target_seed)
- ✅ Document all hyperparameters
- ✅ Save model checkpoints
- ✅ Log all metrics (train/val loss, accuracy, ROC-AUC)
- ✅ Save dataset statistics
- ✅ Version control code and configs
- ✅ Record hardware specs (GPU type, memory)

### Reporting Guidelines

**Minimum Report:**
```
Scenario: general_to_elderly
Source patients: 10,000
Target patients: 5,000

Results:
- Source→Source:        ROC-AUC: 0.98 ± 0.01
- Zero-shot:            ROC-AUC: 0.72 ± 0.03
- Fine-tuned:           ROC-AUC: 0.91 ± 0.02
- Target-only:          ROC-AUC: 0.97 ± 0.01

Transfer benefit: +0.19 ROC-AUC
Efficiency: Fine-tuning 5x faster than target-only training
```

**Full Report Should Include:**
- Domain shift details (prevalence, age distributions)
- Training curves (loss, accuracy over epochs)
- Convergence plots
- Statistical significance tests (if multiple runs)
- Failure analysis (which cases transfer poorly)

### Validation Strategy

**During Development:**
1. Start with small model + small data (quick iteration)
2. Verify domain shift is real (check distributions)
3. Test on medium data (1000 patients)
4. Scale to full data + large model

**For Publication:**
1. Multiple random seeds (5-10 runs)
2. Cross-validation on target domain
3. Statistical significance tests
4. Ablation studies (different scenarios)
5. Error analysis (which patients fail)

---

## Advanced Topics

### Multi-Source Transfer Learning

Train on multiple source domains:

```python
# Generate multiple source domains
sources = []
for scenario in ['hospital_a', 'hospital_b', 'hospital_c']:
    source, _ = generate_domain_shifted_datasets(
        source_patients=5000,
        scenario=scenario
    )
    sources.append(source)

# Combine and train
combined_source = combine_datasets(sources)
model = train_on_combined(combined_source)
```

### Domain Adaptation Techniques

**1. Adversarial Domain Adaptation:**
- Add domain discriminator
- Train model to be domain-invariant

**2. Self-Training:**
- Use model predictions on target as pseudo-labels
- Iteratively retrain

**3. Importance Weighting:**
- Weight source samples by similarity to target
- Focus on relevant source data

### Measuring Domain Shift

**Distribution Distance Metrics:**
```python
from scipy.stats import wasserstein_distance

# Compare code distributions
source_dist = get_code_distribution(source_data)
target_dist = get_code_distribution(target_data)
shift_magnitude = wasserstein_distance(source_dist, target_dist)

print(f"Domain shift magnitude: {shift_magnitude:.3f}")
# Low shift: < 0.1
# Medium shift: 0.1 - 0.3
# High shift: > 0.3
```

---

## Troubleshooting Guide

### Issue: Perfect Performance Everywhere

**Symptoms:**
- All runs achieve ROC-AUC > 0.99
- Zero-shot = Fine-tuned = Target-only

**Diagnosis:**
1. Check if domain shift is real
2. Verify task difficulty (not too easy)
3. Check for data leakage

**Solutions:**
- Increase domain shift magnitude
- Use more challenging task (e.g., rare disease prediction)
- Verify source and target are disjoint

### Issue: No Learning at All

**Symptoms:**
- ROC-AUC stays at 0.5 (random)
- Loss doesn't decrease

**Diagnosis:**
1. Check data quality
2. Verify labels are correct
3. Check model architecture

**Solutions:**
- Print sample data to verify
- Check label distribution (not all same class)
- Try simpler model first

### Issue: Unstable Training

**Symptoms:**
- Loss oscillates wildly
- Validation performance varies dramatically

**Diagnosis:**
1. Learning rate too high
2. Batch size too small
3. Gradient explosion

**Solutions:**
- Reduce learning rate (try 1e-5)
- Increase batch size (try 128)
- Add gradient clipping
- Check for NaN values

---

## Conclusion

Transfer learning benchmarking is essential for evaluating EHR sequence models in realistic scenarios. The 4-way comparison framework provides a systematic way to measure:

1. **Baseline performance** (Source→Source)
2. **Generalization** (Zero-shot transfer)
3. **Adaptation capability** (Fine-tuning benefit)
4. **Upper bound** (Target-only performance)

**Key Takeaways:**

✅ **Always verify domain shift is real** - Zero-shot should show degradation
✅ **Fine-tuning should help** - Substantial improvement over zero-shot
✅ **Transfer should be efficient** - Faster than training from scratch
✅ **Report all four metrics** - Complete picture of transfer learning

**Next Steps:**

1. Run benchmark on your model
2. Verify results match expected patterns
3. Analyze failure cases
4. Iterate on model architecture or training strategy
5. Scale to production with confidence

For questions or issues, see:
- `examples/pretrain_finetune/benchmark_transfer_learning.py` - Implementation
- `src/ehrsequencing/synthetic/domain_shift.py` - Domain shift scenarios
- `docs/testing/pretrain_finetune/TESTING_ROADMAP.md` - Testing guide
