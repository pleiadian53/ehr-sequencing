# Tutorial 2: Synthetic Data Design and Labeling

**Part of:** Discrete-Time Survival Analysis for EHR Sequences  
**Audience:** Researchers building survival models for EHR data

---

## Table of Contents
1. [Why Synthetic Data?](#why-synthetic-data)
2. [Design Principles](#design-principles)
3. [Risk Factor Computation](#risk-factor-computation)
4. [Event Time Simulation](#event-time-simulation)
5. [Censoring Mechanism](#censoring-mechanism)
6. [Validation and Quality Control](#validation-and-quality-control)

---

## Why Synthetic Data?

### The Challenge

When developing survival models, we need labeled data with:
- **Event times:** When did the event occur?
- **Event indicators:** Did the event occur or was it censored?
- **Risk factors:** What patient characteristics predict the event?

**Problem:** Real EHR data requires:
- Manual chart review to identify events
- Clinical expertise to define outcomes
- IRB approval and privacy compliance
- Months of data preparation

### The Solution: Synthetic Outcomes

Generate synthetic survival outcomes from existing patient sequences:
- Use patient visit patterns as features
- Simulate realistic risk-time relationships
- Control correlation strength for testing
- Enable rapid iteration and development

**Key Insight:** We're not creating fake patients, we're creating fake **outcomes** for real patient sequences.

---

## Design Principles

### 1. Realistic Risk-Time Correlation

**Goal:** High-risk patients should have earlier events

**Why:** This matches clinical reality
- Sicker patients progress faster
- More comorbidities → higher risk → earlier events
- Frequent healthcare utilization → higher risk

**Implementation:**
```python
# Negative correlation between risk and event time
# High risk (0.9) → Early event (visit 5)
# Low risk (0.1) → Late event (visit 50)
correlation_target = -0.5  # Moderate to strong
```

### 2. Controlled Noise

**Goal:** Realistic but not perfect correlation

**Why:** Real clinical data has noise
- Individual variation in disease progression
- Unmeasured confounders
- Stochastic biological processes

**Implementation:**
```python
# Add small Gaussian noise to event times
noise_std = 0.08  # Small enough to preserve correlation
                  # Large enough to be realistic
```

### 3. Meaningful Risk Factors

**Goal:** Risk factors should reflect clinical knowledge

**Why:** Model should learn interpretable patterns
- Comorbidity burden (more conditions → higher risk)
- Healthcare utilization (frequent visits → higher risk)
- Code diversity (repeated conditions → higher risk)

**Implementation:**
```python
risk_factors = {
    'comorbidity': avg_codes_per_visit,  # Disease burden
    'frequency': visits_per_year,         # Utilization
    'diversity': unique_codes / total_codes  # Complexity
}
```

### 4. Appropriate Censoring

**Goal:** Simulate realistic censoring patterns

**Why:** Survival models must handle censoring
- Tests model's ability to use partial information
- Reflects real-world data collection

**Implementation:**
```python
censoring_rate = 0.3  # 30% of patients censored
# Censored patients: event_indicator = 0
# Observed events: event_indicator = 1
```

---

## Risk Factor Computation

### Overview

We extract three risk factors from each patient's visit sequence:

1. **Comorbidity burden:** How many conditions does the patient have?
2. **Visit frequency:** How often does the patient seek care?
3. **Code diversity:** How varied are the patient's conditions?

### 1. Comorbidity Burden

**Definition:** Average number of medical codes per visit

**Rationale:**
- More codes → more conditions → higher risk
- Reflects disease complexity and severity

**Computation:**
```python
def compute_comorbidity(patient_sequence):
    codes_per_visit = [visit.num_codes() for visit in patient_sequence.visits]
    avg_codes = np.mean(codes_per_visit)
    return avg_codes

# Example:
# Visit 1: 5 codes
# Visit 2: 8 codes
# Visit 3: 6 codes
# Comorbidity = (5 + 8 + 6) / 3 = 6.33
```

**Normalization:**
```python
# Normalize by typical maximum (20 codes per visit)
norm_comorbidity = avg_codes / 20.0
# Result in [0, 1] range
```

### 2. Visit Frequency

**Definition:** Number of visits per year

**Rationale:**
- Frequent visits → higher healthcare utilization → sicker patients
- Reflects disease severity and monitoring needs

**Computation:**
```python
def compute_frequency(patient_sequence):
    num_visits = len(patient_sequence.visits)
    
    # Calculate time span in years
    first_visit = patient_sequence.visits[0].timestamp
    last_visit = patient_sequence.visits[-1].timestamp
    time_span_years = (last_visit - first_visit).days / 365.0
    
    # Visits per year
    frequency = num_visits / max(time_span_years, 0.1)
    return frequency

# Example:
# 20 visits over 4 years
# Frequency = 20 / 4 = 5 visits/year
```

**Normalization:**
```python
# Normalize by typical maximum (5 visits per year)
norm_frequency = frequency / 5.0
# Result in [0, 1] range
```

### 3. Code Diversity

**Definition:** Ratio of unique codes to total codes

**Rationale:**
- Low diversity (repeated codes) → chronic condition → higher risk
- High diversity (varied codes) → exploratory care → lower risk

**Computation:**
```python
def compute_diversity(patient_sequence):
    all_codes = []
    for visit in patient_sequence.visits:
        all_codes.extend(visit.get_all_codes())
    
    unique_codes = len(set(all_codes))
    total_codes = len(all_codes)
    
    diversity = unique_codes / max(total_codes, 1)
    return diversity

# Example:
# Total codes: 100
# Unique codes: 30
# Diversity = 30 / 100 = 0.30 (low diversity, repeated conditions)
```

**Interpretation:**
```python
# Low diversity (0.2-0.4): Chronic disease with repeated codes
# High diversity (0.7-0.9): Varied conditions, exploratory care
# For risk: LOW diversity = HIGH risk (inverted)
risk_from_diversity = 1 - diversity
```

### Combining Risk Factors

**Weighted combination:**
```python
risk_weights = {
    'comorbidity': 0.4,  # 40% weight
    'frequency': 0.4,    # 40% weight
    'diversity': 0.2     # 20% weight
}

risk_score = (
    risk_weights['comorbidity'] * norm_comorbidity +
    risk_weights['frequency'] * norm_frequency +
    risk_weights['diversity'] * (1 - diversity)  # Inverted!
)

# Clip to [0.1, 0.9] to avoid extreme values
risk_score = np.clip(risk_score, 0.1, 0.9)
```

**Example:**
```python
# Patient A:
# - Avg codes: 15 → norm = 15/20 = 0.75
# - Frequency: 4/year → norm = 4/5 = 0.80
# - Diversity: 0.30 → inverted = 0.70
# Risk = 0.4*0.75 + 0.4*0.80 + 0.2*0.70 = 0.76 (HIGH RISK)

# Patient B:
# - Avg codes: 5 → norm = 5/20 = 0.25
# - Frequency: 1/year → norm = 1/5 = 0.20
# - Diversity: 0.80 → inverted = 0.20
# Risk = 0.4*0.25 + 0.4*0.20 + 0.2*0.20 = 0.22 (LOW RISK)
```

---

## Event Time Simulation

### Goal

Generate event times that have **strong negative correlation** with risk scores:
- High risk → Early event
- Low risk → Late event

### Approach: Deterministic Base + Controlled Noise

**Step 1: Normalize risk score**
```python
# Risk scores are in [0.1, 0.9]
# Normalize to [0, 1] for better spread
normalized_risk = (risk_score - 0.1) / 0.8
```

**Step 2: Compute base event time**
```python
# Invert risk to get time fraction
# High risk (1.0) → early time (fraction ~0.1)
# Low risk (0.0) → late time (fraction ~0.9)
base_time_fraction = 1.0 - normalized_risk
```

**Step 3: Add controlled noise**
```python
# Small Gaussian noise
noise_std = 0.08  # Tuned for r ≈ -0.5
noise = np.random.normal(0, noise_std)
noisy_fraction = np.clip(base_time_fraction + noise, 0.02, 0.98)
```

**Step 4: Scale to visit range**
```python
# Use fixed horizon (e.g., first 30% of visits)
horizon = 10  # First 10 visits
max_event_visit = max(5, num_visits - 1)
event_time = int(noisy_fraction * max_event_visit * time_scale * 2)

# Clip to valid range
event_visit = int(np.clip(event_time, 0, num_visits - 1))
```

### Why This Works

**Deterministic base:** Ensures strong correlation
- Risk 0.9 → fraction 0.1 → early event
- Risk 0.1 → fraction 0.9 → late event

**Controlled noise:** Adds realism
- Not all high-risk patients have immediate events
- Individual variation in progression
- Noise std = 0.08 gives correlation r ≈ -0.5

**Fixed horizon:** Prevents extreme values
- Events occur in reasonable time window
- Avoids events at visit 0 or last visit

### Example

```python
# Patient with high risk
risk_score = 0.85
normalized_risk = (0.85 - 0.1) / 0.8 = 0.9375
base_fraction = 1.0 - 0.9375 = 0.0625
noise = 0.02  # Small positive noise
noisy_fraction = 0.0625 + 0.02 = 0.0825
event_time = int(0.0825 * 50 * 0.3 * 2) = 2
# Result: Event at visit 2 (early!)

# Patient with low risk
risk_score = 0.25
normalized_risk = (0.25 - 0.1) / 0.8 = 0.1875
base_fraction = 1.0 - 0.1875 = 0.8125
noise = -0.03  # Small negative noise
noisy_fraction = 0.8125 - 0.03 = 0.7825
event_time = int(0.7825 * 50 * 0.3 * 2) = 23
# Result: Event at visit 23 (late!)
```

---

## Censoring Mechanism

### Random Censoring

**Approach:** Randomly censor a fraction of patients

```python
censoring_rate = 0.3  # 30% censored

for patient in patients:
    is_censored = np.random.random() < censoring_rate
    
    if is_censored:
        # Censor at random time before potential event
        censor_time = np.random.randint(0, num_visits)
        event_times.append(censor_time)
        event_indicators.append(0)  # Censored
    else:
        # Event occurs
        event_times.append(event_time)
        event_indicators.append(1)  # Observed
```

### Interpretation

**Censored patient (indicator = 0):**
- We observe them until `censor_time`
- We don't know if/when event occurs after that
- Model uses partial information (visits before censoring)

**Observed event (indicator = 1):**
- We observe the actual event at `event_time`
- Model learns from complete trajectory

### Why 30% Censoring?

**Realistic:** Typical in clinical studies
- 20-40% censoring is common
- Reflects patient dropout, study end, loss to follow-up

**Challenging:** Tests model's ability
- Must learn from incomplete data
- Can't just ignore censored patients

**Balanced:** Not too easy, not too hard
- < 20%: Too easy (mostly complete data)
- > 50%: Too hard (mostly incomplete data)

---

## Validation and Quality Control

### 1. Correlation Check

**Metric:** Pearson correlation between risk scores and event times (events only)

```python
event_mask = event_indicators == 1
correlation, p_value = stats.pearsonr(
    risk_scores[event_mask],
    event_times[event_mask]
)

# Target: r < -0.5 (strong negative correlation)
# Acceptable: -0.7 < r < -0.3
# Problematic: r > -0.3 (too weak)
```

**Interpretation:**
- r = -0.50: Good, model can learn
- r = -0.30: Weak, model may struggle
- r = -0.70: Strong, model should excel

### 2. Event Rate Check

**Metric:** Proportion of observed events

```python
event_rate = event_indicators.mean()

# Expected: 1 - censoring_rate
# With censoring_rate = 0.3, expect event_rate ≈ 0.70
```

**Validation:**
```python
expected_rate = 1 - censoring_rate
if abs(event_rate - expected_rate) < 0.05:
    print("✓ Event rate matches expected")
else:
    print("⚠ Event rate mismatch")
```

### 3. Risk Stratification Check

**Metric:** Mean risk score for events vs. censored

```python
event_risk = risk_scores[event_indicators == 1].mean()
censored_risk = risk_scores[event_indicators == 0].mean()

# Events should have higher risk
if event_risk > censored_risk:
    print("✓ Events have higher risk (correct)")
else:
    print("⚠ Censored have higher risk (unexpected)")
```

### 4. Visual Inspection

**Scatter plot:** Risk vs. Event Time
```python
plt.scatter(risk_scores, event_times, c=event_indicators)
plt.xlabel('Risk Score')
plt.ylabel('Event Time')
# Should see negative trend (high risk → low time)
```

**Example patients:**
```
Risk Score   Event Time   Expected      Actual
0.85         2            Early         Early  ✓
0.75         5            Early         Early  ✓
0.50         12           Mid           Mid    ✓
0.30         25           Late          Late   ✓
0.20         40           Late          Late   ✓
```

### 5. Distribution Checks

**Event time distribution:**
```python
plt.hist(event_times[event_indicators == 1], bins=20)
# Should be spread across time range
# Not all at beginning or end
```

**Risk score distribution:**
```python
plt.hist(risk_scores, bins=20)
# Should cover [0.1, 0.9] range
# Not clustered in narrow band
```

---

## Complete Example

### Code

```python
from ehrsequencing.synthetic.survival import DiscreteTimeSurvivalGenerator

# Initialize generator
generator = DiscreteTimeSurvivalGenerator(
    censoring_rate=0.3,
    risk_weights={
        'comorbidity': 0.4,
        'frequency': 0.4,
        'diversity': 0.2
    },
    time_scale=0.3,
    seed=42
)

# Generate outcomes
outcome = generator.generate(sequences)

# Validate
print(f"Event rate: {outcome.event_indicators.float().mean():.2%}")
print(f"Mean risk: {outcome.risk_scores.mean():.3f}")

# Check correlation
event_mask = outcome.event_indicators == 1
correlation = np.corrcoef(
    outcome.risk_scores[event_mask],
    outcome.event_times[event_mask]
)[0, 1]
print(f"Correlation: {correlation:.3f}")
```

### Expected Output

```
Event rate: 69.5%
Mean risk: 0.628
Correlation: -0.500

✓ Strong negative correlation
✓ Event rate matches expected (70%)
✓ Events have higher mean risk (0.639 vs 0.603)
```

---

## Key Takeaways

1. **Synthetic outcomes enable rapid development** without manual labeling
2. **Risk factors should be clinically meaningful** (comorbidity, frequency, diversity)
3. **Strong correlation (r < -0.5) is essential** for model learning
4. **Controlled noise adds realism** without destroying signal
5. **Validation is critical** before training models

---

## Common Pitfalls

### ❌ Too Much Noise
```python
noise_std = 0.20  # Too large!
# Result: Correlation r = -0.15 (too weak)
```

### ❌ Wrong Correlation Direction
```python
event_time = risk_score * num_visits  # WRONG!
# High risk → High time (backwards!)
```

### ❌ No Normalization
```python
risk_score = comorbidity + frequency  # WRONG!
# Different scales, unbounded values
```

### ✅ Correct Approach
```python
# Normalize inputs
norm_comorbidity = comorbidity / 20.0
norm_frequency = frequency / 5.0

# Weighted combination
risk_score = 0.4 * norm_comorbidity + 0.4 * norm_frequency + ...

# Clip to valid range
risk_score = np.clip(risk_score, 0.1, 0.9)

# Invert for time
base_time_fraction = 1.0 - (risk_score - 0.1) / 0.8

# Add small noise
noisy_fraction = base_time_fraction + np.random.normal(0, 0.08)
```

---

**Next Tutorial:** [Loss Function Formulation](TUTORIAL_03_loss_function.md)
