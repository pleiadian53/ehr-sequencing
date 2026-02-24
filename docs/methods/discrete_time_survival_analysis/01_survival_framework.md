# Discrete-Time Survival Analysis: Framework and Foundations

**Topics:** Survival math, hazard function, censoring, label preparation, synthetic data

---

## Table of Contents

1. [Why Discrete-Time Survival?](#why-discrete-time-survival)
2. [Core Quantities](#core-quantities)
3. [Censoring](#censoring)
4. [Label Preparation](#label-preparation)
5. [Synthetic Data Generation](#synthetic-data-generation)
6. [Evaluation Metrics](#evaluation-metrics)

---

## Why Discrete-Time Survival?

Survival analysis models the time until an event occurs, accounting for patients who have not experienced the event by the end of observation (censored observations). In EHR data, typical outcomes include:

- **Hospital readmission** (30-day, 90-day)
- **Mortality** (in-hospital, 30-day, 1-year)
- **Disease onset** (diabetes, heart failure, stroke)
- **Treatment failure** (time to efficacy loss or adverse event)

EHR data arrives in discrete time intervals (visits), making discrete-time survival a natural fit:

- Visits are discrete events — patients are observed at specific time points
- Events occur between visits (interval censoring is the norm)
- Computational simplicity — no need for continuous-time hazard estimation
- Flexible hazard modeling — non-parametric per-visit hazard captures complex patterns

---

## Core Quantities

### Hazard Function

At each visit $t$, the model predicts a **hazard** $h_t$:

$$h_t = P(T = t \mid T \geq t,\, X_t)$$

where:
- $T$ is the event time (visit index when event occurs)
- $X_t$ is the patient history up to visit $t$
- $h_t \in [0, 1]$ is the probability of event at visit $t$, given survival to $t$

**Interpretation:** $h_t = 0.2$ means a 20% chance the event occurs at visit $t$, given the patient survived to visit $t$.

### Survival Function

$$S(t) = P(T > t) = \prod_{i=1}^{t} (1 - h_i)$$

**Example:**
```
Visit 1: h₁ = 0.10  →  S(1) = 0.90
Visit 2: h₂ = 0.15  →  S(2) = 0.90 × 0.85 = 0.765
Visit 3: h₃ = 0.20  →  S(3) = 0.765 × 0.80 = 0.612
```

### Cumulative Incidence

$$F(t) = 1 - S(t) = 1 - \prod_{i=1}^{t} (1 - h_i)$$

---

## Censoring

A patient is **censored** at time $c$ if the event has not occurred by the end of observation. The patient's true event time $T$ is unknown, but we know $T > c$.

**Right censoring** is the standard case in EHR:
- Patient leaves the health system
- Study ends before event occurs
- Patient dies from an unrelated cause

**Key assumption:** Censoring is non-informative — the reason for censoring is independent of the event risk. This assumption should be verified in real data.

---

## Label Preparation

For each patient, survival labels consist of:

1. **Visit sequence** — medical codes at each visit
2. **Event time** — visit index when event occurred (or last observed visit if censored)
3. **Event indicator** — 1 if event occurred, 0 if censored

### Example: 30-Day Readmission

**Patient A (event occurred):**
```python
{
    'patient_id': 'A001',
    'visits': [
        {'date': '2020-01-01', 'codes': [250, 401, 500], 'age': 65},  # Visit 0
        {'date': '2020-01-05', 'codes': [500, 501], 'age': 65},       # Visit 1
        {'date': '2020-01-20', 'codes': [250, 428], 'age': 65},       # Visit 2: readmission
    ],
    'event_time': 2,        # Event at visit 2
    'event_indicator': 1,   # Event occurred
}
```

**Patient B (censored):**
```python
{
    'patient_id': 'B002',
    'visits': [
        {'date': '2020-02-01', 'codes': [493, 500], 'age': 45},  # Visit 0
        {'date': '2020-02-03', 'codes': [500, 501], 'age': 45},  # Visit 1
        {'date': '2020-03-15', 'codes': [500], 'age': 45},       # Visit 2: no event
    ],
    'event_time': 3,        # Last observed visit
    'event_indicator': 0,   # Censored
}
```

### Label Generation

```python
def get_event_time(visits):
    """Find first visit where event occurs, else return censoring time."""
    for t, visit in enumerate(visits):
        if is_readmission(visit, visits[:t]):
            return t, 1  # (event_time, event_indicator)
    return len(visits), 0  # censored

# Batch labels
event_times      = torch.tensor([2, 3, 1, 5, 4])      # visit index
event_indicators = torch.tensor([1, 0, 1, 0, 1])      # 1=event, 0=censored
```

---

## Synthetic Data Generation

For development and testing, `DiscreteTimeSurvivalGenerator` creates outcomes with realistic risk-time correlation.

**Risk factors used:**
- Comorbidity burden (number of unique diagnosis codes)
- Visit frequency (healthcare utilization)
- Code diversity (condition complexity)

```python
from ehrsequencing.synthetic.survival import DiscreteTimeSurvivalGenerator

generator = DiscreteTimeSurvivalGenerator(
    censoring_rate=0.3,   # 30% of patients censored
    time_scale=0.3,       # Controls average event timing
    seed=42
)

outcome = generator.generate(patient_sequences)
# outcome.event_times:      (N,) visit indices
# outcome.event_indicators: (N,) 1=event, 0=censored
```

**Important:** Synthetic outcomes should have negative risk-time correlation (higher-risk patients experience events earlier). Verify with:
```python
from scipy.stats import spearmanr
r, p = spearmanr(risk_scores, event_times)
# Expect r ≈ -0.4 to -0.6
```

---

## Evaluation Metrics

### C-index (Concordance Index)

Measures discrimination — how well the model ranks patients by risk:

$$C = \frac{\text{concordant pairs}}{\text{comparable pairs}}$$

A pair $(i, j)$ is **comparable** if patient $i$ had an event before patient $j$ (and $i$'s event is observed). It is **concordant** if the model assigns higher risk to $i$.

- $C = 1.0$: perfect discrimination
- $C = 0.5$: random (no discrimination)
- $C < 0.5$: worse than random

**Practical note:** Use a fixed-horizon C-index to avoid length bias in variable-length sequences.

### Brier Score

Measures calibration — how close predicted probabilities are to observed outcomes:

$$\text{BS}(t) = \frac{1}{N} \sum_{i=1}^{N} \left( S_i(t) - \mathbb{1}[T_i > t] \right)^2$$

- $\text{BS} = 0$: perfect calibration
- $\text{BS} = 0.25$: uninformative model (predicts 0.5 everywhere)

### Relationship Between Metrics

| Metric | Measures | Optimized by |
|--------|----------|--------------|
| C-index | Discrimination (ranking) | Ranking loss |
| Brier Score | Calibration (probability accuracy) | NLL loss |

A model can be well-calibrated but poorly discriminative, or vice versa. See `06_loss_functions.md` for how to optimize for each.

---

**Next:** `02_ehr_to_tokens.md` — how patient visit sequences become BEHRT-ready tensors.
