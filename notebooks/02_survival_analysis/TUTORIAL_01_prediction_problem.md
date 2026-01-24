# Tutorial 1: The Survival Prediction Problem

**Part of:** Discrete-Time Survival Analysis for EHR Sequences  
**Audience:** Researchers and practitioners new to survival analysis in healthcare

---

## Table of Contents
1. [What is Survival Analysis?](#what-is-survival-analysis)
2. [Why Use Survival Analysis for EHR Data?](#why-use-survival-analysis-for-ehr-data)
3. [The Prediction Problem](#the-prediction-problem)
4. [Discrete-Time vs. Continuous-Time](#discrete-time-vs-continuous-time)
5. [Evaluation Metrics](#evaluation-metrics)

---

## What is Survival Analysis?

**Survival analysis** is a branch of statistics that models **time-to-event** data. The key question is:

> **"When will an event occur?"**

Rather than just:

> "Will an event occur?" (binary classification)

### Key Concepts

**Event:** The outcome of interest
- Disease onset (e.g., diabetes, heart failure)
- Hospital readmission
- Mortality
- Treatment response

**Time-to-Event (T):** How long until the event occurs
- Measured from a defined starting point (e.g., first visit, diagnosis)
- Can be in days, visits, or other time units

**Censoring:** When we don't observe the event
- **Right censoring:** Patient leaves study before event occurs
- **Administrative censoring:** Study ends before event occurs
- **Lost to follow-up:** Patient stops coming to clinic

### Example

**Binary Classification:**
- Question: "Will this patient develop diabetes?"
- Answer: Yes (1) or No (0)
- Problem: Ignores **when** it happens

**Survival Analysis:**
- Question: "**When** will this patient develop diabetes?"
- Answer: Time = 365 days (1 year)
- Benefit: Captures temporal dynamics

---

## Why Use Survival Analysis for EHR Data?

### 1. **Temporal Information is Critical**

In healthcare, **timing matters**:
- A patient who develops disease in 1 month needs **urgent** intervention
- A patient who develops disease in 10 years can be monitored
- Binary classification treats both the same!

### 2. **Censoring is Inevitable**

In real-world EHR data:
- Patients move to different hospitals
- Studies have finite duration
- Not all patients experience the event

**Survival analysis handles censoring properly** by:
- Using partial information from censored patients
- Not treating censored patients as "no event" (which is wrong!)
- Accounting for uncertainty in event timing

### 3. **Risk Stratification**

Survival models enable:
- **High-risk identification:** Patients likely to have events soon
- **Resource allocation:** Prioritize interventions for high-risk patients
- **Personalized timelines:** Different patients have different trajectories

### 4. **Clinical Relevance**

Clinicians think in terms of:
- "5-year survival rate"
- "Time to progression"
- "Median time to event"

Survival analysis directly provides these metrics.

---

## The Prediction Problem

### Problem Formulation

**Given:**
- Patient history up to time $t$: $H_t = \{v_1, v_2, \ldots, v_t\}$
- Each visit $v_i$ contains medical codes (diagnoses, procedures, medications)

**Predict:**
- Probability of event at each future time point
- Risk score indicating overall event likelihood
- Expected time to event

### Mathematical Framework

For each patient, we observe:
- $T$: Time of event or censoring
- $\delta$: Event indicator (1 = event observed, 0 = censored)

We want to model:
- $h(t | H_t)$: **Hazard** at time $t$ given history
  - Probability of event at time $t$ given survival to $t$
- $S(t | H_t)$: **Survival function**
  - Probability of surviving past time $t$

**Relationship:**
$$S(t) = \prod_{i=1}^{t} (1 - h(i))$$

The survival probability is the product of not having an event at each prior time.

### Example: Diabetes Prediction

**Patient History:**
```
Visit 1 (Day 0):   [Hypertension, High BMI]
Visit 2 (Day 90):  [Hypertension, High cholesterol]
Visit 3 (Day 180): [Hypertension, Prediabetes]
Visit 4 (Day 270): [Diabetes] ← EVENT
```

**Prediction Task:**
- At Visit 1: Predict hazard at future visits
- At Visit 2: Update prediction with new information
- At Visit 3: High hazard expected at next visit (prediabetes signal)

**Model Output:**
```
h(Visit 1) = 0.05  (low risk)
h(Visit 2) = 0.10  (moderate risk)
h(Visit 3) = 0.30  (high risk - prediabetes!)
h(Visit 4) = 0.60  (very high risk)
```

---

## Discrete-Time vs. Continuous-Time

### Continuous-Time Survival

**Used when:** Events can occur at any moment
- Example: Death, ICU admission
- Time measured precisely (hours, minutes)

**Model:** Cox proportional hazards
$$h(t | X) = h_0(t) \exp(X^T \beta)$$

**Pros:**
- Flexible baseline hazard $h_0(t)$
- Well-established theory

**Cons:**
- Harder to integrate with deep learning
- Assumes proportional hazards (may not hold)

### Discrete-Time Survival

**Used when:** Events occur at discrete time points
- Example: Disease onset at visits, readmission episodes
- Time measured in visits, months, years

**Model:** Discrete hazard at each time point
$$h_t = P(T = t | T \geq t, H_t)$$

**Pros:**
- Natural fit for visit-based EHR data
- Easy integration with LSTMs/Transformers
- Flexible hazard (no proportional hazards assumption)

**Cons:**
- Requires discretization if time is continuous
- May lose precision if intervals are large

### Why Discrete-Time for EHR?

1. **Visit-based data:** EHR events are naturally grouped into visits
2. **Deep learning compatibility:** LSTMs process sequences of visits
3. **Flexibility:** Hazard can change arbitrarily over time
4. **Interpretability:** Hazard at each visit is easy to understand

---

## Evaluation Metrics

### 1. Concordance Index (C-index)

**Definition:** Fraction of patient pairs correctly ranked by risk

**Interpretation:**
- C-index = 1.0: Perfect ranking
- C-index = 0.5: Random ranking (coin flip)
- C-index = 0.0: Completely wrong ranking

**Calculation:**
For all pairs $(i, j)$ where:
- Patient $i$ has observed event
- Event time $T_i < T_j$

Count as **concordant** if: $\text{risk}_i > \text{risk}_j$

$$\text{C-index} = \frac{\text{concordant pairs}}{\text{total comparable pairs}}$$

**Example:**
```
Patient A: Event at visit 5, risk = 0.8
Patient B: Event at visit 10, risk = 0.6
Patient C: Event at visit 15, risk = 0.4

Pairs:
(A, B): T_A < T_B and risk_A > risk_B ✓ Concordant
(A, C): T_A < T_C and risk_A > risk_C ✓ Concordant
(B, C): T_B < T_C and risk_B > risk_C ✓ Concordant

C-index = 3/3 = 1.0 (perfect!)
```

**Why C-index?**
- Standard metric in survival analysis
- Handles censoring properly
- Interpretable (like AUC but for time-to-event)

### 2. Calibration

**Definition:** Do predicted probabilities match observed frequencies?

**Example:**
- Model predicts 30% of high-risk patients will have events in 1 year
- Observed: 28% actually have events
- Good calibration!

**Metrics:**
- Calibration plots (predicted vs. observed)
- Brier score (mean squared error of probabilities)

### 3. Time-Dependent AUC

**Definition:** AUC for predicting event within time $t$

**Example:**
- AUC at 1 year: How well does model predict events within 1 year?
- AUC at 5 years: How well does model predict events within 5 years?

**Benefit:** Evaluates prediction at specific horizons

---

## Putting It All Together

### The Complete Workflow

1. **Define the event:** What are we predicting?
   - Example: Diabetes onset

2. **Define the time origin:** When does the clock start?
   - Example: First visit to clinic

3. **Define the time scale:** How do we measure time?
   - Example: Visits, days, months

4. **Handle censoring:** Who is censored and why?
   - Example: Patients who leave the health system

5. **Build the model:** Predict hazard at each time point
   - Example: LSTM predicting visit-level hazards

6. **Evaluate:** Use C-index and calibration
   - Example: C-index = 0.70 (good discrimination)

### Example: Heart Failure Prediction

**Setup:**
- Event: First heart failure diagnosis
- Time origin: First cardiology visit
- Time scale: Visits
- Censoring: Patients who leave health system

**Model:**
- Input: Sequence of visits with diagnoses, procedures, medications
- Output: Hazard at each visit
- Risk score: Mean hazard over first 10 visits

**Evaluation:**
- C-index = 0.72 (good)
- Calibration: Predicted vs. observed matches well
- Interpretation: Model successfully identifies high-risk patients

---

## Key Takeaways

1. **Survival analysis captures "when"** not just "if"
2. **Censoring is handled properly** unlike binary classification
3. **Discrete-time models** are natural for visit-based EHR data
4. **C-index measures ranking quality** (like AUC for survival)
5. **Risk scores enable stratification** for clinical decision-making

---

## Further Reading

### Textbooks
- Singer & Willett (2003). *Applied Longitudinal Data Analysis*
- Tutz & Schmid (2016). *Modeling Discrete Time-to-Event Data*

### Papers
- Katzman et al. (2018). "DeepSurv: Personalized treatment recommender system using a Cox proportional hazards deep neural network"
- Lee et al. (2018). "DeepHit: A deep learning approach to survival analysis with competing risks"

### Software
- `lifelines` (Python): Continuous-time survival analysis
- `scikit-survival` (Python): Machine learning for survival analysis
- `PyHealth`: EHR-specific deep learning models

---

**Next Tutorial:** [Synthetic Data Design and Labeling](TUTORIAL_02_synthetic_data_design.md)
