# Causal Survival Analysis for EHR Sequences (Part 1)

## Why Disease Progression Modeling is Different

When we move from simple classification tasks (like "does this patient have diabetes?") to **disease progression modeling**, we enter the realm of causal survival analysis. This is where many EHR sequence modeling papers go wrong—not because the neural networks are poorly designed, but because the **prediction task itself violates causality**.

This tutorial has two parts:

**Part 1 (this document):**

1. Understanding the most dangerous data leakage pattern in temporal prediction
2. Designing progression labels that are both causal and statistically efficient

**Part 2 ([causal-survival-analysis-2.md](causal-survival-analysis-2.md)):**

1. Deep dive into discrete-time survival modeling
2. Deriving the likelihood formula step-by-step
3. Implementation details for PyTorch

---

## Why This Matters: The Diabetes Example

Consider our diabetes prediction task from the LSTM notebook:

```python
# Task: Does patient have diabetes?
label = 1 if any(diabetes_code in patient_history) else 0
```

This works, but it's **not temporal**. The answer depends only on the presence of certain codes—a rule-based system could do this. We're not really using the sequential structure.

**Better question:** Can we predict when a patient will progress from pre-diabetes to diabetes? Or from CKD stage 3 to stage 4? These are **progression** questions that require understanding disease dynamics over time.

This is where sequence modeling becomes essential—and where most papers introduce subtle but fatal leakage.

---

## Part I: The Most Dangerous Leakage Pattern

### The Setup: A Seemingly Reasonable Approach

Imagine you want to predict disease progression. You define a patient-level outcome:

> **Clinical Question:** "Did this patient progress to CKD stage 4 within 1 year?"

This is a perfectly valid clinical question. Next, you build visit-level inputs:

```text
visit 1 → visit 2 → visit 3 → … → visit T
```

Then—and this is where the mistake happens—you train like this:

```python
# Patient-level label (same for all visits)
y_patient = 1  # if progression occurred within 1 year, else 0

# Visit-level representations from LSTM
out, _ = lstm(visit_sequences)  # shape: [batch_size, num_visits, hidden_dim]

# Apply the SAME label at EVERY timestep
for t in range(num_visits):
    loss += BCE(prediction_head(out[:, t]), y_patient)
```

### Why This Is Leakage (Not Just "Suboptimal")

This approach implicitly tells the model:

> "At visit 1, predict something that is only knowable **after visit T**."

**This violates causality.**

#### The Formal Problem

At visit $t$:

- Your **label** depends on events in the window $[t, t+365]$ days
- Your **input** includes information about the patient's *entire* trajectory
- The model can see that this patient *has* future visits (sequence length, padding patterns)

#### What the Model Actually Learns

Instead of learning disease dynamics, the LSTM learns spurious correlations:

- **Number of future visits**: Patients who progress tend to have longer follow-up
- **Visit density**: Sicker patients have more frequent visits
- **Padding patterns**: Sequence length reveals outcome
- **Time-to-last-visit**: Abrupt termination signals events

These are **future-derived signals** that won't be available at prediction time.

---

### Why Performance Looks Amazing (And Why Reviewers Miss It)

The model doesn't need to understand disease biology. It exploits dataset artifacts:

| Artifact | What Model Learns | Why It's Leakage |
|----------|-------------------|------------------|
| **Follow-up length** | Patients who progress → longer follow-up | Not available prospectively |
| **Sequence termination** | Patients who die → abrupt ending | Reveals outcome |
| **Visit frequency** | Severe disease → more frequent visits | Confounds risk with surveillance |

#### The Deceptive Metrics

This is why you see:

- **AUC jumps** from 0.72 → 0.92 (looks like a breakthrough!)
- **Perfect calibration** on held-out test set
- **Collapse in prospective validation** (the model fails in real deployment)

The model is "cheating" by using information that won't exist at prediction time.

---

### How to Prove Leakage Is Happening

Here are three diagnostic tests that always catch temporal leakage:

#### Test 1: Shuffle Visits Within Patients

```python
# Randomly permute visit order for each patient
shuffled_visits = [random.shuffle(patient_visits) for patient_visits in data]
performance_shuffled = evaluate_model(shuffled_visits)
```

**Interpretation:** If performance stays high, your model is **not using temporal disease dynamics**. It's relying on visit-level features or sequence statistics, not temporal patterns.

#### Test 2: Truncate Sequences at Random Times

```python
# Cut each sequence at a random point
for patient in data:
    truncate_at = random.randint(1, len(patient.visits))
    patient.visits = patient.visits[:truncate_at]
```

**Interpretation:** If performance drops sharply, your model was **using future context** (sequence length, padding, etc.).

#### Test 3: Predict Using Only Sequence Length

```python
# Simplest possible baseline
y_hat = logistic_regression(num_visits)
```

**Interpretation:** If this baseline is strong (AUC > 0.75), your task is **contaminated by follow-up artifacts**. The outcome is predictable from metadata, not clinical content.

---

## Part II: Designing Causal Progression Labels (The Right Way)

Now the real work begins.

### Step 1: Decide What You Are Predicting *From When*

Every prediction must answer this question:

> "At visit $t$, using information up to $t$, what am I predicting about the future?"

Examples of well-defined questions:

- Progression within next 6 months
- Progression before next visit
- Time to next disease stage
- Hazard of progression at this moment

If you cannot answer "from when," the label is ill-defined.

---

## Option A: Discrete-Time Horizon Labels (Most Practical)

This is the workhorse approach for most applications.

### Definition

For each visit $t$, define:

$$
y_t = \begin{cases}
1 & \text{if progression occurs in } (t, t + \Delta] \\
0 & \text{if no progression in } (t, t + \Delta] \\
\text{censored} & \text{if follow-up} < \Delta
\end{cases}
$$

Where $\Delta$ is a fixed horizon (e.g., 180 days).

### Implementation Logic

For each visit:

1. Look forward $\Delta$ days from visit $t$
2. If progression occurs → label = 1 (positive)
3. If patient is observed beyond $t + \Delta$ with no progression → label = 0 (negative)
4. Otherwise → drop or mark as censored

### Why This Is Causal

- **Label** uses only future after visit $t$
- **Input** uses only history before visit $t$
- No information flows backward in time

### Why It's Statistically Efficient

- Each patient contributes *multiple labeled visits*
- You're not collapsing the trajectory into one datapoint
- More training signal from the same data

### Loss Masking

```python
# Only compute loss on valid (non-censored) visits
valid_mask = (labels != CENSORED)
loss = BCE(logits[valid_mask], labels[valid_mask])
```

### Example: CKD Staging

At visit $t$:

- **Inputs**: diagnoses, labs, medications, time since last visit
- **Label**: Did CKD stage increase within next 180 days?
  - If yes → positive
  - If no but follow-up ≥ 180 days → negative
  - Else → censored

Repeat for each visit in the patient's sequence.

---

## Option B: Discrete-Time Survival Modeling (Cleaner, Stronger)

Instead of predicting a binary event in a fixed window, you predict **hazard** at each visit.

### Define Hazard at Visit $t$

$$
h_t = P(\text{progression at } t+1 \mid \text{no progression before } t)
$$

Read this as:

> "Given the patient has not progressed up to visit $t$, what is the probability they progress before the next visit?"

Your LSTM outputs $h_t \in (0, 1)$ for each visit.

### Likelihood Function

For each patient, the likelihood of their observed sequence is:

$$
\mathcal{L} = \prod_{t < T^*} (1 - h_t) \times \begin{cases}
h_{T^*} & \text{if event occurred} \\
1 & \text{if censored}
\end{cases}
$$

Where $T^*$ is the event or censoring time.

**Intuition:**

- $\prod_{t < T^*} (1 - h_t)$: Patient survived (didn't progress) through all visits before $T^*$
- $h_{T^*}$: Patient progressed at visit $T^*$ (if event observed)
- No hazard term if censored (we don't know what happened after)

### Why This Is Excellent

- **Naturally handles censoring**: No need to drop data
- **No arbitrary horizon**: Each visit contributes based on actual timing
- **Directly models disease dynamics**: Hazard is the fundamental quantity

### Why It's Harder

- Must implement a custom loss function
- Interpretation is subtler than binary classification
- Requires understanding of survival analysis concepts

### Implementation Preview

```python
def discrete_time_survival_loss(hazards, event_times, event_indicators):
    """
    hazards: [batch_size, num_visits] - predicted hazards
    event_times: [batch_size] - time of event or censoring
    event_indicators: [batch_size] - 1 if event, 0 if censored
    """
    log_likelihood = 0
    
    for i in range(batch_size):
        T = event_times[i]
        
        # Survival through all visits before T
        log_likelihood += torch.sum(torch.log(1 - hazards[i, :T]))
        
        # Event at T (if observed)
        if event_indicators[i] == 1:
            log_likelihood += torch.log(hazards[i, T])
    
    return -log_likelihood  # Negative log-likelihood
```

We'll implement this fully in Part 2.

---

## Option C: Continuous-Time Survival (Cox-Style)

This is the "epidemiologist-approved" approach.

Your LSTM outputs a **risk score** $r_t$, not a probability. The hazard function is:

$$
\lambda(t \mid x_t) = \lambda_0(t) \exp(r_t)
$$

Where:

- $\lambda_0(t)$ is the baseline hazard (learned or assumed)
- $r_t$ is the risk score from the LSTM
- This is the **proportional hazards** assumption

Train with partial likelihood (Cox partial likelihood).

### When This Shines

- Irregular visit spacing (visits at unpredictable times)
- Strong censoring (many patients lost to follow-up)
- Long follow-up windows (years of data)
- Need to compare with classical epidemiology methods

### When It's Overkill

- Early-stage prototyping
- Small datasets (< 1000 patients)
- Regular visit patterns (monthly check-ups)

---

## How to Avoid Throwing Away Data (Statistical Efficiency)

The fear people have:

> "If I censor aggressively, I'll lose half my visits."

The fix is **visit-level supervision**, not patient-level.

### Key Principle

> One patient = many training examples  
> One visit = one prediction moment

This is why LSTMs + visit-level labels are powerful *when done correctly*.

### Comparison: Patient-Level vs. Visit-Level

| Approach | Training Examples | Information Used |
|----------|-------------------|------------------|
| **Patient-level** | 1 per patient | Entire trajectory collapsed |
| **Visit-level** | $T$ per patient | Each visit independently |

With 1000 patients averaging 10 visits each:

- Patient-level: 1,000 training examples
- Visit-level: 10,000 training examples

**10x more data from the same patients!**

---

## A Final Sanity Rule

> **If a model can "predict" an outcome before the clinical evidence exists, you have leakage.**

High AUC is not a virtue. **Causality is.**

### The Gold Standard Test

Can you deploy this model prospectively?

- At visit $t$, can you make a prediction using only data up to $t$?
- Does the prediction make sense clinically?
- Would a clinician trust it?

If the answer to any of these is "no," you have a problem.

---

## Where This Naturally Leads Next

If you want to continue (and this is the real frontier):

- **Joint modeling** of visit frequency + disease progression
- **Separating** surveillance intensity from biological risk
- **Counterfactual** progression curves ("what if no ACE inhibitor?")
- **Multi-state** models (progression through multiple disease stages)

That's where EHR sequencing becomes *scientific*, not just predictive.

---

## Summary

**The Problem:**

- Patient-level labels + visit-level inputs = temporal leakage
- Models learn dataset artifacts, not disease dynamics

**The Solution:**

- Visit-level labels that respect causality
- Three options: fixed-horizon, discrete-time survival, continuous-time survival
- Choose based on your data and clinical question

**Next Steps:**

- Part 2 will dive deep into discrete-time survival modeling
- We'll derive the likelihood, implement the loss, and show examples
- You'll see why this is the cleanest approach for visit-based EHR sequences

---

**Continue to:** [Part 2: Discrete-Time Survival Modeling](causal-survival-analysis-2.md)
