# Tutorial 3: Loss Function Formulation

**Part of:** Discrete-Time Survival Analysis for EHR Sequences  
**Audience:** Researchers implementing survival models with deep learning

---

## Table of Contents
1. [Overview](#overview)
2. [Discrete-Time Survival Framework](#discrete-time-survival-framework)
3. [Likelihood Formulation](#likelihood-formulation)
4. [Implementation Details](#implementation-details)
5. [Training Considerations](#training-considerations)
6. [Common Issues and Solutions](#common-issues-and-solutions)

---

## Overview

### The Goal

Train a neural network to predict **hazard** at each time point:

$$h_t = P(T = t \mid T \geq t, H_t)$$

Where:
- $T$: Time of event
- $h_t$: Probability of event at time $t$ given survival to $t$
- $H_t$: Patient history up to time $t$

### Why Not Binary Cross-Entropy?

**Binary classification approach (WRONG):**
```python
# Treat each visit as binary: event (1) or no event (0)
loss = BCE(predictions, labels)
```

**Problems:**
1. Ignores **survival information** (patient survived to this visit)
2. Doesn't handle **censoring** properly
3. Treats all time points independently

**Survival approach (CORRECT):**
```python
# Model survival process explicitly
loss = -log_likelihood(hazards, event_times, event_indicators)
```

**Benefits:**
1. Uses survival information from all visits before event
2. Handles censoring naturally
3. Respects temporal dependencies

---

## Discrete-Time Survival Framework

### Hazard Function

**Definition:** Probability of event at time $t$ given survival to $t$

$$h_t = P(T = t \mid T \geq t)$$

**Properties:**
- $0 \leq h_t \leq 1$ (it's a probability)
- Can vary arbitrarily over time (no parametric assumptions)
- Predicted by neural network with sigmoid activation

**Example:**
```python
# Patient trajectory
h_1 = 0.05  # Low risk at visit 1
h_2 = 0.08  # Slightly higher at visit 2
h_3 = 0.15  # Increasing risk at visit 3
h_4 = 0.40  # High risk at visit 4 (event occurs)
```

### Survival Function

**Definition:** Probability of surviving past time $t$

$$S(t) = P(T > t) = \prod_{i=1}^{t} (1 - h_i)$$

**Interpretation:**
- Survival = not having event at any prior time
- Product of $(1 - h_i)$ for all times up to $t$

**Example:**
```python
S(1) = (1 - h_1) = 0.95
S(2) = (1 - h_1)(1 - h_2) = 0.95 × 0.92 = 0.874
S(3) = (1 - h_1)(1 - h_2)(1 - h_3) = 0.874 × 0.85 = 0.743
```

### Probability Mass Function

**Definition:** Probability of event exactly at time $t$

$$P(T = t) = S(t-1) \times h_t = \left[\prod_{i=1}^{t-1} (1 - h_i)\right] \times h_t$$

**Interpretation:**
- Survive to $t-1$: $\prod_{i=1}^{t-1} (1 - h_i)$
- Then have event at $t$: $h_t$

**Example:**
```python
# Probability of event at visit 3
P(T = 3) = S(2) × h_3
         = (1 - h_1)(1 - h_2) × h_3
         = 0.95 × 0.92 × 0.15
         = 0.131
```

---

## Likelihood Formulation

### For a Single Patient

**Case 1: Event Observed (δ = 1)**

Patient has event at time $T$:

$$L = P(T = T) = \left[\prod_{t=1}^{T-1} (1 - h_t)\right] \times h_T$$

**Interpretation:**
- Survive through visits $1, 2, \ldots, T-1$: $\prod_{t=1}^{T-1} (1 - h_t)$
- Have event at visit $T$: $h_T$

**Log-likelihood:**
$$\log L = \sum_{t=1}^{T-1} \log(1 - h_t) + \log(h_T)$$

**Case 2: Censored (δ = 0)**

Patient is censored at time $T$ (no event observed):

$$L = P(T > T) = \prod_{t=1}^{T} (1 - h_t)$$

**Interpretation:**
- Survive through all observed visits: $\prod_{t=1}^{T} (1 - h_t)$
- We don't know what happens after

**Log-likelihood:**
$$\log L = \sum_{t=1}^{T} \log(1 - h_t)$$

### Combined Formulation

**For any patient:**

$$\log L = \sum_{t=1}^{T} \log(1 - h_t) + \delta \cdot \log(h_T)$$

Where:
- First term: Survival contribution (all patients)
- Second term: Event contribution (only if $\delta = 1$)

**Batch Loss:**

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[\sum_{t=1}^{T_i} \log(1 - h_{i,t}) + \delta_i \cdot \log(h_{i,T_i})\right]$$

---

## Implementation Details

### PyTorch Implementation

```python
class DiscreteTimeSurvivalLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps  # Numerical stability
    
    def forward(self, hazards, event_times, event_indicators, sequence_mask):
        """
        Args:
            hazards: [batch_size, max_visits] in (0, 1)
            event_times: [batch_size] - index of event/censoring
            event_indicators: [batch_size] - 1 if event, 0 if censored
            sequence_mask: [batch_size, max_visits] - valid visits
        
        Returns:
            Scalar loss (negative log-likelihood)
        """
        batch_size, max_visits = hazards.shape
        
        # Clamp hazards for numerical stability
        hazards = torch.clamp(hazards, min=self.eps, max=1 - self.eps)
        
        # Create time index tensor
        time_idx = torch.arange(max_visits, device=hazards.device).unsqueeze(0)
        event_times_expanded = event_times.unsqueeze(1)
        
        # Mask for visits before event/censoring
        before_event_mask = (time_idx < event_times_expanded).float() * sequence_mask
        
        # Mask for event visit
        at_event_mask = (time_idx == event_times_expanded).float() * sequence_mask
        
        # Survival log-likelihood: sum of log(1 - h_t) for t < T
        survival_ll = torch.sum(
            torch.log(1 - hazards) * before_event_mask,
            dim=1
        )
        
        # Event log-likelihood: log(h_T) if event occurred
        event_ll = torch.sum(
            torch.log(hazards) * at_event_mask,
            dim=1
        ) * event_indicators.float()
        
        # Total log-likelihood per patient
        log_likelihood = survival_ll + event_ll
        
        # Return negative log-likelihood (to minimize)
        return -torch.mean(log_likelihood)
```

### Step-by-Step Example

**Patient data:**
```python
hazards = [0.05, 0.08, 0.15, 0.40, 0.0]  # Padded to max_visits=5
event_time = 3  # Event at visit 3 (0-indexed)
event_indicator = 1  # Event observed
sequence_mask = [1, 1, 1, 1, 0]  # 4 valid visits
```

**Step 1: Create masks**
```python
time_idx = [0, 1, 2, 3, 4]
event_time_expanded = 3

before_event_mask = [1, 1, 1, 0, 0]  # Visits 0, 1, 2
at_event_mask = [0, 0, 0, 1, 0]      # Visit 3
```

**Step 2: Survival log-likelihood**
```python
survival_ll = log(1 - 0.05) + log(1 - 0.08) + log(1 - 0.15)
            = log(0.95) + log(0.92) + log(0.85)
            = -0.051 - 0.083 - 0.163
            = -0.297
```

**Step 3: Event log-likelihood**
```python
event_ll = log(0.40) × 1  # event_indicator = 1
         = -0.916
```

**Step 4: Total log-likelihood**
```python
log_likelihood = -0.297 + (-0.916) = -1.213
```

**Step 5: Loss (negative log-likelihood)**
```python
loss = -(-1.213) = 1.213
```

### Numerical Stability

**Problem:** `log(0)` is undefined

**Solution:** Clamp hazards
```python
eps = 1e-7
hazards = torch.clamp(hazards, min=eps, max=1 - eps)

# Now:
# log(1 - hazards) is safe (1 - hazards >= eps)
# log(hazards) is safe (hazards >= eps)
```

**Why this works:**
- `eps = 1e-7` is tiny (0.0000001)
- Doesn't affect predictions (hazards are typically 0.01-0.9)
- Prevents numerical overflow/underflow

---

## Training Considerations

### 1. Masking for Variable-Length Sequences

**Problem:** Patients have different numbers of visits

**Solution:** Use sequence mask
```python
# Only compute loss for valid visits
survival_ll = torch.sum(
    torch.log(1 - hazards) * before_event_mask * sequence_mask,
    dim=1
)
```

**Example:**
```python
# Patient 1: 10 visits
sequence_mask[0] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]

# Patient 2: 5 visits
sequence_mask[1] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ...]

# Loss only computed for valid visits (mask = 1)
```

### 2. Handling Censoring

**Censored patient (δ = 0):**
```python
# Only survival term contributes
log_likelihood = survival_ll + 0 * event_ll
               = survival_ll
```

**Observed event (δ = 1):**
```python
# Both terms contribute
log_likelihood = survival_ll + 1 * event_ll
               = survival_ll + event_ll
```

**Key insight:** Censored patients still provide information through survival term!

### 3. Gradient Flow

**Survival term gradient:**
```python
∂L/∂h_t = -1/(1 - h_t)  for t < T
```
- Encourages low hazard before event
- Stronger gradient when $h_t$ is high (bad prediction)

**Event term gradient:**
```python
∂L/∂h_T = -1/h_T  for t = T (if event)
```
- Encourages high hazard at event time
- Stronger gradient when $h_T$ is low (bad prediction)

**Combined effect:**
- Model learns to predict low hazard early
- High hazard at event time
- Smooth transition between them

### 4. Batch Size Considerations

**Small batches (16-32):**
- Faster iterations
- More noise in gradient
- Better for small datasets

**Large batches (64-128):**
- Stable gradients
- Slower iterations
- Better for large datasets

**Recommendation:** Start with 32, adjust based on dataset size

---

## Common Issues and Solutions

### Issue 1: Loss Not Decreasing

**Symptoms:**
- Loss stays constant or increases
- Model predicts same hazard for all patients

**Possible causes:**

1. **Learning rate too high**
   ```python
   # Try smaller learning rate
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Instead of 1e-3
   ```

2. **Numerical instability**
   ```python
   # Ensure epsilon clamping
   hazards = torch.clamp(hazards, min=1e-7, max=1-1e-7)
   ```

3. **Weak synthetic data correlation**
   ```python
   # Check correlation
   correlation = np.corrcoef(risk_scores, event_times)[0, 1]
   # Should be r < -0.5
   ```

### Issue 2: Exploding Gradients

**Symptoms:**
- Loss becomes NaN
- Gradients become very large

**Solutions:**

1. **Gradient clipping**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Check for log(0)**
   ```python
   # Ensure epsilon clamping in loss function
   hazards = torch.clamp(hazards, min=self.eps, max=1 - self.eps)
   ```

3. **Reduce learning rate**
   ```python
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   ```

### Issue 3: Overfitting

**Symptoms:**
- Training loss decreases, validation loss increases
- Large gap between train and validation C-index

**Solutions:**

1. **Dropout**
   ```python
   model = DiscreteTimeSurvivalLSTM(
       vocab_size=vocab_size,
       dropout=0.3  # Increase dropout
   )
   ```

2. **L2 regularization**
   ```python
   optimizer = torch.optim.Adam(
       model.parameters(),
       lr=1e-3,
       weight_decay=1e-4  # L2 penalty
   )
   ```

3. **Early stopping**
   ```python
   if val_loss > best_val_loss:
       patience_counter += 1
       if patience_counter >= patience:
           break  # Stop training
   ```

### Issue 4: Poor C-index Despite Low Loss

**Symptoms:**
- Loss decreases normally
- C-index stays around 0.5 (random)

**Possible causes:**

1. **Length bias in risk score**
   ```python
   # WRONG: Sum of hazards (length-dependent)
   risk_score = hazards.sum(dim=1)
   
   # CORRECT: Fixed-horizon mean
   risk_score = hazards[:, :10].mean(dim=1)
   ```

2. **Weak correlation in synthetic data**
   ```python
   # Regenerate with stronger correlation
   generator = DiscreteTimeSurvivalGenerator(
       time_scale=0.3,
       noise_std=0.05  # Reduce noise
   )
   ```

3. **Model not learning from data**
   ```python
   # Check if model is actually using input
   # Try predicting with random input - should be worse
   ```

---

## Comparison with Other Losses

### Binary Cross-Entropy (WRONG)

```python
# Treats each visit independently
loss = BCE(hazards, labels)

# Problems:
# - Doesn't use survival information
# - Doesn't handle censoring
# - Ignores temporal structure
```

### Mean Squared Error (WRONG)

```python
# Regression on event time
loss = MSE(predicted_time, actual_time)

# Problems:
# - Doesn't handle censoring
# - Assumes Gaussian errors (wrong for time-to-event)
# - Doesn't model hazard process
```

### Discrete-Time Survival (CORRECT)

```python
# Models survival process explicitly
loss = -log_likelihood(hazards, event_times, event_indicators)

# Benefits:
# - Uses all survival information
# - Handles censoring naturally
# - Respects temporal dependencies
# - Theoretically grounded
```

---

## Key Takeaways

1. **Discrete-time survival loss** models the survival process explicitly
2. **Two components:** Survival term (all patients) + Event term (observed events only)
3. **Handles censoring** naturally through likelihood formulation
4. **Numerical stability** requires epsilon clamping
5. **Masking** is essential for variable-length sequences
6. **C-index evaluation** requires careful risk score formulation to avoid length bias

---

## Further Reading

### Theory
- Singer & Willett (2003). *Applied Longitudinal Data Analysis* - Chapter 10
- Tutz & Schmid (2016). *Modeling Discrete Time-to-Event Data*

### Implementation
- PyHealth: `pyhealth.models.DeepSurv`
- PyCox: `pycox.models.LogisticHazard`

### Related Approaches
- DeepSurv (continuous-time with Cox model)
- DeepHit (competing risks)
- DRSA (deep recurrent survival analysis)

---

**Previous Tutorial:** [Synthetic Data Design](TUTORIAL_02_synthetic_data_design.md)  
**Notebook:** [01_discrete_time_survival_lstm.ipynb](01_discrete_time_survival_lstm.ipynb)
