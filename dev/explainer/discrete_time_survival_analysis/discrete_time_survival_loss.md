# Discrete Time Survival Loss Function

## Overview

The `DiscreteTimeSurvivalLoss` class implements a loss function for discrete-time survival analysis, which is particularly well-suited for modeling time-to-event outcomes in visit-based Electronic Health Record (EHR) sequences. This includes applications like disease progression, hospital readmission, or mortality prediction.

## Mathematical Foundation

### The Discrete-Time Survival Model

In discrete-time survival analysis, time is divided into discrete intervals (in our case, clinical visits). For each patient, we model:

- **Hazard function** $h_t$: The probability that an event occurs at visit $t$, given that it has not occurred before visit $t$
- **Survival function** $S_t$: The probability of surviving (not experiencing the event) up to and including visit $t$

The relationship between hazard and survival is:

$$S_t = \prod_{j=1}^{t} (1 - h_j)$$

### The Likelihood Function

For a patient with:
- Event time or censoring time $T$ (0-indexed visit number)
- Event indicator $\delta \in \{0, 1\}$ where:
  - $\delta = 1$ if event was observed
  - $\delta = 0$ if patient was censored (event not observed)

The likelihood contribution is:

$$\mathcal{L} = \left[\prod_{t=1}^{T-1} (1 - h_t)\right] \times h_T^{\delta}$$

This decomposes into two components:

1. **Survival component**: $\prod_{t=1}^{T-1} (1 - h_t)$ - probability of surviving all visits before $T$
2. **Event component**: $h_T^{\delta}$ - probability of event at time $T$ (only if event observed)

### The Loss Function

Taking the negative log-likelihood to create a loss function to minimize:

$$\mathcal{L}_{\text{NLL}} = -\log \mathcal{L} = -\sum_{t=1}^{T-1} \log(1 - h_t) - \delta \cdot \log(h_T)$$

For a batch of $N$ patients, we average:

$$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[\sum_{t=1}^{T_i-1} \log(1 - h_{i,t}) + \delta_i \cdot \log(h_{i,T_i})\right]$$

## Implementation Details

### Code Structure

The loss is implemented in `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/models/losses.py:16-116`

```python
class DiscreteTimeSurvivalLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
```

### Input Tensors

The `forward` method accepts four tensors:

1. **`hazards`** - Shape: `[batch_size, max_visits]`
   - Predicted hazard probabilities at each visit
   - Values in $(0, 1)$, typically from sigmoid activation
   - Example: `hazards[i, t]` = probability of event at visit $t$ for patient $i$

2. **`event_times`** - Shape: `[batch_size]`
   - 0-indexed visit number where event occurred or censoring happened
   - Example: `event_times[i] = 5` means event/censoring at 6th visit (index 5)

3. **`event_indicators`** - Shape: `[batch_size]`
   - Binary indicator: 1 if event observed, 0 if censored
   - Example: `event_indicators[i] = 1` means patient $i$ had the event

4. **`sequence_mask`** - Shape: `[batch_size, max_visits]`
   - Binary mask: 1 for valid visits, 0 for padding
   - Handles variable-length sequences in batched processing

### Algorithm Walkthrough

Let's trace through the implementation step by step:

#### Step 1: Hazard Clamping
```python
hazards = torch.clamp(hazards, min=self.eps, max=1 - self.eps)
```
- Prevents numerical instability from `log(0)` or `log(1)`
- Clamps hazards to $[\epsilon, 1-\epsilon]$ where $\epsilon = 10^{-7}$

#### Step 2: Create Time Index
```python
time_idx = torch.arange(max_visits, device=hazards.device).unsqueeze(0)
# Shape: [1, max_visits], e.g., [[0, 1, 2, 3, 4, ...]]
```
- Creates a reference tensor for visit indices
- Broadcasting-compatible with batch dimension

#### Step 3: Expand Event Times
```python
event_times_expanded = event_times.unsqueeze(1)
# Shape: [batch_size, 1]
```
- Prepares event times for broadcasting comparison

#### Step 4: Create Masks

**Before-event mask** (visits $t < T$):
```python
before_event_mask = (time_idx < event_times_expanded).float() * sequence_mask
# Shape: [batch_size, max_visits]
```
- Identifies all visits before the event/censoring time
- Multiplied by `sequence_mask` to exclude padding

**At-event mask** (visit $t = T$):
```python
at_event_mask = (time_idx == event_times_expanded).float() * sequence_mask
# Shape: [batch_size, max_visits]
```
- Identifies the exact visit where event/censoring occurred

#### Step 5: Compute Survival Log-Likelihood
```python
survival_ll = torch.sum(
    torch.log(1 - hazards) * before_event_mask,
    dim=1
)
# Shape: [batch_size]
```
- Computes $\sum_{t < T} \log(1 - h_t)$ for each patient
- Sums over the time dimension (dim=1)
- Represents the log-probability of surviving all visits before $T$

#### Step 6: Compute Event Log-Likelihood
```python
event_ll = torch.sum(
    torch.log(hazards) * at_event_mask,
    dim=1
) * event_indicators.float()
# Shape: [batch_size]
```
- Computes $\log(h_T)$ at the event time
- Multiplied by `event_indicators` to zero out for censored patients
- For censored patients ($\delta = 0$), this contributes 0 to the likelihood

#### Step 7: Combine and Return Loss
```python
log_likelihood = survival_ll + event_ll
return -torch.mean(log_likelihood)
```
- Combines survival and event components
- Negates to convert log-likelihood to loss (for minimization)
- Averages over batch

## Concrete Example

Let's walk through a specific example with 2 patients:

### Input Data
```python
hazards = torch.tensor([
    [0.1, 0.2, 0.3, 0.0],  # Patient 0
    [0.15, 0.25, 0.0, 0.0]  # Patient 1
])

event_times = torch.tensor([2, 1])  # Events at visits 3 and 2 (0-indexed)
event_indicators = torch.tensor([1, 1])  # Both events observed
sequence_mask = torch.tensor([
    [1, 1, 1, 0],  # Patient 0: 3 valid visits
    [1, 1, 0, 0]   # Patient 1: 2 valid visits
])
```

### Patient 0 Analysis
- Event at visit 2 (3rd visit), observed ($\delta = 1$)
- Visits before event: 0, 1
- Hazards: $h_0 = 0.1, h_1 = 0.2, h_2 = 0.3$

**Survival component** (visits 0, 1):
$$\log(1 - 0.1) + \log(1 - 0.2) = \log(0.9) + \log(0.8) \approx -0.105 - 0.223 = -0.328$$

**Event component** (visit 2):
$$\log(0.3) \approx -1.204$$

**Total log-likelihood**: $-0.328 + (-1.204) = -1.532$

### Patient 1 Analysis
- Event at visit 1 (2nd visit), observed ($\delta = 1$)
- Visits before event: 0
- Hazards: $h_0 = 0.15, h_1 = 0.25$

**Survival component** (visit 0):
$$\log(1 - 0.15) = \log(0.85) \approx -0.163$$

**Event component** (visit 1):
$$\log(0.25) \approx -1.386$$

**Total log-likelihood**: $-0.163 + (-1.386) = -1.549$

### Final Loss
$$\text{Loss} = -\frac{1}{2}(-1.532 - 1.549) = \frac{3.081}{2} = 1.541$$

## Handling Censoring

Censoring is a critical concept in survival analysis. A patient is **censored** when:
- They drop out of the study before experiencing the event
- The study ends before they experience the event
- They experience a competing event that prevents observation

### How the Loss Handles Censoring

For a censored patient at time $T$ ($\delta = 0$):

$$\mathcal{L}_{\text{censored}} = -\sum_{t=1}^{T} \log(1 - h_t)$$

Note that:
1. We still include the survival probability up to time $T$
2. We do NOT include the event term $\log(h_T)$ because $\delta = 0$
3. This represents "we know they survived at least until $T$"

### Example: Censored Patient
```python
hazards = torch.tensor([[0.1, 0.2, 0.3]])
event_times = torch.tensor([2])
event_indicators = torch.tensor([0])  # Censored!
sequence_mask = torch.tensor([[1, 1, 1]])
```

**Computation**:
- Survival component: $\log(0.9) + \log(0.8) + \log(0.7) \approx -0.828$
- Event component: $0$ (because $\delta = 0$)
- Total: $-0.828$

The loss encourages low hazards for all observed visits, reflecting that the patient survived those visits.

## Theoretical Properties

### 1. Proper Scoring Rule
The negative log-likelihood is a **proper scoring rule**, meaning it is minimized when the predicted hazards match the true hazard probabilities. This ensures the model is incentivized to produce well-calibrated probabilities.

### 2. Handling Right-Censoring
The loss correctly handles **right-censored** data (the most common type in medical studies). Right-censoring occurs when we observe a patient up to time $T$ but don't know if/when the event occurs after $T$.

### 3. Gradient Properties
The gradients with respect to hazards are:

For visits $t < T$:
$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{1}{1 - h_t}$$

For visit $t = T$ (if event observed):
$$\frac{\partial \mathcal{L}}{\partial h_T} = -\frac{\delta}{h_T}$$

These gradients:
- Push hazards down for visits where patient survived
- Push hazards up at the event time (if observed)
- Are well-behaved due to clamping

## Comparison with Other Approaches

### vs. Cox Proportional Hazards
- **Cox model**: Continuous-time, semi-parametric, requires proportional hazards assumption
- **Discrete-time**: Naturally fits visit-based data, no proportional hazards assumption, easier to implement in neural networks

### vs. Binary Classification at Each Visit
- **Naive approach**: Treat each visit independently with BCE loss
- **Problem**: Ignores temporal dependencies and censoring structure
- **Discrete survival**: Properly accounts for the fact that surviving visit $t$ requires surviving all previous visits

### vs. Regression on Event Time
- **Regression**: Predict time directly (e.g., MSE on event times)
- **Problem**: Cannot handle censored data properly
- **Discrete survival**: Naturally incorporates censoring through likelihood

## Usage in EHR Sequence Models

### Typical Model Architecture
```python
class SurvivalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output hazard probabilities
        )
    
    def forward(self, visit_sequences, sequence_lengths):
        # visit_sequences: [batch_size, max_visits, input_dim]
        rnn_out, _ = self.rnn(visit_sequences)
        # rnn_out: [batch_size, max_visits, hidden_dim]
        
        hazards = self.hazard_head(rnn_out).squeeze(-1)
        # hazards: [batch_size, max_visits]
        
        return hazards
```

### Training Loop
```python
model = SurvivalRNN(input_dim=100, hidden_dim=128)
loss_fn = DiscreteTimeSurvivalLoss(eps=1e-7)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    visit_sequences = batch['visits']
    event_times = batch['event_times']
    event_indicators = batch['event_indicators']
    sequence_mask = batch['mask']
    
    # Forward pass
    hazards = model(visit_sequences, sequence_mask.sum(dim=1))
    
    # Compute loss
    loss = loss_fn(hazards, event_times, event_indicators, sequence_mask)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Evaluation Metrics

The loss function is used for training, but evaluation typically uses:

### 1. Concordance Index (C-index)
Implemented in `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/models/losses.py:217-281`

- Measures ranking quality: patients with earlier events should have higher predicted risk
- Range: [0, 1], where 0.5 = random, 1.0 = perfect
- Analogous to AUC-ROC but for survival data

### 2. Calibration Curves
- Compare predicted vs. observed event rates
- Ensures probabilities are well-calibrated

### 3. Time-Dependent AUC
- AUC for predicting events within specific time windows
- Useful for clinical decision-making

## Common Pitfalls and Solutions

### 1. Numerical Instability
**Problem**: `log(0)` or `log(1)` causing NaN gradients

**Solution**: Hazard clamping with `eps=1e-7`
```python
hazards = torch.clamp(hazards, min=self.eps, max=1 - self.eps)
```

### 2. Imbalanced Event Rates
**Problem**: If events are rare, model may predict low hazards everywhere

**Solutions**:
- Use class weights or focal loss modifications
- Ensure sufficient positive examples in each batch
- Monitor calibration, not just discrimination

### 3. Variable Sequence Lengths
**Problem**: Patients have different numbers of visits

**Solution**: Use `sequence_mask` to handle padding properly
```python
before_event_mask = (time_idx < event_times_expanded).float() * sequence_mask
```

### 4. Temporal Leakage
**Problem**: Using future information to predict past events

**Solution**: Ensure model only uses information available up to prediction time
- Use causal masking in attention mechanisms
- Validate that features don't contain future information

## References

### Foundational Papers
1. **Singer, J. D., & Willett, J. B. (2003)**. *Applied Longitudinal Data Analysis: Modeling Change and Event Occurrence*. Oxford University Press.
   - Comprehensive treatment of discrete-time survival models
   - Chapter 10-11 cover the likelihood derivation

2. **Tutz, G., & Schmid, M. (2016)**. *Modeling Discrete Time-to-Event Data*. Springer.
   - Modern statistical perspective
   - Extensions to competing risks and multi-state models

### Deep Learning Applications
3. **Katzman, J. L., et al. (2018)**. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." *BMC Medical Research Methodology*, 18(1), 24.
   - Neural network approach to survival analysis

4. **Lee, C., et al. (2018)**. "DeepHit: A deep learning approach to survival analysis with competing risks." *AAAI Conference on Artificial Intelligence*.
   - Discrete-time deep survival model

### EHR-Specific Applications
5. **Rajkomar, A., et al. (2018)**. "Scalable and accurate deep learning with electronic health records." *NPJ Digital Medicine*, 1(1), 18.
   - Application to clinical prediction tasks

## Appendix: Alternative Formulations

### Cumulative Hazard Formulation
Instead of modeling $h_t$ directly, some approaches model cumulative hazard:
$$H_t = \sum_{j=1}^{t} h_j$$

Then survival is:
$$S_t = \exp(-H_t)$$

### Competing Risks Extension
For multiple event types (e.g., death vs. transplant):
$$h_{t,k} = P(\text{event type } k \text{ at time } t | \text{survived to } t)$$

Constraint: $\sum_{k=1}^{K} h_{t,k} \leq 1$

### Continuous-Time Approximation
As visit intervals become finer, discrete-time survival converges to continuous-time:
$$h_t \Delta t \approx \lambda(t) \Delta t$$

where $\lambda(t)$ is the continuous hazard rate.

---

**Document Version**: 1.0  
**Last Updated**: January 26, 2026  
**Maintainer**: EHR Sequencing Project Team
