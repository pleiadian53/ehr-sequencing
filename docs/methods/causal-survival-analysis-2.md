# Causal Survival Analysis for EHR Sequences (Part 2)

## Discrete-Time Survival Modeling: From Intuition to Implementation

**Previous:** [Part 1: Causal Progression Labels](causal-survival-analysis-1.md)

In Part 1, we identified the leakage problem and introduced three approaches to causal progression modeling. This tutorial focuses on **discrete-time survival modeling**—the most natural fit for visit-based EHR sequences.

We'll cover:

1. What "discrete-time survival" actually means
2. What censoring is, conceptually and operationally
3. Deriving the likelihood formula step by step
4. Implementing the loss function in PyTorch

No symbols without intuition. Every equation will be built from first principles.

---

## 1. What Discrete-Time Survival Modeling Really Is

### The Natural Discretization: Visits

You already have a natural discretization of time: **visits**.

Instead of modeling time as continuous (like Cox models with hazards per day), we say:

> Time advances in discrete steps: visit 1 → visit 2 → visit 3 → …

At each visit $t$, the patient is in one of two states:

- **Event-free** (alive / not yet progressed)
- **Event occurred** (e.g., disease progressed, died, readmitted)

We model **the probability that the event happens at the next step, given it has not happened before**.

That conditional probability is the **hazard**.

### Why This Makes Sense for EHR Data

EHR data is naturally discrete:

- Patients are observed at **specific visits**, not continuously
- Between visits, we have no information
- The "next visit" is the natural prediction horizon

This is fundamentally different from continuous-time survival analysis (Cox models), which assumes you can observe events at any moment in time.

---

## 2. The Hazard in Discrete Time

### Definition

At visit $t$, the hazard is:

$$
h_t = P(T = t \mid T \geq t) = P(\text{event at time } t \mid \text{survived to } t)
$$

Equivalently, in terms of visit intervals:

$$
h_t = P(\text{event in interval } (t-1, t] \mid \text{no event through } t-1)
$$

Read this as:

> "Given the patient has *not* progressed through visit $t-1$, what is the probability they progress by visit $t$?"

**Important:** The hazard $h_t$ represents the risk for the **current** time point $t$ (or equivalently, the interval since the last observation), not for some future time.

### Key Properties

**This is NOT:**

- Probability of ever progressing
- Cumulative risk over all time
- A patient-level summary statistic

**This IS:**

- **Local in time**: specific to this moment
- **Conditional**: depends on survival up to now
- **Causal**: uses only history up to $t$

### LSTM Output and Prediction Timing

Your LSTM outputs $h_t \in (0, 1)$ at each visit:

```python
# LSTM forward pass
hidden_states, _ = lstm(visit_sequences)  # [batch, num_visits, hidden_dim]

# Map to hazard (one value per visit)
hazards = torch.sigmoid(hazard_head(hidden_states))  # [batch, num_visits]
```

The sigmoid ensures $0 < h_t < 1$ (valid probability).

**Interpretation for prediction:**
- At visit $t$, the LSTM has seen data through visit $t$
- The hazard $h_t$ represents: "What is the risk that the event occurred by this visit?"
- For **forward prediction**, use $h_{t+1}$ (the hazard at the next visit) to predict future risk
- This maintains causality: predictions at time $t$ use only data through time $t$

---

## 3. What Censoring Is (And Why It's Unavoidable)

### The Core Problem

In real data, you often don't know what eventually happened.

**Examples:**

- Patient moved to another hospital
- Study ended before event occurred
- Patient still stable at last observation
- Lost to follow-up

You only know:

> "Up to this point, no event has occurred."

That is **censoring**.

### Formal Definition

A patient is **right-censored** if:

- They have no observed event
- But follow-up ends at time $T_c$

**Crucially:**

- Censoring is *not* a negative outcome
- It means "unknown beyond this point"
- We cannot say the event *didn't* happen—we just don't know

### Why Censoring Matters

If you treat censored patients as "no event," you're making a false assumption. This biases your model to underestimate risk.

**Correct approach:** Use censoring information in the likelihood, but don't force a label.

---

## 4. Timeline Picture (Mental Model)

### Patient 1: Event Observed

```text
Visit:    1 ---- 2 ---- 3 ---- 4 ---- 5
Event:                     ↑
                           progression
```

- Visits 1-3: No event (contributes survival information)
- Visit 4: Event occurs (contributes hazard information)

### Patient 2: Censored

```text
Visit:    1 ---- 2 ---- 3
Event:          (study ends here)
```

- Visits 1-3: No event (contributes survival information)
- After visit 3: Unknown (no contribution)

**Both patients contribute information, but differently.**

---

## 5. Survival Probability and Hazard Are Linked

### Definitions

- $h_t$: hazard at visit $t$
- $S_t$: probability of surviving (no event) **through** visit $t$

### Relationship

$$
S_t = \prod_{k=1}^{t} (1 - h_k)
$$

### Interpretation

To survive **through** time $t$ (i.e., no event at times $1, 2, \ldots, t$), you must not have the event at any step:

- No event at time 1: probability $(1 - h_1)$
- No event at time 2: probability $(1 - h_2)$
- ...
- No event at time $t$: probability $(1 - h_t)$

The overall survival probability $S_t = P(T > t)$ is the product of all these "didn't happen" factors.

**Note:** $S_t$ means "survived through time $t$" (equivalently, "event-free through visit $t$"), so $S_t = P(T > t)$.

### Example

If hazards are constant at $h_t = 0.1$ for all $t$:

- $S_1 = 0.9$
- $S_2 = 0.9 \times 0.9 = 0.81$
- $S_3 = 0.9 \times 0.9 \times 0.9 = 0.729$
- ...

Survival probability decreases over time, even with constant hazard.

---

## 6. Deriving the Likelihood Formula

Let's derive the likelihood for **one patient** from first principles.

### Case A: Event Occurs at Time $T^*$

**What must be true?**

1. No event at times $1, 2, \ldots, T^*-1$
2. Event occurs at time $T^*$

**Probability:**

$$
P = \underbrace{(1 - h_1)(1 - h_2)\cdots(1 - h_{T^*-1})}_{\text{survived through time } T^*-1} \times \underbrace{h_{T^*}}_{\text{event at time } T^*}
$$

**Compactly:**

$$
\mathcal{L} = \left[\prod_{t=1}^{T^*-1} (1 - h_t)\right] \cdot h_{T^*}
$$

**Intuition:** The patient "rolled the dice" at each time point and didn't have the event, until time $T^*$ when the event occurred.

**Note:** We write $\prod_{t=1}^{T^*-1}$ to make explicit that we're taking the product over times $1$ through $T^*-1$, not including $T^*$.

---

### Case B: Patient Is Censored at Time $T_c$

**What do we know?**

- No event at times $1, 2, \ldots, T_c$
- After that: **unknown**

**Probability of observed data:**

$$
\mathcal{L} = \prod_{t=1}^{T_c} (1 - h_t)
$$

**Notice:**

- We only count survival through $T_c$
- **No event hazard term** (we don't know if/when the event happened after)
- We never claim the event *didn't* happen beyond $T_c$
- We just stop accumulating likelihood contributions

**Intuition:** The patient survived through all observed time points. We don't know what happened after.

**Subtlety:** Some formulations write this as $\prod_{t=1}^{T_c-1} (1-h_t)$ if censoring occurs "just before" time $T_c$. Here we use the convention that censoring at $T_c$ means the patient was observed event-free through time $T_c$.

---

### Case C: Unified Likelihood Formula

For patient $i$, the likelihood is:

$$
\mathcal{L}_i = \left[\prod_{t=1}^{T_i - 1} (1 - h_{it})\right] \times \left[h_{iT_i}\right]^{\delta_i}
$$

Where:

- $T_i$ is the observed time for patient $i$ (event or censoring)
- $h_{it}$ is the hazard at time $t$ for patient $i$
- $\delta_i$ is the event indicator: $\delta_i = 1$ if event observed, $\delta_i = 0$ if censored

**Explanation:**
- The first term $\prod_{t=1}^{T_i-1} (1 - h_{it})$ is the probability of surviving through time $T_i - 1$
- The second term $h_{iT_i}^{\delta_i}$ equals:
  - $h_{iT_i}$ if $\delta_i = 1$ (event observed)
  - $1$ if $\delta_i = 0$ (censored, no contribution from event hazard)

**That's the full likelihood. No magic. Just logic.**

---

## 7. Why This Formulation Is Powerful

### 1. Censoring Is Handled Correctly

- Censored patients still contribute survival information
- They are not mislabeled as "no event"
- No data is thrown away

### 2. Multiple Training Signals Per Patient

Each visit contributes:

- Either a "did not progress yet" factor: $(1 - h_t)$
- Or an "event happened now" factor: $h_t$

**This is far more data-efficient than patient-level labels.**

With 1000 patients averaging 10 visits each:

- Patient-level: 1,000 training examples
- Visit-level (survival): ~10,000 training signals

### 3. Natural Temporal Structure

The likelihood respects the sequential nature of the data:

- Earlier visits affect later probabilities through the survival product
- The model learns disease dynamics, not just static risk

---

## 8. Turning This Into a Loss Function

Take logs (because products are numerically unstable):

### For an Event Patient (Event at Time $T^*$)

$$
\log \mathcal{L} = \sum_{t=1}^{T^*-1} \log(1 - h_t) + \log(h_{T^*})
$$

### For a Censored Patient (Censored at Time $T_c$)

$$
\log \mathcal{L} = \sum_{t=1}^{T_c} \log(1 - h_t)
$$

### Unified Log-Likelihood

$$
\log \mathcal{L}_i = \sum_{t=1}^{T_i - 1} \log(1 - h_{it}) + \delta_i \log(h_{iT_i})
$$

### Negative Log-Likelihood (What You Minimize)

For a batch of patients:

$$
\text{Loss} = -\sum_{i=1}^{N} \log \mathcal{L}_i
$$

### Pseudocode

```python
loss = 0
for patient in batch:
    for t in range(patient.num_visits):
        if event_at_visit[t]:
            loss -= log(hazard[t])
            break  # Stop after event
        else:
            loss -= log(1 - hazard[t])
```

Mask appropriately across batch for variable-length sequences.

---

## 9. PyTorch Implementation

Here's a complete implementation:

```python
import torch
import torch.nn as nn

def discrete_time_survival_loss(hazards, event_times, event_indicators, sequence_mask):
    """
    Discrete-time survival analysis loss function.
    
    Args:
        hazards: [batch_size, max_visits] - predicted hazards at each visit
        event_times: [batch_size] - index of event or censoring (0-indexed)
        event_indicators: [batch_size] - 1 if event observed, 0 if censored
        sequence_mask: [batch_size, max_visits] - 1 for real visits, 0 for padding
    
    Returns:
        Negative log-likelihood (scalar)
    """
    batch_size, max_visits = hazards.shape
    
    # Clamp hazards to avoid log(0)
    hazards = torch.clamp(hazards, min=1e-7, max=1 - 1e-7)
    
    # Initialize log-likelihood
    log_likelihood = torch.zeros(batch_size, device=hazards.device)
    
    for i in range(batch_size):
        T = event_times[i].item()  # Event or censoring time
        
        # Contribution from survival up to time T
        # Sum log(1 - h_t) for all t < T
        if T > 0:
            survival_log_prob = torch.sum(
                torch.log(1 - hazards[i, :T]) * sequence_mask[i, :T]
            )
            log_likelihood[i] += survival_log_prob
        
        # Contribution from event at time T (if observed)
        if event_indicators[i] == 1 and T < max_visits:
            event_log_prob = torch.log(hazards[i, T])
            log_likelihood[i] += event_log_prob
    
    # Return negative log-likelihood (to minimize)
    return -torch.mean(log_likelihood)


# Alternative: Vectorized implementation (more efficient)
def discrete_time_survival_loss_vectorized(hazards, event_times, event_indicators, sequence_mask):
    """
    Vectorized version for better performance.
    """
    batch_size, max_visits = hazards.shape
    
    # Clamp hazards
    hazards = torch.clamp(hazards, min=1e-7, max=1 - 1e-7)
    
    # Create time index tensor
    time_idx = torch.arange(max_visits, device=hazards.device).unsqueeze(0)  # [1, max_visits]
    event_times_expanded = event_times.unsqueeze(1)  # [batch_size, 1]
    
    # Mask for visits before event/censoring
    before_event_mask = (time_idx < event_times_expanded).float() * sequence_mask
    
    # Mask for event visit
    at_event_mask = (time_idx == event_times_expanded).float() * sequence_mask
    
    # Log-likelihood from survival (all visits before event)
    survival_ll = torch.sum(
        torch.log(1 - hazards) * before_event_mask,
        dim=1
    )
    
    # Log-likelihood from event (only if event observed)
    event_ll = torch.sum(
        torch.log(hazards) * at_event_mask,
        dim=1
    ) * event_indicators
    
    # Total log-likelihood
    log_likelihood = survival_ll + event_ll
    
    return -torch.mean(log_likelihood)
```

### Usage Example

```python
# Model output
hazards = model(visit_sequences)  # [batch_size, num_visits]

# Ground truth
event_times = torch.tensor([5, 3, 10, 7])  # Visit index of event/censoring
event_indicators = torch.tensor([1, 1, 0, 1])  # 1=event, 0=censored
sequence_mask = create_sequence_mask(visit_sequences)  # [batch, num_visits]

# Compute loss
loss = discrete_time_survival_loss_vectorized(
    hazards, event_times, event_indicators, sequence_mask
)

# Backprop
loss.backward()
optimizer.step()
```

---

## 10. How the LSTM Fits In

### Architecture

```python
class DiscreteTimeSurvivalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hazard_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, visit_codes, visit_mask, sequence_mask):
        """
        Args:
            visit_codes: [batch, num_visits, max_codes_per_visit]
            visit_mask: [batch, num_visits, max_codes_per_visit]
            sequence_mask: [batch, num_visits]
        
        Returns:
            hazards: [batch, num_visits] - hazard at each visit
        """
        # Embed codes
        embeddings = self.embedding(visit_codes)  # [B, V, C, E]
        
        # Aggregate codes within each visit (mean pooling)
        visit_mask_expanded = visit_mask.unsqueeze(-1)  # [B, V, C, 1]
        visit_vectors = (embeddings * visit_mask_expanded).sum(dim=2) / \
                       visit_mask.sum(dim=2, keepdim=True).clamp(min=1)  # [B, V, E]
        
        # LSTM over visits
        lstm_out, _ = self.lstm(visit_vectors)  # [B, V, H]
        
        # Map to hazard (sigmoid for valid probability)
        hazards = torch.sigmoid(self.hazard_head(lstm_out)).squeeze(-1)  # [B, V]
        
        return hazards
```

### Training Loop

```python
model = DiscreteTimeSurvivalLSTM(vocab_size=5000, embedding_dim=128, hidden_dim=256)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        hazards = model(
            batch['visit_codes'],
            batch['visit_mask'],
            batch['sequence_mask']
        )
        
        # Compute loss
        loss = discrete_time_survival_loss_vectorized(
            hazards,
            batch['event_times'],
            batch['event_indicators'],
            batch['sequence_mask']
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 11. Why This Is Causal by Construction

At time $t$:

- **Input** to the LSTM uses only history through time $t$ (due to LSTM causality)
- **Hazard** $h_t$ represents the risk **at time** $t$ (given history through $t-1$)
- For **prediction** at time $t$: use $h_{t+1}, h_{t+2}, \ldots$ to forecast future risk
- **No future leakage** is possible (assuming correct masking)

This is the cleanest way to do progression modeling with sequences.

### Clarifying Prediction vs. Estimation

**During training:**
- At each time $t$, we estimate $h_t$ using data through time $t$
- This is appropriate for likelihood computation (we're modeling the data generation process)

**During inference/prediction:**
- If we want to predict future risk from visit $t$, we look at $h_{t+1}, h_{t+2}, \ldots$
- These represent "what will happen next" from the perspective of time $t$
- Each $h_{t+k}$ uses only information through time $t+k$, maintaining causality

### Verifying Causality

```python
# Test: Shuffle future visits (shouldn't affect prediction at time t)
def test_causality(model, sequence, t):
    # Prediction at time t
    pred_original = model(sequence[:t+1])
    
    # Shuffle visits after t
    shuffled = sequence.copy()
    shuffled[t+1:] = np.random.permutation(shuffled[t+1:])
    
    # Prediction should be identical
    pred_shuffled = model(shuffled[:t+1])
    
    assert torch.allclose(pred_original, pred_shuffled)
```

---

## 12. Common Misunderstandings

### "Isn't this just many binary classifiers?"

**No.**

- The hazards are **coupled through survival**: $S_t = \prod_{k=1}^t (1 - h_k)$
- Earlier predictions affect later likelihood
- The model learns temporal dependencies, not independent classifications

### "Does censoring mean negative?"

**No.**

- **Negative** = observed no event in a specific window
- **Censored** = unknown beyond this point
- They are fundamentally different concepts

### "Why not just use Cox regression?"

You can—but:

- Cox assumes **proportional hazards** (hazard ratios constant over time)
- Discrete-time lets the network learn **time-varying risk** flexibly
- Easier to integrate with visit-based LSTMs
- No need to estimate baseline hazard

---

## 13. A Final Intuition Anchor

Think of each time point as asking:

> "Did the event occur **at this time** (in the interval since the last observation)?"

- Most of the time, the answer is "not yet" → contributes $(1 - h_t)$
- Once (for event patients), the answer is "yes" → contributes $h_t$
- Sometimes, you stop asking → censored (no more likelihood contributions)

**That's discrete-time survival.**

### Reconciling with "Prediction"

If you think in terms of "predicting the future":
- At time $t$ (having seen data through $t$), you've computed $h_t$
- To predict "what happens next," you'd look at $h_{t+1}$ (which uses data through $t+1$)
- But during training, we're not predicting—we're estimating the hazard function that generated the observed data

---

## 14. Practical Considerations

### Handling Irregular Visit Spacing

If visits are irregularly spaced, you can:

1. **Include time delta as a feature**:
   ```python
   time_since_last_visit = compute_time_deltas(visit_timestamps)
   lstm_input = torch.cat([visit_vectors, time_deltas], dim=-1)
   ```

2. **Use time-dependent hazard**:
   ```python
   hazard = sigmoid(W_h @ hidden + W_t @ time_delta + b)
   ```

### Competing Risks

If multiple event types can occur (e.g., progression vs. death):

```python
# Multi-output hazard head
hazards = softmax(hazard_head(lstm_out))  # [B, V, num_event_types]

# Cause-specific likelihood
for event_type in range(num_event_types):
    if observed_event == event_type:
        ll += log(hazards[:, :, event_type])
```

### Recurrent Events

For events that can happen multiple times:

- Reset after each event
- Model inter-event times
- Use counting process formulation

---

## 15. Next Steps

### Implementation Checklist

- [ ] Implement discrete-time survival loss
- [ ] Create data loader with event times and indicators
- [ ] Train LSTM with survival objective
- [ ] Evaluate with concordance index (C-index)
- [ ] Compare with fixed-horizon baseline

### Evaluation Metrics

- **Concordance Index (C-index)**: Measures ranking of predicted risks
- **Calibration plots**: Are predicted hazards well-calibrated?
- **Survival curves**: Plot $S_t$ for different risk groups

### Extensions

- **Attention mechanisms**: Learn which visits are most important
- **Multi-task learning**: Joint prediction of multiple outcomes
- **Counterfactual analysis**: "What if treatment was different?"

---

## Summary

**What We Learned:**

1. Discrete-time survival is natural for visit-based EHR data
2. Hazard is a conditional probability at each visit
3. Censoring is handled correctly in the likelihood
4. The loss function is derived from first principles
5. Implementation in PyTorch is straightforward

**Why This Matters:**

- Respects causality (no temporal leakage)
- Efficient use of data (multiple signals per patient)
- Flexible modeling of time-varying risk
- Clinically interpretable (hazard = "risk right now")

**Next:**

- Implement this in a notebook
- Apply to real EHR data (CKD progression, readmission, etc.)
- Compare with simpler baselines

---

**Previous:** [Part 1: Causal Progression Labels](causal-survival-analysis-1.md)
