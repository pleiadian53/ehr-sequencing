# Discrete-Time Survival Analysis Documentation

**Date:** January 22, 2026  
**Purpose:** Overview of documentation for discrete-time survival modeling in EHR sequences

---

## Overview

This directory contains comprehensive documentation for implementing discrete-time survival analysis models for EHR data, covering architecture, loss functions, and best practices.

---

## Documents

### 1. [Survival LSTM Architecture](survival_lstm_architecture.md)

**What it covers:**
- Complete walkthrough of `DiscreteTimeSurvivalLSTM` architecture
- Component-by-component breakdown (embedding → aggregation → LSTM → hazard head)
- Comparison with standard `LSTMBaseline` model
- Integration with `VisitEncoder` for attention-based aggregation
- Extensions: time-aware, competing risks, hierarchical models

**Key insights:**
- Why hazard predictions at every visit (not just final prediction)
- How survival LSTM differs from classification LSTM
- Benefits of adding attention-based visit encoding
- When to use different architectural variants

**Read this first** to understand the model architecture.

---

### 2. [Custom Loss Functions](custom_loss_functions.md)

**What it covers:**
- Detailed walkthrough of `DiscreteTimeSurvivalLoss` implementation
- Mathematical foundation and PyTorch implementation
- Concrete numerical examples
- 10+ custom loss function examples:
  - Calibrated survival loss
  - Ranking loss
  - Focal loss for imbalanced data
  - Multi-task loss
  - Time-stratified loss
  - Contrastive loss
  - Smooth hazard loss
  - Bayesian uncertainty loss
  - Auxiliary task loss
  - Sample-weighted loss

**Key insights:**
- Why standard losses (cross-entropy) don't work for survival data
- How to handle censoring correctly
- Masking strategies for variable-length sequences
- Numerical stability techniques
- Loss function design patterns

**Read this** to understand survival loss computation and explore advanced loss designs.

---

## Quick Start

### Basic Usage

```python
from ehrsequencing.models.survival_lstm import DiscreteTimeSurvivalLSTM
from ehrsequencing.models.losses import DiscreteTimeSurvivalLoss

# Initialize model
model = DiscreteTimeSurvivalLSTM(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.1
)

# Initialize loss
loss_fn = DiscreteTimeSurvivalLoss()

# Training
for batch in dataloader:
    # Forward pass
    hazards = model(
        batch['visit_codes'],
        batch['visit_mask'],
        batch['sequence_mask']
    )
    
    # Compute loss
    loss = loss_fn(
        hazards,
        batch['event_times'],
        batch['event_indicators'],
        batch['sequence_mask']
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    hazards = model(visit_codes, visit_mask, sequence_mask)
    survival_probs, cumulative_hazards = model.predict_survival(
        visit_codes, visit_mask, sequence_mask
    )
```

---

### With Attention-Based Visit Encoding

```python
# Import VisitEncoder
from ehrsequencing.models.lstm_baseline import VisitEncoder

class ImprovedSurvivalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Use attention for visit encoding
        self.visit_encoder = VisitEncoder(
            embedding_dim=embedding_dim,
            aggregation='attention',  # Learn code importance
            dropout=0.1
        )
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, visit_codes, visit_mask, sequence_mask):
        # Embed codes
        embeddings = self.embedding(visit_codes)
        batch_size, num_visits, max_codes, embed_dim = embeddings.shape
        
        # Flatten for visit encoder
        embeddings_flat = embeddings.view(batch_size * num_visits, max_codes, embed_dim)
        mask_flat = visit_mask.view(batch_size * num_visits, max_codes)
        
        # Attention-based aggregation
        visit_vectors = self.visit_encoder(embeddings_flat, mask_flat)
        visit_vectors = visit_vectors.view(batch_size, num_visits, embed_dim)
        
        # LSTM and hazard prediction
        lstm_out, _ = self.lstm(visit_vectors)
        hazards = self.hazard_head(lstm_out).squeeze(-1)
        
        return hazards * sequence_mask.float()
```

---

## Key Concepts

### 1. Hazard vs. Survival

**Hazard $h_t$:**
- Conditional probability of event at time $t$
- Given survival up to time $t$
- Local measure (specific to time $t$)

**Survival $S_t$:**
- Probability of no event through time $t$
- Cumulative measure
- Related: $S_t = \prod_{k=1}^{t} (1 - h_k)$

---

### 2. Censoring

**Definition:** Follow-up ends before event observed

**Types:**
- **Right censoring:** Don't know what happens after last visit (most common)
- **Left censoring:** Don't know if event occurred before first visit (rare in EHR)
- **Interval censoring:** Event occurred between two visits (can handle with discrete-time)

**Handling:** Include in likelihood as survival contribution only (no event term)

---

### 3. Event Indicators

```python
# Event indicator
delta_i = 1  # Event observed at time T_i
delta_i = 0  # Censored at time T_i (unknown beyond)

# In loss function:
event_contribution = log(h_T) * delta_i
# If delta_i = 0: no event contribution (censored)
# If delta_i = 1: add log(h_T) (event occurred)
```

---

## Common Pitfalls

### Pitfall 1: Treating Censoring as Negative

❌ **Wrong:**
```python
# Censored patient labeled as "no event"
labels[censored_patients] = 0
loss = BCELoss(predictions, labels)
```

✅ **Correct:**
```python
# Use survival loss that handles censoring
loss = DiscreteTimeSurvivalLoss(
    hazards, event_times, event_indicators, mask
)
```

---

### Pitfall 2: Not Clamping Probabilities

❌ **Wrong:**
```python
log_prob = torch.log(hazards)  # Might be log(0) = -inf
```

✅ **Correct:**
```python
hazards = torch.clamp(hazards, min=1e-7, max=1-1e-7)
log_prob = torch.log(hazards)  # Always defined
```

---

### Pitfall 3: Ignoring Padding

❌ **Wrong:**
```python
loss = torch.log(1 - hazards).sum()  # Includes padding!
```

✅ **Correct:**
```python
loss = (torch.log(1 - hazards) * sequence_mask).sum()
```

---

### Pitfall 4: Wrong Mask Creation

❌ **Wrong:**
```python
# Event at visit 2 (0-indexed)
before_mask = time_idx <= event_time  # [1, 1, 1, 0, ...]  Wrong!
```

✅ **Correct:**
```python
before_mask = time_idx < event_time  # [1, 1, 0, 0, ...]  Correct!
```

---

## Evaluation Metrics

### 1. Concordance Index (C-index)

```python
def concordance_index(hazards, event_times, event_indicators):
    """
    Measures ranking quality: do higher risks predict earlier events?
    
    C-index = P(risk_i > risk_j | time_i < time_j, both events)
    
    Range: [0, 1]
    - 0.5: Random
    - > 0.7: Good
    - > 0.8: Excellent
    """
    # Compute cumulative risks
    cumulative_risk = hazards.sum(dim=1)
    
    # Count concordant pairs
    concordant = 0
    total = 0
    
    for i in range(len(event_times)):
        if event_indicators[i] == 0:
            continue
        
        for j in range(len(event_times)):
            if i == j:
                continue
            
            if event_times[j] > event_times[i]:
                total += 1
                if cumulative_risk[i] > cumulative_risk[j]:
                    concordant += 1
    
    return concordant / total if total > 0 else 0.5
```

---

### 2. Integrated Brier Score

```python
def integrated_brier_score(survival_probs, event_times, event_indicators, times):
    """
    Measures calibration: are predicted probabilities accurate?
    
    Lower is better. Range: [0, 1]
    """
    brier_scores = []
    
    for t in times:
        # At time t, predicted: S_t
        # Observed: 1 if no event by t, 0 if event by t
        observed = (event_times > t).float()
        predicted = survival_probs[:, t]
        
        # Brier score at time t
        brier_t = ((observed - predicted) ** 2).mean()
        brier_scores.append(brier_t)
    
    # Integrate over time
    return torch.mean(torch.stack(brier_scores))
```

---

### 3. Time-Dependent AUC

```python
def time_dependent_auc(hazards, event_times, event_indicators, time_point):
    """
    AUC for predicting events by specific time point.
    
    Args:
        time_point: Prediction horizon (e.g., predict events by visit 5)
    """
    # Binary outcome: event by time_point?
    labels = (event_times <= time_point).float() * event_indicators
    
    # Predicted risk: cumulative hazard up to time_point
    cumulative_risk = hazards[:, :time_point+1].sum(dim=1)
    
    # Compute AUC
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels.cpu().numpy(), cumulative_risk.cpu().numpy())
```

---

## Further Reading

### Related Documentation

1. **Theory:** `docs/methods/causal-survival-analysis-2.md`
   - Mathematical foundations
   - Discrete-time survival theory
   - Likelihood derivations

2. **Model Architecture:** `survival_lstm_architecture.md` (this directory)
   - Implementation details
   - Architectural variants
   - Integration with VisitEncoder

3. **LSTM Deep Dive:** `dev/methods/lstm_deep_dive.md`
   - LSTM mechanics
   - Memory representation
   - Gradient flow

---

### External Resources

**Classic texts:**
- Singer & Willett (2003). *Applied Longitudinal Data Analysis*
- Tutz & Schmid (2016). *Modeling Discrete Time-to-Event Data*

**Recent papers:**
- Katzman et al. (2018). "DeepSurv: Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network"
- Lee et al. (2018). "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks"
- Kvamme et al. (2019). "Time-to-Event Prediction with Neural Networks and Cox Regression"

**Python libraries:**
- `lifelines`: Survival analysis (Cox, Kaplan-Meier, etc.)
- `scikit-survival`: ML for survival analysis
- `pycox`: Deep learning for survival analysis

---

## Summary

This documentation provides:

1. **Complete architecture walkthrough** of survival LSTM models
2. **Comprehensive loss function library** with 10+ examples
3. **Practical implementation guidance** with numerical examples
4. **Best practices** for numerical stability and validation
5. **Extensions** for real-world clinical applications

The survival LSTM + custom loss framework enables principled handling of censored EHR data while maintaining causal prediction properties!
