# Custom Loss Functions for EHR Survival Analysis

**Date:** January 22, 2026  
**Source:** `src/ehrsequencing/models/losses.py`  
**Purpose:** Understanding and implementing custom loss functions for medical prediction tasks

---

## Table of Contents

1. [Why Custom Loss Functions?](#why-custom-loss-functions)
2. [Discrete-Time Survival Loss](#discrete-time-survival-loss)
3. [Additional Custom Loss Examples](#additional-custom-loss-examples)
4. [Loss Function Design Patterns](#loss-function-design-patterns)
5. [Implementation Best Practices](#implementation-best-practices)

---

## Why Custom Loss Functions?

Standard losses (cross-entropy, MSE) assume:
- Simple labels (0/1, or continuous values)
- All examples are fully observed
- Predictions are independent

**EHR data violates these assumptions:**

| Standard Assumption | EHR Reality | Solution |
|---------------------|-------------|----------|
| Fully observed labels | Censored data (unknown outcomes) | Survival likelihood |
| Fixed prediction time | Variable follow-up lengths | Time-varying predictions |
| Independent predictions | Temporal dependencies | Sequence-aware losses |
| Single task | Multiple outcomes (progression, death) | Multi-task losses |
| Balanced classes | Rare events (5% positive) | Weighted losses |

**Custom losses handle these complexities.**

---

## Discrete-Time Survival Loss

### Mathematical Foundation

**For patient $i$ with observed time $T_i$ and event indicator $\delta_i$:**

$$
\mathcal{L}_i = \left[\prod_{t=1}^{T_i-1} (1 - h_{it})\right] \cdot \left[h_{iT_i}\right]^{\delta_i}
$$

**Log-likelihood:**

$$
\log \mathcal{L}_i = \sum_{t=1}^{T_i-1} \log(1 - h_{it}) + \delta_i \log(h_{iT_i})
$$

**Loss (negative log-likelihood):**

$$
\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \log \mathcal{L}_i
$$

---

### PyTorch Implementation Walkthrough

```python
class DiscreteTimeSurvivalLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps  # Prevent log(0)
    
    def forward(
        self,
        hazards: torch.Tensor,        # [batch, max_visits]
        event_times: torch.Tensor,    # [batch]
        event_indicators: torch.Tensor,  # [batch]
        sequence_mask: torch.Tensor,  # [batch, max_visits]
    ) -> torch.Tensor:
        """
        Compute discrete-time survival loss.
        
        Args:
            hazards: Predicted hazards h_t ∈ (0,1) at each visit
            event_times: Index of event/censoring (0-indexed)
            event_indicators: 1 if event observed, 0 if censored
            sequence_mask: 1 for real visits, 0 for padding
        """
        batch_size, max_visits = hazards.shape
        
        # Step 1: Clamp hazards to avoid log(0)
        hazards = torch.clamp(hazards, min=self.eps, max=1 - self.eps)
        
        # Step 2: Create masks for "before event" and "at event"
        time_idx = torch.arange(max_visits, device=hazards.device).unsqueeze(0)
        event_times_expanded = event_times.unsqueeze(1)
        
        # Visits before event: t < T_i
        before_event_mask = (time_idx < event_times_expanded).float() * sequence_mask
        
        # Visit at event: t == T_i
        at_event_mask = (time_idx == event_times_expanded).float() * sequence_mask
        
        # Step 3: Survival contribution (all visits before event)
        survival_ll = torch.sum(
            torch.log(1 - hazards) * before_event_mask,
            dim=1
        )
        
        # Step 4: Event contribution (only if event observed)
        event_ll = torch.sum(
            torch.log(hazards) * at_event_mask,
            dim=1
        ) * event_indicators.float()
        
        # Step 5: Total log-likelihood
        log_likelihood = survival_ll + event_ll
        
        # Step 6: Return negative log-likelihood (to minimize)
        return -torch.mean(log_likelihood)
```

---

### Concrete Example

**Patient data:**
```python
# Patient 1: Event at visit 3
hazards_1 = [0.05, 0.10, 0.25, 0.00]  # 3 visits + padding
event_time_1 = 2  # 0-indexed (3rd visit)
event_indicator_1 = 1  # Event observed
mask_1 = [1, 1, 1, 0]

# Patient 2: Censored at visit 2
hazards_2 = [0.08, 0.15, 0.00, 0.00]  # 2 visits + padding
event_time_2 = 1  # 0-indexed (2nd visit)
event_indicator_2 = 0  # Censored
mask_2 = [1, 1, 0, 0]
```

**Loss computation:**

**Patient 1 (Event at visit 3):**
```python
# Before event: t < 2 (visits 1 and 2)
before_event_mask = [1, 1, 0, 0]
survival_ll = log(1 - 0.05) + log(1 - 0.10)
            = log(0.95) + log(0.90)
            = -0.051 + (-0.105)
            = -0.156

# At event: t == 2 (visit 3)
at_event_mask = [0, 0, 1, 0]
event_ll = log(0.25) * 1  # event_indicator = 1
         = -1.386

# Total
log_likelihood_1 = -0.156 + (-1.386) = -1.542
```

**Patient 2 (Censored at visit 2):**
```python
# Before event: t < 1 (visit 1 only)
before_event_mask = [1, 0, 0, 0]
survival_ll = log(1 - 0.08)
            = log(0.92)
            = -0.083

# At event: none (censored)
event_ll = log(0.15) * 0  # event_indicator = 0
         = 0

# Total
log_likelihood_2 = -0.083 + 0 = -0.083
```

**Batch loss:**
```python
loss = -mean([log_likelihood_1, log_likelihood_2])
     = -mean([-1.542, -0.083])
     = -(-0.813)
     = 0.813
```

---

### Key Implementation Details

#### 1. Masking Strategy

**Three levels of masking:**

```python
# Level 1: Padding visits
sequence_mask: [batch, num_visits]
# Ensures padding visits don't contribute

# Level 2: Before/at event
before_event_mask = (time_idx < event_times_expanded) * sequence_mask
at_event_mask = (time_idx == event_times_expanded) * sequence_mask
# Separates survival and event contributions

# Level 3: Event indicator
event_ll = event_ll * event_indicators
# Only add event hazard if event observed
```

**Example:**
```python
# Patient with event at visit 3
time_idx = [0, 1, 2, 3, 4]  # Visit indices
event_time = 2
sequence_mask = [1, 1, 1, 0, 0]  # 3 real visits

# Before event: visits 0, 1 (before visit 2)
before_event_mask = [1, 1, 0, 0, 0]

# At event: visit 2
at_event_mask = [0, 0, 1, 0, 0]
```

---

#### 2. Numerical Stability

**Problem:** `log(0)` and `log(1)` are undefined/problematic

**Solution:** Clamp hazards

```python
hazards = torch.clamp(hazards, min=1e-7, max=1 - 1e-7)

# Now safe:
log(1 - hazards)  # Always defined
log(hazards)      # Always defined
```

**Why 1e-7?**
```python
# Too large (e.g., 1e-3): affects valid probabilities
h = 0.999  # Valid hazard
h_clamped = min(h, 1 - 1e-3) = 0.999  # No change (good)

# Too small (e.g., 1e-15): numerical precision issues
log(1e-15) = -34.5  # Can cause underflow in exp()

# 1e-7: Good balance
log(1e-7) = -16.1  # Safe range
```

---

#### 3. Vectorized Computation

**Naive approach (slow):**
```python
# Loop over batch
loss = 0
for i in range(batch_size):
    T = event_times[i].item()
    for t in range(T):
        loss += log(1 - hazards[i, t])
    if event_indicators[i]:
        loss += log(hazards[i, T])
```

**Vectorized (fast):**
```python
# Create masks for all patients at once
before_event_mask = (time_idx < event_times_expanded).float()

# Sum all at once
survival_ll = (log(1 - hazards) * before_event_mask).sum(dim=1)
```

**Speedup:** ~100-1000x faster (depends on batch size and sequence length)

---

## Additional Custom Loss Examples

### 1. Calibrated Survival Loss

**Motivation:** Standard survival loss may produce poorly calibrated probabilities

```python
class CalibratedSurvivalLoss(nn.Module):
    """
    Survival loss with calibration penalty.
    """
    def __init__(self, calibration_weight: float = 0.1):
        super().__init__()
        self.survival_loss = DiscreteTimeSurvivalLoss()
        self.calibration_weight = calibration_weight
    
    def forward(self, hazards, event_times, event_indicators, sequence_mask):
        # Standard survival loss
        surv_loss = self.survival_loss(hazards, event_times, event_indicators, sequence_mask)
        
        # Calibration penalty: observed vs. predicted event rates
        # For visits before events, compare:
        # - Predicted: mean hazard
        # - Observed: empirical event rate
        
        # Create mask for visits before events
        time_idx = torch.arange(hazards.size(1), device=hazards.device).unsqueeze(0)
        event_times_exp = event_times.unsqueeze(1)
        before_mask = (time_idx < event_times_exp).float() * sequence_mask
        
        # Predicted event rate
        predicted_rate = (hazards * before_mask).sum() / before_mask.sum()
        
        # Observed event rate (fraction with events)
        observed_rate = event_indicators.float().mean()
        
        # Calibration penalty (squared error)
        calibration_penalty = (predicted_rate - observed_rate) ** 2
        
        # Combined loss
        total_loss = surv_loss + self.calibration_weight * calibration_penalty
        
        return total_loss
```

**Why calibration matters:**
```python
# Model might predict:
mean(hazards) = 0.05  # 5% average risk

# But in data:
actual_event_rate = 0.20  # 20% of patients have event

# Model is underestimating risk!
# Calibration penalty encourages better alignment
```

---

### 2. Ranking Loss for Survival

**Motivation:** Focus on correct ranking (earlier events = higher risk)

```python
class SurvivalRankingLoss(nn.Module):
    """
    Ranking-based survival loss (inspired by Cox partial likelihood).
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, hazards, event_times, event_indicators, sequence_mask):
        """
        Maximize probability that patient with earlier event has higher cumulative risk.
        
        Args:
            hazards: [batch, num_visits]
            event_times: [batch]
            event_indicators: [batch]
            sequence_mask: [batch, num_visits]
        """
        batch_size = hazards.size(0)
        
        # Compute cumulative risk scores (sum of hazards)
        cumulative_risk = (hazards * sequence_mask.float()).sum(dim=1)  # [batch]
        
        # For each patient with event
        loss = 0
        num_pairs = 0
        
        for i in range(batch_size):
            if event_indicators[i] == 0:
                continue  # Skip censored patients
            
            # Compare with all patients who survived longer
            for j in range(batch_size):
                if event_times[j] > event_times[i]:
                    # Patient i had event earlier → should have higher risk
                    # Loss: -log P(risk_i > risk_j)
                    
                    # Softmax for probability
                    diff = (cumulative_risk[i] - cumulative_risk[j]) / self.temperature
                    prob = torch.sigmoid(diff)
                    
                    loss += -torch.log(prob + 1e-7)
                    num_pairs += 1
        
        if num_pairs == 0:
            return torch.tensor(0.0, device=hazards.device)
        
        return loss / num_pairs
```

**Use case:** When correct ranking is more important than calibrated probabilities

---

### 3. Focal Loss for Imbalanced Events

**Motivation:** Rare events (e.g., 2% progression rate) get overwhelmed by common events

```python
class FocalSurvivalLoss(nn.Module):
    """
    Focal loss adaptation for survival analysis.
    
    Downweights easy examples, focuses on hard examples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Balance between event and non-event
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.survival_loss = DiscreteTimeSurvivalLoss()
    
    def forward(self, hazards, event_times, event_indicators, sequence_mask):
        batch_size, max_visits = hazards.shape
        
        # Standard log-likelihood
        hazards_clamped = torch.clamp(hazards, min=1e-7, max=1 - 1e-7)
        
        time_idx = torch.arange(max_visits, device=hazards.device).unsqueeze(0)
        event_times_exp = event_times.unsqueeze(1)
        
        before_mask = (time_idx < event_times_exp).float() * sequence_mask
        at_event_mask = (time_idx == event_times_exp).float() * sequence_mask
        
        # Compute focal weights
        # For survival (1 - h_t): weight by h_t^gamma (focus on high hazards)
        survival_weights = (hazards ** self.gamma) * before_mask
        
        # For events (h_t): weight by (1-h_t)^gamma (focus on low hazards = hard cases)
        event_weights = ((1 - hazards) ** self.gamma) * at_event_mask
        
        # Weighted log-likelihood
        survival_ll = torch.sum(
            torch.log(1 - hazards_clamped) * survival_weights,
            dim=1
        )
        
        event_ll = torch.sum(
            torch.log(hazards_clamped) * event_weights,
            dim=1
        ) * event_indicators.float()
        
        # Alpha balancing
        survival_ll = (1 - self.alpha) * survival_ll
        event_ll = self.alpha * event_ll
        
        log_likelihood = survival_ll + event_ll
        
        return -torch.mean(log_likelihood)
```

**Effect:**
```python
# Easy example: h_t = 0.01 (correctly low hazard)
# Weight: (1 - 0.01)^2 = 0.98 (high weight, but slightly down from 1.0)

# Hard example: h_t = 0.4 (should be lower for survival)
# Weight: (0.4)^2 = 0.16 (heavily downweighted)

# Model focuses on getting hard examples right
```

---

### 4. Multi-Task Survival Loss

**Motivation:** Joint prediction of multiple outcomes

```python
class MultiTaskSurvivalLoss(nn.Module):
    """
    Combine multiple survival tasks with learned weights.
    """
    def __init__(self, task_names: list[str], learnable_weights: bool = True):
        """
        Args:
            task_names: List of task names (e.g., ['progression', 'death', 'readmission'])
            learnable_weights: Whether to learn task weights during training
        """
        super().__init__()
        self.task_names = task_names
        
        if learnable_weights:
            # Learnable task weights (log scale for numerical stability)
            self.log_vars = nn.Parameter(torch.zeros(len(task_names)))
        else:
            # Fixed weights
            self.register_buffer('log_vars', torch.zeros(len(task_names)))
        
        self.learnable_weights = learnable_weights
    
    def forward(self, task_losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted sum of task losses.
        
        Args:
            task_losses: Dictionary mapping task name to loss tensor
        
        Returns:
            Total weighted loss
        """
        total_loss = 0
        
        for i, task_name in enumerate(self.task_names):
            if task_name not in task_losses:
                continue
            
            task_loss = task_losses[task_name]
            
            if self.learnable_weights:
                # Uncertainty-weighted multi-task loss
                # loss_i / (2 * sigma_i^2) + log(sigma_i)
                # Using log_var = log(sigma^2)
                precision = torch.exp(-self.log_vars[i])
                loss_term = precision * task_loss + self.log_vars[i]
            else:
                # Simple sum
                loss_term = task_loss
            
            total_loss += loss_term
        
        return total_loss
```

**Usage:**
```python
# Three tasks
model = MultiTaskSurvivalModel(
    tasks=['progression', 'death', 'readmission']
)
loss_fn = MultiTaskSurvivalLoss(task_names=['progression', 'death', 'readmission'])

# Forward pass
outputs = model(visit_sequences)
# outputs['progression']: [batch, visits]
# outputs['death']: [batch, visits]
# outputs['readmission']: [batch, visits]

# Compute losses
task_losses = {
    'progression': survival_loss(outputs['progression'], ...),
    'death': survival_loss(outputs['death'], ...),
    'readmission': survival_loss(outputs['readmission'], ...),
}

# Combined loss (with learned weighting)
total_loss = loss_fn(task_losses)
```

**Learned weights:**
```python
# After training, inspect learned weights
task_weights = torch.exp(-loss_fn.log_vars)
# e.g., [1.2, 0.8, 0.5]
# Progression: highest weight (most reliable signal)
# Death: medium weight
# Readmission: lower weight (noisier signal)
```

---

### 5. Time-Stratified Loss

**Motivation:** Different time periods may need different emphasis

```python
class TimeStratifiedSurvivalLoss(nn.Module):
    """
    Weight different time periods differently.
    """
    def __init__(self, time_weights: Optional[list[float]] = None):
        """
        Args:
            time_weights: Weight for each visit position
                         Example: [1.0, 1.0, 2.0, 2.0] emphasizes later visits
        """
        super().__init__()
        if time_weights is not None:
            self.register_buffer('time_weights', torch.tensor(time_weights))
        else:
            self.time_weights = None
    
    def forward(self, hazards, event_times, event_indicators, sequence_mask):
        batch_size, max_visits = hazards.shape
        
        # Standard survival loss computation
        hazards_clamped = torch.clamp(hazards, min=1e-7, max=1 - 1e-7)
        
        time_idx = torch.arange(max_visits, device=hazards.device).unsqueeze(0)
        event_times_exp = event_times.unsqueeze(1)
        
        before_mask = (time_idx < event_times_exp).float() * sequence_mask
        at_event_mask = (time_idx == event_times_exp).float() * sequence_mask
        
        # Apply time-dependent weights
        if self.time_weights is not None:
            weights = self.time_weights.unsqueeze(0)  # [1, max_visits]
            before_mask = before_mask * weights
            at_event_mask = at_event_mask * weights
        
        # Weighted log-likelihood
        survival_ll = torch.sum(torch.log(1 - hazards_clamped) * before_mask, dim=1)
        event_ll = torch.sum(torch.log(hazards_clamped) * at_event_mask, dim=1) * event_indicators.float()
        
        log_likelihood = survival_ll + event_ll
        return -torch.mean(log_likelihood)
```

**Use cases:**
```python
# Emphasize early detection (weight early visits more)
time_weights = [2.0, 1.5, 1.0, 1.0, 1.0, ...]

# Emphasize long-term outcomes (weight later visits more)
time_weights = [1.0, 1.0, 1.0, 1.5, 2.0, ...]

# Exponential decay (focus on near-term)
time_weights = [1.0, 0.9, 0.81, 0.73, 0.66, ...]  # 0.9^t
```

---

### 6. Contrastive Survival Loss

**Motivation:** Learn better representations by contrasting event vs. non-event trajectories

```python
class ContrastiveSurvivalLoss(nn.Module):
    """
    Combine survival loss with contrastive learning.
    """
    def __init__(self, contrastive_weight: float = 0.1, temperature: float = 0.5):
        super().__init__()
        self.survival_loss = DiscreteTimeSurvivalLoss()
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
    
    def forward(
        self,
        hazards: torch.Tensor,
        representations: torch.Tensor,  # LSTM hidden states
        event_times: torch.Tensor,
        event_indicators: torch.Tensor,
        sequence_mask: torch.Tensor
    ):
        """
        Args:
            hazards: [batch, num_visits]
            representations: [batch, num_visits, hidden_dim] - LSTM hidden states
            event_times, event_indicators, sequence_mask: standard
        """
        # Standard survival loss
        surv_loss = self.survival_loss(hazards, event_times, event_indicators, sequence_mask)
        
        # Contrastive loss: event patients should be similar to each other,
        # different from non-event patients
        
        # Get representations at event/censoring time
        batch_size = representations.size(0)
        final_reps = []
        for i in range(batch_size):
            T = event_times[i].long()
            final_reps.append(representations[i, T])
        final_reps = torch.stack(final_reps)  # [batch, hidden_dim]
        
        # Normalize for cosine similarity
        final_reps = F.normalize(final_reps, dim=-1)
        
        # Compute similarity matrix
        similarity = final_reps @ final_reps.T  # [batch, batch]
        
        # Contrastive loss: pull together event patients, push apart event vs. censored
        contrastive_loss = 0
        num_contrasts = 0
        
        for i in range(batch_size):
            if event_indicators[i] == 0:
                continue  # Only use event patients as anchors
            
            # Positive pairs: other event patients
            # Negative pairs: censored patients
            
            pos_similarity = []
            neg_similarity = []
            
            for j in range(batch_size):
                if i == j:
                    continue
                
                sim = similarity[i, j] / self.temperature
                
                if event_indicators[j] == 1:
                    pos_similarity.append(sim)
                else:
                    neg_similarity.append(sim)
            
            if pos_similarity:
                # InfoNCE-style loss
                pos_sim = torch.stack(pos_similarity)
                neg_sim = torch.stack(neg_similarity) if neg_similarity else torch.tensor([]).to(hazards.device)
                
                # Numerator: exp(positive similarities)
                numerator = torch.exp(pos_sim).sum()
                
                # Denominator: exp(all similarities)
                denominator = numerator + torch.exp(neg_sim).sum()
                
                contrastive_loss += -torch.log(numerator / (denominator + 1e-7))
                num_contrasts += 1
        
        if num_contrasts > 0:
            contrastive_loss = contrastive_loss / num_contrasts
        
        # Combined loss
        total_loss = surv_loss + self.contrastive_weight * contrastive_loss
        
        return total_loss
```

**Intuition:**
- Event patients should have similar trajectories
- Censored (non-event) patients should be different
- Improves representation quality

---

### 7. Regularized Hazard Smoothness Loss

**Motivation:** Hazards shouldn't jump erratically between visits

```python
class SmoothHazardLoss(nn.Module):
    """
    Penalize large changes in hazards between consecutive visits.
    """
    def __init__(self, smoothness_weight: float = 0.01):
        super().__init__()
        self.survival_loss = DiscreteTimeSurvivalLoss()
        self.smoothness_weight = smoothness_weight
    
    def forward(self, hazards, event_times, event_indicators, sequence_mask):
        # Standard survival loss
        surv_loss = self.survival_loss(hazards, event_times, event_indicators, sequence_mask)
        
        # Smoothness penalty: penalize large jumps
        # |h_{t+1} - h_t|^2
        hazard_diffs = hazards[:, 1:] - hazards[:, :-1]  # [batch, visits-1]
        
        # Only penalize real visits
        mask_diffs = sequence_mask[:, 1:] * sequence_mask[:, :-1]
        
        smoothness_penalty = (hazard_diffs ** 2 * mask_diffs).sum() / mask_diffs.sum().clamp(min=1)
        
        # Combined loss
        total_loss = surv_loss + self.smoothness_weight * smoothness_penalty
        
        return total_loss
```

**Effect:**
```python
# Without smoothness:
hazards = [0.05, 0.08, 0.45, 0.12, ...]  # Erratic (0.45 spike)
# Model might overfit to noisy signal

# With smoothness:
hazards = [0.05, 0.08, 0.12, 0.15, ...]  # Gradual increase
# Encourages monotonic or smooth progression
```

---

### 8. Uncertainty-Aware Loss (Bayesian)

**Motivation:** Quantify prediction uncertainty

```python
class BayesianSurvivalLoss(nn.Module):
    """
    Bayesian loss with uncertainty estimation.
    
    Model predicts both hazard mean and variance.
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        hazard_means: torch.Tensor,     # [batch, visits]
        hazard_log_vars: torch.Tensor,  # [batch, visits] - log(σ²)
        event_times: torch.Tensor,
        event_indicators: torch.Tensor,
        sequence_mask: torch.Tensor
    ):
        """
        Args:
            hazard_means: Predicted mean hazards
            hazard_log_vars: Predicted log-variance (uncertainty)
        """
        # Negative log-likelihood with Gaussian uncertainty
        # p(event | h) = N(event | h_mean, σ²)
        
        batch_size, max_visits = hazard_means.shape
        
        # Variance (from log-variance)
        variance = torch.exp(hazard_log_vars)
        
        # Standard masks
        time_idx = torch.arange(max_visits, device=hazard_means.device).unsqueeze(0)
        event_times_exp = event_times.unsqueeze(1)
        
        before_mask = (time_idx < event_times_exp).float() * sequence_mask
        at_event_mask = (time_idx == event_times_exp).float() * sequence_mask
        
        # Survival log-likelihood with uncertainty
        # log N(0 | h_mean, σ²) where 0 = no event
        survival_ll = torch.sum(
            -0.5 * (hazard_means ** 2) / (variance + 1e-7) 
            - 0.5 * torch.log(variance + 1e-7)
            - 0.5 * torch.log(torch.tensor(2 * 3.14159))
        ) * before_mask
        
        # Event log-likelihood
        # log N(1 | h_mean, σ²) where 1 = event
        event_ll = torch.sum(
            -0.5 * ((1 - hazard_means) ** 2) / (variance + 1e-7)
            - 0.5 * torch.log(variance + 1e-7)
            - 0.5 * torch.log(torch.tensor(2 * 3.14159))
        ) * at_event_mask * event_indicators.float()
        
        log_likelihood = survival_ll + event_ll
        
        return -torch.mean(log_likelihood)
```

**Model modification:**
```python
# Hazard head outputs both mean and variance
self.hazard_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 2),  # Output: [mean, log_var]
)

# Forward pass
hazard_params = self.hazard_head(lstm_out)
hazard_means = torch.sigmoid(hazard_params[..., 0])  # Mean in (0,1)
hazard_log_vars = hazard_params[..., 1]  # Log-variance (unbounded)
```

**Benefit:** Uncertainty estimates for clinical decision support
```python
# Prediction with uncertainty
h_mean = 0.25
h_std = 0.05

# 95% confidence interval
h_lower = h_mean - 1.96 * h_std = 0.15
h_upper = h_mean + 1.96 * h_std = 0.35

# Risk between 15% and 35% (uncertain)
```

---

### 9. Survival Loss with Auxiliary Tasks

**Motivation:** Improve representations via related auxiliary tasks

```python
class SurvivalWithAuxiliaryLoss(nn.Module):
    """
    Primary survival loss + auxiliary prediction tasks.
    """
    def __init__(
        self,
        survival_weight: float = 1.0,
        next_visit_weight: float = 0.3,
        code_prediction_weight: float = 0.2
    ):
        super().__init__()
        self.survival_loss = DiscreteTimeSurvivalLoss()
        self.survival_weight = survival_weight
        self.next_visit_weight = next_visit_weight
        self.code_prediction_weight = code_prediction_weight
    
    def forward(
        self,
        hazards: torch.Tensor,
        next_visit_predictions: Optional[torch.Tensor],
        code_predictions: Optional[torch.Tensor],
        event_times: torch.Tensor,
        event_indicators: torch.Tensor,
        sequence_mask: torch.Tensor,
        next_visit_labels: Optional[torch.Tensor] = None,
        code_labels: Optional[torch.Tensor] = None
    ):
        """
        Args:
            hazards: Primary survival predictions [batch, visits]
            next_visit_predictions: Predicted next visit embedding [batch, visits-1, dim]
            code_predictions: Predicted codes for next visit [batch, visits-1, vocab_size]
            (labels for auxiliary tasks)
        """
        # Primary: survival loss
        survival_loss = self.survival_loss(
            hazards, event_times, event_indicators, sequence_mask
        )
        
        total_loss = self.survival_weight * survival_loss
        
        # Auxiliary 1: Predict next visit embedding (regression)
        if next_visit_predictions is not None and next_visit_labels is not None:
            next_visit_loss = F.mse_loss(
                next_visit_predictions,
                next_visit_labels,
                reduction='mean'
            )
            total_loss += self.next_visit_weight * next_visit_loss
        
        # Auxiliary 2: Predict codes in next visit (multi-label)
        if code_predictions is not None and code_labels is not None:
            code_loss = F.binary_cross_entropy_with_logits(
                code_predictions,
                code_labels,
                reduction='mean'
            )
            total_loss += self.code_prediction_weight * code_loss
        
        return total_loss
```

**Benefit:** Auxiliary tasks improve LSTM representations
- Next visit prediction: Forces model to learn trajectory patterns
- Code prediction: Learns code co-occurrence patterns
- Both help survival prediction (shared representations)

---

### 10. Sample-Weighted Survival Loss

**Motivation:** Some patients more important (e.g., high-risk population)

```python
class WeightedSurvivalLoss(nn.Module):
    """
    Weight individual patients differently.
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        hazards: torch.Tensor,
        event_times: torch.Tensor,
        event_indicators: torch.Tensor,
        sequence_mask: torch.Tensor,
        sample_weights: torch.Tensor  # [batch] - weight per patient
    ):
        """
        Args:
            sample_weights: Per-patient weights (e.g., inverse propensity scores)
        """
        batch_size, max_visits = hazards.shape
        
        # Standard computation
        hazards_clamped = torch.clamp(hazards, min=1e-7, max=1 - 1e-7)
        
        time_idx = torch.arange(max_visits, device=hazards.device).unsqueeze(0)
        event_times_exp = event_times.unsqueeze(1)
        
        before_mask = (time_idx < event_times_exp).float() * sequence_mask
        at_event_mask = (time_idx == event_times_exp).float() * sequence_mask
        
        survival_ll = torch.sum(torch.log(1 - hazards_clamped) * before_mask, dim=1)
        event_ll = torch.sum(torch.log(hazards_clamped) * at_event_mask, dim=1) * event_indicators.float()
        
        log_likelihood = survival_ll + event_ll
        
        # Weight by sample importance
        weighted_ll = log_likelihood * sample_weights
        
        return -torch.mean(weighted_ll)
```

**Use cases:**
```python
# Inverse propensity weighting (for causal inference)
sample_weights = 1 / propensity_scores

# Importance sampling (focus on rare subgroups)
sample_weights[rare_disease_patients] = 2.0
sample_weights[common_disease_patients] = 1.0

# Cost-sensitive learning
sample_weights[high_cost_patients] = 3.0  # False negatives expensive
```

---

## Loss Function Design Patterns

### Pattern 1: Base Loss + Regularization

```python
class CustomLoss(nn.Module):
    def forward(self, predictions, targets, ...):
        # Primary loss
        primary_loss = base_loss(predictions, targets)
        
        # Regularization
        regularization = compute_regularization(predictions)
        
        # Combined
        return primary_loss + lambda_reg * regularization
```

**Examples:**
- Smoothness regularization
- Entropy regularization (encourage confidence)
- KL divergence (match target distribution)

---

### Pattern 2: Multi-Component Likelihood

```python
class CustomLoss(nn.Module):
    def forward(self, predictions, targets, ...):
        # Multiple likelihood components
        ll_component1 = compute_ll_1(predictions, targets)
        ll_component2 = compute_ll_2(predictions, targets)
        
        # Combined log-likelihood
        total_ll = ll_component1 + ll_component2
        
        return -total_ll  # Negative for minimization
```

**Examples:**
- Survival + auxiliary tasks
- Multiple event types
- Hierarchical models (within-visit + across-visit)

---

### Pattern 3: Learned Loss Weights

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable weights
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, task_losses):
        total_loss = 0
        for i, loss in enumerate(task_losses):
            # Uncertainty weighting
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss
```

**Benefit:** Automatically balance task importance during training

---

### Pattern 4: Metric-Driven Losses

```python
class CustomLoss(nn.Module):
    def forward(self, predictions, targets, ...):
        # Directly optimize evaluation metric
        # E.g., maximize C-index, AUC, F1
        
        metric = compute_differentiable_metric(predictions, targets)
        
        return -metric  # Negative to maximize
```

**Examples:**
- Differentiable C-index
- Soft F1 loss
- Differentiable ranking metrics

---

## Implementation Best Practices

### 1. Numerical Stability

```python
# Always clamp probabilities
hazards = torch.clamp(hazards, min=1e-7, max=1 - 1e-7)

# Use log-sum-exp trick for numerical stability
log_survival = torch.cumsum(torch.log(1 - hazards + 1e-7), dim=1)
survival_probs = torch.exp(log_survival)
# Instead of: survival = torch.cumprod(1 - hazards, dim=1)
```

---

### 2. Masking

```python
# Always handle padding correctly
# Multiply losses by mask before summing
masked_loss = loss * sequence_mask
total_loss = masked_loss.sum() / sequence_mask.sum()
```

---

### 3. Gradient Clipping

```python
# For complex losses, clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 4. Validation

```python
# Test loss function on simple examples
def test_survival_loss():
    # Example 1: Event at time 2
    hazards = torch.tensor([[0.1, 0.2, 0.3]])
    event_times = torch.tensor([2])
    event_indicators = torch.tensor([1])
    mask = torch.ones(1, 3)
    
    loss = loss_fn(hazards, event_times, event_indicators, mask)
    
    # Expected: -[log(0.9) + log(0.8) + log(0.3)]
    expected = -(torch.log(torch.tensor(0.9)) + 
                 torch.log(torch.tensor(0.8)) + 
                 torch.log(torch.tensor(0.3)))
    
    assert torch.allclose(loss, expected, atol=1e-5)
```

---

### 5. Reduction Strategies

```python
# Mean reduction (default)
loss = -log_likelihood.mean()

# Sum reduction (for comparison with other implementations)
loss = -log_likelihood.sum()

# Weighted mean (sample importance)
loss = -(log_likelihood * weights).sum() / weights.sum()
```

---

## Summary

### Key Custom Losses for Survival Analysis

| Loss | Purpose | When to Use |
|------|---------|-------------|
| **DiscreteTimeSurvival** | Standard survival likelihood | Default for time-to-event |
| **Calibrated** | Well-calibrated probabilities | Clinical decision support |
| **Ranking** | Correct risk ordering | Risk stratification |
| **Focal** | Handle rare events | Imbalanced data |
| **Multi-task** | Multiple outcomes | Competing risks |
| **Smooth** | Gradual hazard changes | Noisy data |
| **Contrastive** | Better representations | Small datasets |
| **Bayesian** | Uncertainty quantification | High-stakes decisions |

---

### Design Principles

1. **Start with standard survival loss** (works well for most cases)
2. **Add regularization** if overfitting (smoothness, uncertainty)
3. **Use multi-task** if multiple outcomes available
4. **Consider focal loss** for rare events (< 5% positive rate)
5. **Validate numerically** on simple test cases
6. **Monitor calibration** (not just discrimination metrics)

---

### Implementation Checklist

When implementing custom loss:

- ✅ Handle censoring correctly (if applicable)
- ✅ Clamp probabilities to avoid log(0)
- ✅ Mask padding visits/patients
- ✅ Use vectorized operations (avoid loops)
- ✅ Test on simple examples
- ✅ Check gradient flow (no NaN/inf)
- ✅ Document assumptions and limitations
- ✅ Compare with standard baseline

Custom loss functions are powerful tools for adapting models to the unique challenges of EHR data, particularly for handling censoring, temporal dependencies, and rare events!
