# Discrete-Time Survival Analysis for EHR Sequences

**A comprehensive guide to survival modeling with LSTM and BEHRT**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Discrete-Time Survival Framework](#discrete-time-survival-framework)
3. [Label Preparation](#label-preparation)
4. [Loss Function and Optimization](#loss-function-and-optimization)
5. [Model Architectures](#model-architectures)
6. [BEHRT vs LSTM Comparison](#behrt-vs-lstm-comparison)
7. [Implementation Details](#implementation-details)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Best Practices](#best-practices)

---

## Introduction

Survival analysis models the time until an event occurs, accounting for censored observations (patients who haven't experienced the event by the end of observation). In EHR data, we predict outcomes like:

- **Hospital readmission** (30-day)
- **Mortality** (in-hospital, 30-day, 1-year)
- **Disease onset** (diabetes, heart failure, stroke)
- **Treatment response** (time to efficacy or failure)

### Why Discrete-Time?

EHR data naturally arrives in discrete time intervals (visits), making discrete-time survival analysis a natural fit:

- **Visits are discrete events** - Patients are seen at specific time points
- **Interval censoring** - Events occur between visits
- **Computational efficiency** - Simpler than continuous-time models
- **Flexible hazard modeling** - Can capture complex time-varying patterns

---

## Discrete-Time Survival Framework

### Hazard Function

At each visit $t$, the model predicts a **hazard** $h_t$:

$$h_t = P(T = t \mid T \geq t, X_t)$$

where:
- $T$ is the event time (visit index when event occurs)
- $X_t$ is the patient history up to visit $t$
- $h_t \in [0, 1]$ is the probability of event at visit $t$, given survival to $t$

**Interpretation:** If $h_t = 0.2$, there's a 20% chance the event occurs at visit $t$, given the patient survived to visit $t$.

### Survival Function

The **survival probability** at visit $t$ is:

$$S(t) = P(T > t) = \prod_{i=1}^{t} (1 - h_i)$$

This is the probability of surviving (not experiencing the event) beyond visit $t$.

**Example:**
```
Visit 1: h₁ = 0.1  →  S(1) = 0.9
Visit 2: h₂ = 0.15 →  S(2) = 0.9 × 0.85 = 0.765
Visit 3: h₃ = 0.2  →  S(3) = 0.765 × 0.8 = 0.612
```

### Cumulative Incidence

The probability of event by visit $t$:

$$F(t) = 1 - S(t) = 1 - \prod_{i=1}^{t} (1 - h_i)$$

---

## Label Preparation

### Dataset Structure

For each patient, we need:

1. **Visit sequence** - Medical codes at each visit
2. **Event time** - Visit index when event occurred (or last observed visit)
3. **Event indicator** - Whether event occurred (1) or censored (0)

### Example: 30-Day Readmission

**Patient A (Event Occurred):**
```python
{
    'patient_id': 'A001',
    'visits': [
        {'date': '2020-01-01', 'codes': [250, 401, 500], 'age': 65},  # Visit 0: Admission
        {'date': '2020-01-05', 'codes': [500, 501], 'age': 65},       # Visit 1: Discharge
        {'date': '2020-01-20', 'codes': [250, 428], 'age': 65},       # Visit 2: Readmission (event!)
    ],
    'event_time': 2,           # Event at visit 2
    'event_indicator': 1,      # Event occurred
    'censoring_time': None     # Not censored
}
```

**Patient B (Censored):**
```python
{
    'patient_id': 'B002',
    'visits': [
        {'date': '2020-02-01', 'codes': [493, 500], 'age': 45},  # Visit 0: Admission
        {'date': '2020-02-03', 'codes': [500, 501], 'age': 45},  # Visit 1: Discharge
        {'date': '2020-03-15', 'codes': [500], 'age': 45},       # Visit 2: Follow-up (no readmission)
    ],
    'event_time': 3,           # Last observed visit
    'event_indicator': 0,      # No event (censored)
    'censoring_time': 3        # Censored at visit 3
}
```

### Label Generation Process

**Step 1: Define Event**

For readmission:
```python
def is_readmission(visit, previous_visits):
    """Check if visit is a readmission within 30 days"""
    if len(previous_visits) == 0:
        return False
    
    last_discharge = previous_visits[-1]['date']
    current_admission = visit['date']
    days_since_discharge = (current_admission - last_discharge).days
    
    return days_since_discharge <= 30 and visit['type'] == 'admission'
```

**Step 2: Identify Event Time**

```python
def get_event_time(visits):
    """Find first visit where event occurs"""
    for t, visit in enumerate(visits):
        if is_readmission(visit, visits[:t]):
            return t, 1  # (event_time, event_indicator)
    
    # No event found - censored
    return len(visits), 0
```

**Step 3: Create Labels Tensor**

For a batch of patients:

```python
# Shape: (batch_size,)
event_times = torch.tensor([2, 3, 1, 5, 4])      # Visit index of event/censoring
event_indicators = torch.tensor([1, 0, 1, 0, 1])  # 1=event, 0=censored
```

### Synthetic Label Generation

For development and testing, we generate synthetic outcomes with realistic risk-time correlation:

```python
from ehrsequencing.synthetic.survival import DiscreteTimeSurvivalGenerator

generator = DiscreteTimeSurvivalGenerator(
    risk_correlation=-0.5,  # Higher risk → earlier events
    censoring_rate=0.3,     # 30% of patients censored
    time_scale=10           # Average event time
)

# Generate outcomes for patient sequences
outcomes = generator.generate_outcomes(patient_sequences)
```

**Risk Score Computation:**
```python
def compute_risk_score(patient_sequence):
    """Compute risk based on patient characteristics"""
    # Risk factors
    num_comorbidities = count_unique_diagnoses(patient_sequence)
    visit_frequency = len(patient_sequence) / time_span
    code_diversity = len(set(flatten(patient_sequence)))
    
    # Weighted combination
    risk = (
        0.4 * num_comorbidities +
        0.3 * visit_frequency +
        0.3 * code_diversity
    )
    
    return risk
```

---

## Loss Function and Optimization

### Negative Log-Likelihood Loss

The discrete-time survival loss is derived from maximum likelihood estimation:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \delta_i \log h_{t_i} + (1-\delta_i) \log S(t_i) \right]$$

where:
- $N$ is the number of patients
- $\delta_i$ is the event indicator (1 if event, 0 if censored)
- $t_i$ is the event/censoring time
- $h_{t_i}$ is the predicted hazard at time $t_i$
- $S(t_i) = \prod_{j=1}^{t_i} (1 - h_j)$ is the survival probability

### Loss Components

**For patients with events ($\delta_i = 1$):**

$$\mathcal{L}_{\text{event}} = -\log h_{t_i}$$

This maximizes the hazard at the observed event time.

**For censored patients ($\delta_i = 0$):**

$$\mathcal{L}_{\text{censored}} = -\log S(t_i) = -\sum_{j=1}^{t_i} \log(1 - h_j)$$

This maximizes the survival probability up to the censoring time.

### Implementation

```python
class DiscreteTimeSurvivalLoss(nn.Module):
    """Negative log-likelihood for discrete-time survival"""
    
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, hazards, event_times, event_indicators):
        """
        Args:
            hazards: (batch_size, max_visits) - Predicted hazards at each visit
            event_times: (batch_size,) - Visit index of event/censoring
            event_indicators: (batch_size,) - 1 if event, 0 if censored
        
        Returns:
            loss: Scalar tensor
        """
        batch_size, max_visits = hazards.shape
        
        # Clamp hazards for numerical stability
        hazards = torch.clamp(hazards, self.epsilon, 1 - self.epsilon)
        
        # Create mask for visits up to event/censoring time
        visit_indices = torch.arange(max_visits, device=hazards.device)
        mask = visit_indices.unsqueeze(0) < event_times.unsqueeze(1)  # (batch, max_visits)
        
        # Compute survival probability: S(t) = ∏(1 - h_j) for j ≤ t
        log_survival = torch.log(1 - hazards)  # (batch, max_visits)
        log_survival_masked = log_survival * mask.float()
        log_S = log_survival_masked.sum(dim=1)  # (batch,)
        
        # Get hazard at event time
        event_hazards = hazards[torch.arange(batch_size), event_times]
        log_h = torch.log(event_hazards)  # (batch,)
        
        # Combine event and censored losses
        # For events: -log(h_t)
        # For censored: -log(S(t))
        loss_event = -log_h * event_indicators
        loss_censored = -log_S * (1 - event_indicators)
        
        loss = (loss_event + loss_censored).mean()
        
        return loss
```

### Optimization Objective

**Goal:** Minimize the negative log-likelihood

$$\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)$$

where $\theta$ represents model parameters (embeddings, LSTM/BEHRT weights, prediction head).

**Optimizer:** Adam with learning rate scheduling

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

**Training Loop:**

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # Forward pass
        hazards = model(
            codes=batch['codes'],
            ages=batch['ages'],
            visit_mask=batch['visit_mask']
        )
        
        # Compute loss
        loss = loss_fn(
            hazards=hazards,
            event_times=batch['event_times'],
            event_indicators=batch['event_indicators']
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Validation
    val_loss, c_index = evaluate(model, val_loader)
    scheduler.step(val_loss)
```

---

## Model Architectures

### LSTM Baseline

**Architecture:**
```
Input Codes → Embedding Layer → LSTM → Hazard Prediction Head
```

**Components:**

1. **Embedding Layer:** Maps medical codes to dense vectors
   ```python
   self.embedding = nn.Embedding(vocab_size, embedding_dim)
   ```

2. **LSTM Encoder:** Processes visit sequence
   ```python
   self.lstm = nn.LSTM(
       input_size=embedding_dim,
       hidden_size=hidden_dim,
       num_layers=num_layers,
       dropout=dropout,
       batch_first=True
   )
   ```

3. **Hazard Prediction Head:** Outputs hazard per visit
   ```python
   self.hazard_head = nn.Sequential(
       nn.Linear(hidden_dim, hidden_dim // 2),
       nn.ReLU(),
       nn.Dropout(dropout),
       nn.Linear(hidden_dim // 2, 1),
       nn.Sigmoid()  # Hazard ∈ [0, 1]
   )
   ```

**Forward Pass:**

```python
def forward(self, codes, visit_mask, sequence_mask):
    """
    Args:
        codes: (batch, max_visits, max_codes_per_visit)
        visit_mask: (batch, max_visits) - 1 for real visits, 0 for padding
        sequence_mask: (batch, max_visits, max_codes_per_visit) - 1 for real codes
    
    Returns:
        hazards: (batch, max_visits) - Hazard at each visit
    """
    batch_size, max_visits, max_codes = codes.shape
    
    # Embed codes
    embedded = self.embedding(codes)  # (batch, max_visits, max_codes, emb_dim)
    
    # Average codes within each visit
    visit_embeddings = (embedded * sequence_mask.unsqueeze(-1)).sum(dim=2)
    visit_embeddings = visit_embeddings / (sequence_mask.sum(dim=2, keepdim=True) + 1e-7)
    # Shape: (batch, max_visits, emb_dim)
    
    # LSTM encoding
    lstm_out, _ = self.lstm(visit_embeddings)  # (batch, max_visits, hidden_dim)
    
    # Predict hazard at each visit
    hazards = self.hazard_head(lstm_out).squeeze(-1)  # (batch, max_visits)
    
    # Mask padding visits
    hazards = hazards * visit_mask
    
    return hazards
```

### BEHRT for Survival

**Architecture:**
```
Input Codes → BEHRT Encoder (pre-trained) → Visit Aggregation → Hazard Prediction Head
```

**Key Differences from LSTM:**

1. **Pre-trained Encoder:** BEHRT is pre-trained with MLM on large unlabeled data
2. **Bidirectional Context:** Self-attention captures relationships in both directions
3. **EHR-Specific Embeddings:** Age, visit, segment, and position embeddings
4. **Visit-Level Aggregation:** Pool code-level representations to visit-level

**Components:**

1. **BEHRT Encoder (Pre-trained):**
   ```python
   from ehrsequencing.models.behrt import BEHRT, BEHRTForMLM
   
   # Load pre-trained BEHRT
   pretrained = BEHRTForMLM.from_pretrained('checkpoints/behrt_mlm/')
   self.behrt = pretrained.behrt  # Extract encoder
   ```

2. **Visit Aggregation:** Pool code representations within each visit
   ```python
   def aggregate_visits(self, code_embeddings, visit_ids):
       """
       Args:
           code_embeddings: (batch, seq_len, hidden_dim) - BEHRT output
           visit_ids: (batch, seq_len) - Visit ID for each code
       
       Returns:
           visit_embeddings: (batch, max_visits, hidden_dim)
       """
       batch_size, seq_len, hidden_dim = code_embeddings.shape
       max_visit_id = visit_ids.max().item() + 1
       
       # Initialize visit embeddings
       visit_embeddings = torch.zeros(
           batch_size, max_visit_id, hidden_dim,
           device=code_embeddings.device
       )
       
       # Average pool codes within each visit
       for b in range(batch_size):
           for v in range(max_visit_id):
               mask = (visit_ids[b] == v)
               if mask.any():
                   visit_embeddings[b, v] = code_embeddings[b, mask].mean(dim=0)
       
       return visit_embeddings
   ```

3. **Hazard Prediction Head:**
   ```python
   self.hazard_head = nn.Sequential(
       nn.Linear(behrt_hidden_dim, hidden_dim),
       nn.LayerNorm(hidden_dim),
       nn.GELU(),
       nn.Dropout(dropout),
       nn.Linear(hidden_dim, 1),
       nn.Sigmoid()
   )
   ```

**Forward Pass:**

```python
def forward(self, codes, ages, visit_ids, segment_ids, attention_mask):
    """
    Args:
        codes: (batch, seq_len) - Flattened code sequence
        ages: (batch, seq_len) - Age at each code
        visit_ids: (batch, seq_len) - Visit ID for each code
        segment_ids: (batch, seq_len) - Segment ID (always 0 for single-sequence)
        attention_mask: (batch, seq_len) - 1 for real codes, 0 for padding
    
    Returns:
        hazards: (batch, max_visits) - Hazard at each visit
    """
    # BEHRT encoding (bidirectional)
    code_embeddings = self.behrt(
        codes=codes,
        ages=ages,
        segments=segment_ids,
        attention_mask=attention_mask
    )  # (batch, seq_len, hidden_dim)
    
    # Aggregate to visit-level
    visit_embeddings = self.aggregate_visits(code_embeddings, visit_ids)
    # Shape: (batch, max_visits, hidden_dim)
    
    # Predict hazard at each visit
    hazards = self.hazard_head(visit_embeddings).squeeze(-1)
    # Shape: (batch, max_visits)
    
    return hazards
```

---

## BEHRT vs LSTM Comparison

### Comparison Framework

**Goal:** Demonstrate that BEHRT's transformer-based representation learning provides advantages over LSTM for survival analysis.

### Experimental Design

**1. Same Data, Same Task**

- Use identical train/val/test splits
- Same synthetic data generation (controlled risk correlation)
- Same evaluation metrics (C-index, Brier score, calibration)

**2. Multiple Fine-Tuning Strategies for BEHRT**

Compare three approaches:

a. **Frozen Encoder** - Train only hazard head
   ```python
   for param in model.behrt.parameters():
       param.requires_grad = False
   ```

b. **LoRA Fine-Tuning** - Efficient adaptation
   ```python
   from ehrsequencing.models.lora import apply_lora_to_behrt
   model.behrt = apply_lora_to_behrt(model.behrt, rank=16, alpha=32)
   ```

c. **Full Fine-Tuning** - All parameters trainable
   ```python
   # All parameters trainable by default
   ```

**3. Fair Comparison Metrics**

| Metric | Purpose |
|--------|---------|
| **C-index** | Discrimination (ranking patients by risk) |
| **Brier Score** | Calibration (predicted vs observed probabilities) |
| **Training Time** | Computational efficiency |
| **Convergence Speed** | Epochs to reach best validation C-index |
| **Parameter Count** | Model complexity |
| **Train-Val Gap** | Generalization (overfitting indicator) |

### Expected Results

**Performance (C-index):**
```
LSTM Baseline:           0.70 - 0.75
BEHRT (Frozen):          0.72 - 0.77  (+2-5%)
BEHRT (LoRA):            0.75 - 0.80  (+5-10%)
BEHRT (Full):            0.76 - 0.82  (+8-12%)
```

**Training Efficiency:**
```
LSTM:                    50-100 epochs to converge
BEHRT (Frozen):          20-30 epochs (pre-trained!)
BEHRT (LoRA):            30-50 epochs
BEHRT (Full):            40-80 epochs
```

**Parameter Efficiency:**
```
LSTM:                    ~5M parameters
BEHRT (Frozen):          ~0.5M trainable (head only)
BEHRT (LoRA, rank=16):   ~1.2M trainable (10-20% of full)
BEHRT (Full):            ~25M trainable
```

### Hypothesis Testing

**H1: Pre-training improves performance**
- Compare BEHRT (pre-trained) vs BEHRT (from scratch)
- Expected: Pre-trained achieves higher C-index

**H2: BEHRT outperforms LSTM**
- Compare best BEHRT variant vs LSTM baseline
- Expected: BEHRT achieves 5-10% higher C-index

**H3: LoRA is parameter-efficient**
- Compare BEHRT (LoRA) vs BEHRT (Full)
- Expected: Similar C-index with 80-90% fewer trainable parameters

**H4: BEHRT generalizes better**
- Compare train-val gap for BEHRT vs LSTM
- Expected: BEHRT has smaller gap (less overfitting)

### Benchmark Script

```bash
# Run comprehensive comparison
python examples/survival_analysis/benchmark_behrt_vs_lstm.py \
    --num-patients 10000 \
    --vocab-size 1000 \
    --epochs 100 \
    --batch-size 64 \
    --output-dir experiments/behrt_vs_lstm/
```

**Output:**
```
experiments/behrt_vs_lstm/
├── SUMMARY.txt                    # Performance comparison table
├── summary.json                   # Detailed metrics
├── plots/
│   ├── c_index_comparison.png    # C-index across models
│   ├── training_curves.png       # Loss/C-index over epochs
│   ├── calibration_plots.png     # Predicted vs observed
│   └── parameter_efficiency.png  # Performance vs parameters
├── checkpoints/
│   ├── lstm_best.pt
│   ├── behrt_frozen_best.pt
│   ├── behrt_lora_best.pt
│   └── behrt_full_best.pt
└── logs/
    ├── lstm_training.log
    ├── behrt_frozen_training.log
    ├── behrt_lora_training.log
    └── behrt_full_training.log
```

---

## Implementation Details

### Data Preparation for BEHRT

BEHRT requires flattened sequences (all codes in one sequence) with visit boundaries:

```python
def prepare_behrt_format(patient_visits):
    """
    Convert visit-grouped sequences to BEHRT format
    
    Args:
        patient_visits: List of visits, each with codes
    
    Returns:
        codes: (seq_len,) - Flattened code sequence
        ages: (seq_len,) - Age at each code
        visit_ids: (seq_len,) - Visit ID for each code
        segment_ids: (seq_len,) - Always 0 (single sequence)
    """
    codes = []
    ages = []
    visit_ids = []
    
    for visit_id, visit in enumerate(patient_visits):
        visit_codes = visit['codes']
        visit_age = visit['age']
        
        codes.extend(visit_codes)
        ages.extend([visit_age] * len(visit_codes))
        visit_ids.extend([visit_id] * len(visit_codes))
    
    segment_ids = [0] * len(codes)  # Single sequence
    
    return {
        'codes': torch.tensor(codes),
        'ages': torch.tensor(ages),
        'visit_ids': torch.tensor(visit_ids),
        'segment_ids': torch.tensor(segment_ids)
    }
```

### Handling Variable-Length Sequences

Both models must handle variable-length sequences:

**LSTM:** Use `pack_padded_sequence` and `pad_packed_sequence`

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Pack sequences
lengths = visit_mask.sum(dim=1).cpu()
packed = pack_padded_sequence(
    visit_embeddings, lengths, batch_first=True, enforce_sorted=False
)

# LSTM forward
packed_out, _ = self.lstm(packed)

# Unpack
lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
```

**BEHRT:** Use attention masks

```python
# Attention mask: 1 for real codes, 0 for padding
attention_mask = (codes != PAD_TOKEN).long()

# BEHRT automatically handles masking
code_embeddings = self.behrt(
    codes=codes,
    ages=ages,
    segments=segment_ids,
    attention_mask=attention_mask
)
```

### Early Stopping

Prevent overfitting with early stopping on validation C-index:

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_c_index):
        if self.best_score is None:
            self.best_score = val_c_index
        elif val_c_index < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_c_index
            self.counter = 0
```

---

## Evaluation Metrics

### Concordance Index (C-index)

**Definition:** Probability that model correctly ranks pairs of patients by risk

$$C = \frac{\text{# concordant pairs}}{\text{# comparable pairs}}$$

**Implementation:**

```python
def concordance_index(risk_scores, event_times, event_indicators):
    """
    Compute C-index (Harrell's concordance)
    
    Args:
        risk_scores: (N,) - Predicted risk scores (higher = higher risk)
        event_times: (N,) - Observed event/censoring times
        event_indicators: (N,) - 1 if event, 0 if censored
    
    Returns:
        c_index: Scalar in [0, 1]
    """
    n = len(risk_scores)
    concordant = 0
    comparable = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Only compare if at least one has event
            if event_indicators[i] == 0 and event_indicators[j] == 0:
                continue
            
            # Determine which patient should have higher risk
            if event_times[i] < event_times[j]:
                # Patient i had event earlier (or censored earlier)
                if event_indicators[i] == 1:  # i had event
                    comparable += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
            elif event_times[j] < event_times[i]:
                # Patient j had event earlier
                if event_indicators[j] == 1:  # j had event
                    comparable += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
    
    return concordant / comparable if comparable > 0 else 0.5
```

**Interpretation:**
- C-index = 0.5: Random predictions
- C-index = 0.7: Good discrimination
- C-index = 0.8: Strong discrimination
- C-index = 1.0: Perfect discrimination

### Brier Score

**Definition:** Mean squared error of predicted probabilities

$$\text{Brier}(t) = \frac{1}{N} \sum_{i=1}^{N} (S_i(t) - I(T_i > t))^2$$

where $S_i(t)$ is predicted survival probability and $I(T_i > t)$ is true survival indicator.

**Implementation:**

```python
def brier_score(predicted_survival, true_survival, time_horizon):
    """
    Compute Brier score at specific time horizon
    
    Args:
        predicted_survival: (N, max_time) - Predicted S(t) for each patient
        true_survival: (N,) - True survival indicator (1 if survived, 0 if event)
        time_horizon: int - Time point to evaluate
    
    Returns:
        brier: Scalar
    """
    pred_at_t = predicted_survival[:, time_horizon]
    squared_errors = (pred_at_t - true_survival) ** 2
    return squared_errors.mean().item()
```

**Interpretation:**
- Lower is better
- Brier = 0: Perfect calibration
- Brier = 0.25: Random predictions (for balanced data)

### Calibration Plot

**Purpose:** Visualize agreement between predicted and observed event rates

```python
def plot_calibration(predicted_risks, event_indicators, n_bins=10):
    """
    Create calibration plot
    
    Args:
        predicted_risks: (N,) - Predicted event probabilities
        event_indicators: (N,) - True event indicators
        n_bins: Number of risk bins
    """
    import matplotlib.pyplot as plt
    
    # Bin patients by predicted risk
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted_risks, bins) - 1
    
    # Compute observed event rate in each bin
    observed_rates = []
    predicted_rates = []
    
    for b in range(n_bins):
        mask = (bin_indices == b)
        if mask.sum() > 0:
            observed_rates.append(event_indicators[mask].mean())
            predicted_rates.append(predicted_risks[mask].mean())
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(predicted_rates, observed_rates, 'o-', label='Model')
    plt.xlabel('Predicted Event Rate')
    plt.ylabel('Observed Event Rate')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('calibration_plot.png')
```

---

## Best Practices

### 1. Data Quality

- **Sufficient events:** Need ≥100 events for reliable C-index
- **Balanced censoring:** Avoid >80% censoring rate
- **Representative splits:** Stratify by event indicator

### 2. Model Selection

- **Start with LSTM** - Fast baseline, good performance
- **Try BEHRT if pre-trained** - Leverage transfer learning
- **Use LoRA for efficiency** - 80% fewer parameters, similar performance

### 3. Hyperparameter Tuning

**LSTM:**
```python
config = {
    'embedding_dim': 128,      # 64-256
    'hidden_dim': 256,         # 128-512
    'num_layers': 2,           # 1-3
    'dropout': 0.3,            # 0.2-0.5
    'learning_rate': 1e-3,     # 1e-4 to 1e-2
    'weight_decay': 1e-5       # 1e-6 to 1e-4
}
```

**BEHRT:**
```python
config = {
    'hidden_dim': 256,         # Hazard head hidden dim
    'dropout': 0.1,            # Lower than LSTM (pre-trained)
    'learning_rate': 1e-5,     # Lower for fine-tuning
    'lora_rank': 16,           # 8-32 for LoRA
    'lora_alpha': 32,          # Usually 2x rank
    'freeze_embeddings': True  # Freeze pre-trained embeddings
}
```

### 4. Avoiding Overfitting

- **Early stopping:** Patience 10-20 epochs on validation C-index
- **Dropout:** 0.2-0.4 for LSTM, 0.1-0.2 for BEHRT
- **Weight decay:** L2 regularization (1e-5)
- **Cross-validation:** For small datasets (<1000 patients)

### 5. Clinical Validation

- **Calibration:** Check predicted vs observed event rates
- **Subgroup analysis:** Stratify by age, gender, comorbidities
- **Temporal validation:** Train on old data, test on new
- **External validation:** Test on different hospital/dataset

---

## References

### Papers

1. **Discrete-Time Survival Analysis**
   - Tutz & Schmid (2016). "Modeling Discrete Time-to-Event Data"
   - Singer & Willett (1993). "It's About Time: Using Discrete-Time Survival Analysis"

2. **C-index and Evaluation**
   - Harrell et al. (1982). "Evaluating the Yield of Medical Tests"
   - Graf et al. (1999). "Assessment and Comparison of Prognostic Classification Schemes"

3. **BEHRT**
   - Li et al. (2020). "BEHRT: Transformer for Electronic Health Records"
   - Rasmy et al. (2021). "Med-BERT: Pre-trained Contextualized Embeddings"

4. **LoRA**
   - Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"

### Code Examples

- `src/ehrsequencing/models/survival_lstm.py` - LSTM implementation
- `src/ehrsequencing/models/behrt_survival.py` - BEHRT survival (in development)
- `src/ehrsequencing/models/losses.py` - Survival loss functions
- `examples/survival_analysis/train_lstm.py` - LSTM training script
- `examples/survival_analysis/benchmark_behrt_vs_lstm.py` - Comparison benchmark
- `notebooks/01_discrete_time_survival_lstm.ipynb` - Interactive tutorial

---

**Document Status:** Complete  
**Last Updated:** February 5, 2026  
**Next Steps:** Implement BEHRTForSurvival and run benchmarks
