# Handling Variable-Length Patient Histories in Deep Learning

**A Practical Guide to Avoiding Temporal Leakage in EHR Sequence Models**

---

## Table of Contents

1. [The Challenge: Variable-Length Sequences](#the-challenge)
2. [Why Padding is Dangerous](#why-padding-is-dangerous)
3. [The Solution: Packed Sequences](#the-solution)
4. [Implementation Guide](#implementation-guide)
5. [Common Pitfalls and How to Avoid Them](#common-pitfalls)
6. [Best Practices for EHR Modeling](#best-practices)

---

## The Challenge: Variable-Length Sequences {#the-challenge}

### The Problem

In real-world EHR data, patients have vastly different history lengths:

```text
Patient A: 3 visits   [Visit 1] → [Visit 2] → [Visit 3]
Patient B: 17 visits  [Visit 1] → [Visit 2] → ... → [Visit 17]
Patient C: 42 visits  [Visit 1] → [Visit 2] → ... → [Visit 42]
```

But deep learning frameworks require **fixed-size tensors**:

```python
# PyTorch/TensorFlow want rectangular tensors
batch_tensor = torch.zeros(batch_size, max_visits, feature_dim)
```

### The Standard Solution: Padding

We pad shorter sequences with zeros to match the longest sequence:

```python
# Patient A (3 real visits, 39 padding)
[Visit_1, Visit_2, Visit_3, PAD, PAD, PAD, ..., PAD]

# Patient C (42 real visits, 0 padding)
[Visit_1, Visit_2, Visit_3, ..., Visit_42]
```

**This creates a fundamental problem:**

> Padding is not data, but models don't automatically know that.

---

## Why Padding is Dangerous {#why-padding-is-dangerous}

### Problem 1: The "Last Timestep" Trap

Consider a patient with 3 real visits in a batch where `max_visits = 10`:

```text
Index:  0        1        2        3    4    5    ...  9
Data:   Visit_1  Visit_2  Visit_3  PAD  PAD  PAD  ...  PAD
                          ↑ real last visit
                                                        ↑ tensor last position
```

**Naïve approach (WRONG):**

```python
lstm_output, (hidden, cell) = lstm(padded_sequences)
last_hidden = lstm_output[:, -1, :]  # ❌ This is the hidden state AFTER padding!
```

This hidden state is influenced by:
- Zero inputs from padding
- Learned biases applied to padding
- The recurrent computation continuing through non-existent time

### Problem 2: Temporal Information Leakage

The real danger emerges when computing loss over all timesteps:

```python
# Dangerous pattern
for t in range(max_visits):
    loss += criterion(predictions[:, t], labels[:, t])
```

This means the model learns:

```text
PAD → PAD → PAD → prediction
```

**Result:** The model learns that padding patterns predict outcomes, which is catastrophic for generalization.

### Problem 3: Future Information Leakage

In disease progression tasks, this is especially dangerous:

```python
# Labels computed using full patient history
progression_label = did_patient_progress_within_1yr(patient)

# Then used at ALL timesteps
for t in range(num_visits):
    loss += criterion(model_output[t], progression_label)
```

This leaks future knowledge backwards:
- Early visits "know" what happens years later
- Model performance looks amazing in training
- Model fails completely in prospective deployment

---

## The Solution: Packed Sequences {#the-solution}

### What `pack_padded_sequence` Does

PyTorch's `pack_padded_sequence` tells the LSTM:

> "Only process real timesteps. Ignore padding entirely."

It transforms the rectangular tensor:

```python
[batch_size, max_visits, feature_dim]
```

into a compact representation:

```python
(total_real_visits_across_batch, feature_dim)
```

plus metadata that tells the LSTM when each sequence ends.

### Key Benefits

1. **No padding processing**: LSTM never sees padding tokens
2. **Correct recurrence**: Stops exactly at each patient's last real visit
3. **Computational efficiency**: Skips unnecessary computations
4. **Correct hidden states**: `h_n` contains true final states by construction

---

## Implementation Guide {#implementation-guide}

### Step-by-Step: Correct Usage

#### Step 1: Track True Sequence Lengths

```python
# Number of real visits per patient
lengths = torch.tensor([3, 7, 10, 5])  # batch_size = 4
```

#### Step 2: Create Padded Tensor

```python
batch_size = 4
max_visits = 10
feature_dim = 128

# Padded sequences
padded_sequences = torch.zeros(batch_size, max_visits, feature_dim)

# Fill with real data
for i, patient_data in enumerate(batch_data):
    real_length = lengths[i]
    padded_sequences[i, :real_length] = patient_data
```

#### Step 3: Pack the Sequences

```python
from torch.nn.utils.rnn import pack_padded_sequence

packed_sequences = pack_padded_sequence(
    padded_sequences,
    lengths.cpu(),  # Must be on CPU
    batch_first=True,
    enforce_sorted=False  # Allows unsorted lengths
)
```

**Note:** If `enforce_sorted=True`, you must sort by length descending:

```python
lengths, perm_idx = lengths.sort(descending=True)
padded_sequences = padded_sequences[perm_idx]
```

#### Step 4: Run LSTM

```python
packed_output, (h_n, c_n) = lstm(packed_sequences)
```

Now:
- `h_n[-1, i]` is the hidden state at patient `i`'s **true last visit**
- No padding was processed

#### Step 5: (Optional) Unpack for Per-Visit Outputs

```python
from torch.nn.utils.rnn import pad_packed_sequence

unpacked_output, output_lengths = pad_packed_sequence(
    packed_output,
    batch_first=True
)
```

**Important:** `unpacked_output[i, t]` is only valid for `t < lengths[i]`

---

## Common Pitfalls and How to Avoid Them {#common-pitfalls}

### Pitfall 1: Using Wrong "Last" Hidden State

❌ **Wrong:**

```python
lstm_output, _ = lstm(padded_sequences)
last_hidden = lstm_output[:, -1, :]  # After padding!
```

✅ **Correct:**

```python
packed_input = pack_padded_sequence(padded_sequences, lengths, batch_first=True)
_, (h_n, c_n) = lstm(packed_input)
last_hidden = h_n[-1]  # True last visit
```

### Pitfall 2: Computing Loss Over Padding

❌ **Wrong:**

```python
predictions = model(padded_sequences)  # [B, T_max, K]
loss = criterion(predictions, labels)  # Includes padding!
```

✅ **Correct:**

```python
# Create mask for real timesteps
mask = torch.arange(max_visits)[None, :] < lengths[:, None]  # [B, T_max]

# Only compute loss on real visits
loss = criterion(predictions[mask], labels[mask])
```

### Pitfall 3: Non-Causal Labels

❌ **Wrong:**

```python
# Label uses information from entire patient history
label = patient.had_outcome_ever()

# Applied to all visits
for t in range(num_visits):
    loss += criterion(pred[t], label)
```

✅ **Correct:**

```python
# Label is causal: only uses information available at time t
for t in range(num_visits):
    # Predict outcome in next 6 months from visit t
    label_t = patient.had_outcome_between(t, t + 6_months)
    loss += criterion(pred[t], label_t)
```

---

## Best Practices for EHR Modeling {#best-practices}

### Checklist: EHR-Safe LSTM Implementation

Ensure all of these are true:

- ✅ Use `pack_padded_sequence` for variable visit counts
- ✅ Use `h_n[-1]` for patient-level representations
- ✅ Mask losses for visit-level predictions
- ✅ Define labels causally per visit
- ✅ Never let padding participate in loss or recurrence
- ✅ Validate that padding is truly ignored (check gradients)

### Pattern A: Patient-Level Prediction (Many-to-One)

**Task:** Predict patient outcome using full history

```python
def forward(self, visit_sequences, lengths):
    # Pack sequences
    packed = pack_padded_sequence(
        visit_sequences, 
        lengths, 
        batch_first=True,
        enforce_sorted=False
    )
    
    # LSTM
    _, (h_n, c_n) = self.lstm(packed)
    
    # Use final hidden state
    patient_repr = h_n[-1]  # [batch_size, hidden_dim]
    
    # Predict
    logits = self.classifier(patient_repr)
    return logits
```

**Interpretation:** "Predict using what the patient looked like at their last real visit"

### Pattern B: Visit-Level Prediction (Many-to-Many)

**Task:** Predict outcome at each visit

```python
def forward(self, visit_sequences, lengths):
    # Pack
    packed = pack_padded_sequence(
        visit_sequences,
        lengths,
        batch_first=True,
        enforce_sorted=False
    )
    
    # LSTM
    packed_output, _ = self.lstm(packed)
    
    # Unpack
    lstm_output, _ = pad_packed_sequence(
        packed_output,
        batch_first=True
    )  # [batch_size, max_visits, hidden_dim]
    
    # Predict at each visit
    logits = self.classifier(lstm_output)  # [batch_size, max_visits, num_classes]
    
    return logits, lengths

def compute_loss(self, logits, labels, lengths):
    batch_size, max_visits = logits.shape[:2]
    
    # Create mask for real visits
    mask = torch.arange(max_visits)[None, :] < lengths[:, None]
    
    # Only compute loss on real visits
    loss = self.criterion(logits[mask], labels[mask])
    return loss
```

### Pattern C: Causal Label Definition

**Critical:** Labels must only use information available up to time `t`

```python
def create_causal_labels(patient_visits, prediction_horizon='6M'):
    """
    Create labels that are causal with respect to each visit.
    
    Args:
        patient_visits: List of visits with timestamps
        prediction_horizon: How far ahead to predict (e.g., '6M', '1Y')
    
    Returns:
        labels: [num_visits] - outcome occurred within horizon
    """
    labels = []
    
    for i, visit in enumerate(patient_visits):
        visit_time = visit.timestamp
        horizon_end = visit_time + prediction_horizon
        
        # Check if outcome occurred in prediction window
        # ONLY using information AFTER this visit
        future_visits = patient_visits[i+1:]
        outcome_in_window = any(
            visit_time < v.timestamp <= horizon_end and v.has_outcome
            for v in future_visits
        )
        
        labels.append(outcome_in_window)
    
    return torch.tensor(labels)
```

---

## Philosophical Understanding

### Mental Model: Stopped Stochastic Processes

An LSTM with packed sequences models:

> A stopped stochastic process where each patient trajectory ends at a different time.

**Key insight:** Padding is not "missing data" - it's **non-existent time**.

If padding is treated as time, the model will exploit it. This is not a bug in the code; it's a fundamental modeling error.

### The Clinical Interpretation

When you use `pack_padded_sequence`:

- **Without packing:** "Last timestep" = `max_visits - 1` (artifact of batching)
- **With packing:** "Last timestep" = `lengths[i] - 1` (clinical event boundary)

This distinction is everything in EHR modeling.

---

## Summary

Variable-length sequences are ubiquitous in EHR data. Handling them correctly requires:

1. **Use packed sequences** to prevent padding from influencing the model
2. **Extract hidden states correctly** using `h_n` not `output[:, -1]`
3. **Mask losses** when making per-visit predictions
4. **Define labels causally** to prevent future information leakage
5. **Consider carefully** what "time" means in the model

**The bottom line:** If you see suspiciously good results on EHR sequence modeling, check your padding and temporal leakage first.

---

## Further Reading

- **Next topic:** Designing progression labels that are both causal and statistically efficient (handling censoring and irregular follow-up)
- **Related:** How to handle missing visits vs. padding (they're different!)
- **Advanced:** Attention mechanisms and variable-length sequences

---

**Last updated:** January 2026  
**Related code:** `src/ehrsequencing/models/lstm_baseline.py`
