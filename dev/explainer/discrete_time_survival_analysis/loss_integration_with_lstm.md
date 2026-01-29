# Loss Function Integration with Survival LSTM Models

## Overview

This document explains how the discrete-time survival loss functions (`@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/models/losses.py`) integrate with the LSTM-based survival models (`@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/models/survival_lstm.py`) and clarifies the role of sequence packing in the training pipeline.

## Model Architecture Overview

### DiscreteTimeSurvivalLSTM

The `DiscreteTimeSurvivalLSTM` model follows this architecture:

```
Input: Medical codes per visit
    ↓
[1] Code Embedding Layer
    ↓
[2] Within-Visit Aggregation (mean pooling)
    ↓
[3] LSTM over Visit Sequence
    ↓
[4] Hazard Prediction Head (with sigmoid)
    ↓
Output: Hazard probabilities per visit
```

### Key Components

From `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/models/survival_lstm.py:95-138`:

```python
def forward(
    self,
    visit_codes: torch.Tensor,      # [B, V, C]
    visit_mask: torch.Tensor,       # [B, V, C]
    sequence_mask: torch.Tensor,    # [B, V]
) -> torch.Tensor:
    # Step 1: Embed codes
    embeddings = self.embedding(visit_codes)  # [B, V, C, E]
    
    # Step 2: Aggregate codes within each visit (mean pooling)
    visit_mask_expanded = visit_mask.unsqueeze(-1).float()
    visit_vectors = (embeddings * visit_mask_expanded).sum(dim=2)  # [B, V, E]
    num_codes_per_visit = visit_mask.sum(dim=2, keepdim=True).clamp(min=1)  # [B, V, 1]
    visit_vectors = visit_vectors / num_codes_per_visit  # [B, V, E] / [B, V, 1] = [B, V, E]
    
    # Step 3: LSTM over visits
    lstm_out, _ = self.lstm(visit_vectors)  # [B, V, H]
    
    # Step 4: Map to hazards
    hazards = self.hazard_head(lstm_out).squeeze(-1)  # [B, V]
    
    # Step 5: Mask padding
    hazards = hazards * sequence_mask.float()
    
    return hazards
```

**Dimensions**:
- `B` = batch size
- `V` = max number of visits (padded)
- `C` = max codes per visit (padded)
- `E` = embedding dimension
- `H` = LSTM hidden dimension

## Integration with Loss Function

### Training Loop

From `@/Users/pleiadian53/work/ehr-sequencing/examples/train_survival_lstm.py:236-262`:

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Move to device
        visit_codes = batch['visit_codes'].to(device)          # [B, V, C]
        visit_mask = batch['visit_mask'].to(device)            # [B, V, C]
        sequence_mask = batch['sequence_mask'].to(device)      # [B, V]
        event_times = batch['event_times'].to(device)          # [B]
        event_indicators = batch['event_indicators'].to(device) # [B]
        
        # Forward pass: Get hazards at each visit
        hazards = model(visit_codes, visit_mask, sequence_mask)  # [B, V]
        
        # Compute loss
        loss = criterion(hazards, event_times, event_indicators, sequence_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### Data Flow Diagram

```
Patient Sequences (Variable Length)
         ↓
    [Collate Function]
         ↓
Padded Tensors + Masks
         ↓
    [LSTM Model]
         ↓
Hazard Predictions (with padding masked to 0)
         ↓
    [Loss Function]
         ↓
Negative Log-Likelihood (averaged over batch)
```

### Why the Loss Needs sequence_mask

The loss function uses `sequence_mask` to distinguish real visits from padding:

From `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/models/losses.py:91-95`:

```python
# Mask for visits before event/censoring
before_event_mask = (time_idx < event_times_expanded).float() * sequence_mask

# Mask for event visit
at_event_mask = (time_idx == event_times_expanded).float() * sequence_mask
```

**Why this matters**:
1. **Padding visits have hazard = 0** (from model masking)
2. **But we need to exclude them from loss computation** to avoid:
   - Adding spurious $\log(1 - 0) = 0$ terms that don't contribute meaningful gradients
   - Counting padding as "survival" which would bias the loss
   - Including padding in the event time comparison

### Example: Loss Computation with Padding

Consider a batch with 2 patients:

```python
# Patient 0: 3 real visits, event at visit 2
# Patient 1: 2 real visits, censored at visit 1

hazards = torch.tensor([
    [0.1, 0.2, 0.3, 0.0, 0.0],  # Patient 0: 3 real + 2 padding
    [0.15, 0.25, 0.0, 0.0, 0.0]  # Patient 1: 2 real + 3 padding
])

event_times = torch.tensor([2, 1])
event_indicators = torch.tensor([1, 0])  # 0 observed, 1 censored
sequence_mask = torch.tensor([
    [1, 1, 1, 0, 0],  # 3 valid visits
    [1, 1, 0, 0, 0]   # 2 valid visits
])
```

**Patient 0 (event at visit 2)**:
- `before_event_mask[0]`: `[1, 1, 0, 0, 0]` (visits 0, 1)
- `at_event_mask[0]`: `[0, 0, 1, 0, 0]` (visit 2)
- Survival LL: $\log(0.9) + \log(0.8)$
- Event LL: $\log(0.3)$
- Padding visits 3, 4 are excluded

**Patient 1 (censored at visit 1)**:
- `before_event_mask[1]`: `[1, 0, 0, 0, 0]` (visit 0)
- `at_event_mask[1]`: `[0, 1, 0, 0, 0]` (visit 1)
- Survival LL: $\log(0.85) + \log(0.75)$ (includes visit 1 since censored)
- Event LL: $0$ (censored, so $\delta = 0$)
- Padding visits 2, 3, 4 are excluded

## The Role of pack_padded_sequence

### What is pack_padded_sequence?

`nn.utils.rnn.pack_padded_sequence` is a PyTorch utility that:
1. Removes padding from variable-length sequences
2. Packs them into a compact representation
3. Allows LSTM to process only real data (not padding)

### Why Do We Need It?

**Short answer**: Computational efficiency and gradient quality.

**Detailed explanation**:

#### Without Packing (Naive Approach)

```python
# All sequences padded to max_visits
visit_vectors = ...  # [batch_size, max_visits, embedding_dim]
lstm_out, (hidden, cell) = self.lstm(visit_vectors)
```

**Problems**:
1. **Wasted computation**: LSTM processes padding tokens
2. **Hidden state contamination**: LSTM hidden states get updated with padding information
3. **Gradient dilution**: Gradients flow through padding steps

#### With Packing (Efficient Approach)

From `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/models/lstm_baseline.py:252-279`:

```python
if sequence_mask is not None:
    # Pack padded sequence for efficiency
    lengths = sequence_mask.sum(dim=1).cpu()  # Get actual lengths
    
    packed_input = nn.utils.rnn.pack_padded_sequence(
        visit_vectors,
        lengths,
        batch_first=True,
        enforce_sorted=False
    )
    
    packed_output, (hidden, cell) = self.lstm(packed_input)
    
    lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
        packed_output,
        batch_first=True
    )
else:
    lstm_output, (hidden, cell) = self.lstm(visit_vectors)
```

**Benefits**:
1. **Efficiency**: LSTM only processes real visits
2. **Clean hidden states**: No contamination from padding
3. **Better gradients**: Gradients only flow through real data

### Visualization: Packed vs Unpacked

**Unpacked (Naive)**:
```
Patient 0: [v0, v1, v2, PAD, PAD]  → LSTM processes all 5
Patient 1: [v0, v1, PAD, PAD, PAD] → LSTM processes all 5
Patient 2: [v0, PAD, PAD, PAD, PAD] → LSTM processes all 5
```

**Packed (Efficient)**:
```
Step 0: [v0_p0, v0_p1, v0_p2]  → Process 3 patients
Step 1: [v1_p0, v1_p1]         → Process 2 patients (p2 done)
Step 2: [v2_p0]                → Process 1 patient  (p1 done)
                               → (p0 done)
```

The packed representation groups visits by time step and only processes patients still active.

### Connection to Loss Computation

**Important**: While `pack_padded_sequence` improves LSTM efficiency, it's **not strictly required for correct loss computation** because:

1. The model already masks hazards: `hazards = hazards * sequence_mask.float()`
2. The loss function uses `sequence_mask` to exclude padding

However, packing provides:
- **Better training efficiency** (faster, less memory)
- **Cleaner gradients** (no gradient flow through padding)
- **More accurate hidden states** (not contaminated by padding)

### Why Current Implementation Doesn't Use Packing

Looking at `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/models/survival_lstm.py:129-130`:

```python
# LSTM over visits: [B, V, H]
lstm_out, _ = self.lstm(visit_vectors)
```

The current `DiscreteTimeSurvivalLSTM` **does not use packing**. This is acceptable because:

1. **Simplicity**: Easier to implement and understand
2. **Masking handles correctness**: The model masks outputs, loss masks inputs
3. **Small overhead**: For typical EHR sequences (10-50 visits), padding overhead is manageable

However, for production systems with:
- Very long sequences (100+ visits)
- Large batch sizes
- Highly variable sequence lengths

Adding packing would improve efficiency.

### Recommended Implementation with Packing

Here's how to add packing to `DiscreteTimeSurvivalLSTM`:

```python
def forward(
    self,
    visit_codes: torch.Tensor,
    visit_mask: torch.Tensor,
    sequence_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_visits, max_codes = visit_codes.shape
    
    # Embed and aggregate (same as before)
    embeddings = self.embedding(visit_codes)
    visit_mask_expanded = visit_mask.unsqueeze(-1).float()
    visit_vectors = (embeddings * visit_mask_expanded).sum(dim=2)
    num_codes_per_visit = visit_mask.sum(dim=2, keepdim=True).clamp(min=1)
    visit_vectors = visit_vectors / num_codes_per_visit
    
    # Pack sequences for efficient LSTM processing
    lengths = sequence_mask.sum(dim=1).cpu()
    packed_input = nn.utils.rnn.pack_padded_sequence(
        visit_vectors,
        lengths,
        batch_first=True,
        enforce_sorted=False
    )
    
    # LSTM over packed sequences
    packed_output, _ = self.lstm(packed_input)
    
    # Unpack to get back padded format
    lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
        packed_output,
        batch_first=True,
        total_length=num_visits  # Ensure same length as input
    )
    
    # Map to hazards (same as before)
    hazards = self.hazard_head(lstm_out).squeeze(-1)
    hazards = hazards * sequence_mask.float()
    
    return hazards
```

## Complete Data Pipeline

### 1. Data Preparation (Collate Function)

From `@/Users/pleiadian53/work/ehr-sequencing/examples/train_survival_lstm.py:179-233`:

```python
def collate_fn(batch):
    sequences = [item['sequence'] for item in batch]
    event_times = torch.tensor([item['event_time'] for item in batch])
    event_indicators = torch.tensor([item['event_indicator'] for item in batch])
    
    # Find max dimensions
    max_visits = max(len(seq.visits) for seq in sequences)
    max_codes_per_visit = max(
        max(visit.num_codes() for visit in seq.visits)
        for seq in sequences
    )
    
    batch_size = len(sequences)
    
    # Initialize padded tensors
    visit_codes = torch.zeros(batch_size, max_visits, max_codes_per_visit, dtype=torch.long)
    visit_mask = torch.zeros(batch_size, max_visits, max_codes_per_visit, dtype=torch.bool)
    sequence_mask = torch.zeros(batch_size, max_visits, dtype=torch.bool)
    
    # Fill with actual data
    for i, seq in enumerate(sequences):
        num_visits = len(seq.visits)
        sequence_mask[i, :num_visits] = True
        
        for j, visit in enumerate(seq.visits):
            codes = visit.get_all_codes()
            encoded_codes = [
                builder.vocab.get(code, builder.unk_id)
                for code in codes[:max_codes_per_visit]
            ]
            num_codes = len(encoded_codes)
            
            if num_codes > 0:
                visit_codes[i, j, :num_codes] = torch.tensor(encoded_codes)
                visit_mask[i, j, :num_codes] = True
    
    return {
        'visit_codes': visit_codes,
        'visit_mask': visit_mask,
        'sequence_mask': sequence_mask,
        'event_times': event_times,
        'event_indicators': event_indicators,
    }
```

**Key points**:
- Pads all sequences to `max_visits` in batch
- Creates `sequence_mask` to track real vs. padded visits
- Creates `visit_mask` to track real vs. padded codes within visits

### 2. Model Forward Pass

```python
hazards = model(visit_codes, visit_mask, sequence_mask)
```

**Output**: `hazards` shape `[batch_size, max_visits]`
- Real visits have hazards in $(0, 1)$
- Padded visits have hazards = 0 (due to masking)

### 3. Loss Computation

```python
loss = criterion(hazards, event_times, event_indicators, sequence_mask)
```

**The loss function**:
1. Uses `sequence_mask` to identify real visits
2. Computes survival likelihood for visits before event
3. Computes event likelihood at event time (if observed)
4. Ignores all padding visits

### 4. Backward Pass

```python
loss.backward()
optimizer.step()
```

**Gradient flow**:
- Gradients flow from loss → hazards → LSTM → embeddings
- Only real visits contribute to gradients (due to masking)
- Padding visits have zero gradients

## Masking Strategy: Three Levels

The pipeline uses **three levels of masking**:

### Level 1: Code-Level Masking (visit_mask)

**Purpose**: Handle variable number of codes per visit

```python
# In model forward
visit_mask_expanded = visit_mask.unsqueeze(-1).float()  # [B, V, C, 1]
visit_vectors = (embeddings * visit_mask_expanded).sum(dim=2)
```

**Effect**: Only real codes contribute to visit representation

### Level 2: Visit-Level Masking (sequence_mask in model)

**Purpose**: Zero out hazards for padding visits

```python
# In model forward
hazards = hazards * sequence_mask.float()
```

**Effect**: Padding visits have hazard = 0

### Level 3: Visit-Level Masking (sequence_mask in loss)

**Purpose**: Exclude padding from loss computation

```python
# In loss forward
before_event_mask = (time_idx < event_times_expanded).float() * sequence_mask
at_event_mask = (time_idx == event_times_expanded).float() * sequence_mask
```

**Effect**: Padding visits don't contribute to loss

### Why Three Levels?

1. **Level 1** ensures visit representations are meaningful
2. **Level 2** ensures model outputs are clean (for inference)
3. **Level 3** ensures loss is computed correctly (for training)

All three work together to handle variable-length sequences correctly.

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Mask Model Outputs

**Problem**:
```python
# BAD: No masking
hazards = self.hazard_head(lstm_out).squeeze(-1)
return hazards
```

**Result**: Padding visits have non-zero hazards, contaminating loss

**Solution**:
```python
# GOOD: Mask outputs
hazards = self.hazard_head(lstm_out).squeeze(-1)
hazards = hazards * sequence_mask.float()
return hazards
```

### Pitfall 2: Not Passing sequence_mask to Loss

**Problem**:
```python
# BAD: No mask
loss = criterion(hazards, event_times, event_indicators, None)
```

**Result**: Loss tries to compute on padding visits, gets incorrect values

**Solution**:
```python
# GOOD: Pass mask
loss = criterion(hazards, event_times, event_indicators, sequence_mask)
```

### Pitfall 3: Event Time Beyond Sequence Length

**Problem**: `event_times[i] >= sequence_mask[i].sum()`

**Result**: Event time points to padding, loss computation fails

**Solution**: Validate during data loading:
```python
for i in range(batch_size):
    num_visits = sequence_mask[i].sum().item()
    assert event_times[i] < num_visits, f"Event time {event_times[i]} >= num visits {num_visits}"
```

### Pitfall 4: Using pack_padded_sequence Without enforce_sorted=False

**Problem**:
```python
# BAD: Requires sorted sequences
packed_input = nn.utils.rnn.pack_padded_sequence(
    visit_vectors,
    lengths,
    batch_first=True
)  # Default: enforce_sorted=True
```

**Result**: Error if sequences not sorted by length

**Solution**:
```python
# GOOD: Allow unsorted
packed_input = nn.utils.rnn.pack_padded_sequence(
    visit_vectors,
    lengths,
    batch_first=True,
    enforce_sorted=False
)
```

## Performance Considerations

### Memory Usage

**Without packing**:
- LSTM processes: `batch_size × max_visits × hidden_dim`
- Memory scales with longest sequence in batch

**With packing**:
- LSTM processes: `sum(actual_lengths) × hidden_dim`
- Memory scales with total real visits

**Example**: Batch of 32 patients, max 50 visits, average 15 visits
- Without packing: 32 × 50 = 1,600 visit-steps
- With packing: 32 × 15 = 480 visit-steps
- **Speedup**: ~3.3x

### When to Use Packing

**Use packing when**:
- Sequence lengths vary significantly (e.g., 10-100 visits)
- Batch sizes are large (>32)
- Training time is a bottleneck

**Skip packing when**:
- Sequences have similar lengths
- Batch sizes are small
- Code simplicity is prioritized

## Summary

### Key Takeaways

1. **Loss function requires `sequence_mask`** to correctly handle variable-length sequences and exclude padding from likelihood computation

2. **Model masking and loss masking are complementary**:
   - Model masking: Ensures clean outputs (hazards = 0 for padding)
   - Loss masking: Ensures correct loss computation (excludes padding)

3. **`pack_padded_sequence` is optional but beneficial**:
   - Not required for correctness (masking handles that)
   - Improves efficiency by skipping padding in LSTM
   - Provides cleaner gradients and hidden states

4. **Three-level masking strategy**:
   - Code-level: Aggregate only real codes
   - Visit-level (model): Zero out padding hazards
   - Visit-level (loss): Exclude padding from loss

5. **The pipeline is robust** because masking is applied at multiple stages, ensuring correctness even if one level is missed (though all three are recommended)

### Architecture Flow

```
Raw EHR Data
    ↓
[Collate: Pad + Create Masks]
    ↓
Padded Tensors + sequence_mask
    ↓
[Model: Embed → Aggregate → LSTM → Hazard Head]
    ↓
Hazards (masked to 0 for padding)
    ↓
[Loss: Use sequence_mask to exclude padding]
    ↓
Negative Log-Likelihood
    ↓
[Backward: Gradients only for real visits]
    ↓
Parameter Updates
```

---

**Document Version**: 1.0  
**Last Updated**: January 27, 2026  
**Related Documents**:
- `discrete_time_survival_loss.md` - Mathematical foundation of the loss function
- `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/models/losses.py` - Loss implementation
- `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/models/survival_lstm.py` - Model implementation
- `@/Users/pleiadian53/work/ehr-sequencing/examples/train_survival_lstm.py` - Training example
