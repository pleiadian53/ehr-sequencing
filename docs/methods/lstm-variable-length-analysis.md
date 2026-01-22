# Analysis: How Our LSTM Baseline Handles Variable-Length Sequences

**Date:** January 21, 2026  
**Model:** `src/ehrsequencing/models/lstm_baseline.py`

---

## Summary

Our current LSTM baseline implementation **correctly handles variable-length sequences** using `pack_padded_sequence`. Here's the detailed analysis.

---

## Implementation Review

### Code Location: Lines 256-278

```python
# Apply LSTM across visits
if sequence_mask is not None:
    # Pack padded sequence for efficiency
    lengths = sequence_mask.sum(dim=1).cpu()
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

# Use last hidden state for prediction
if self.bidirectional:
    # Concatenate forward and backward final hidden states
    final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
else:
    final_hidden = hidden[-1]
```

---

## What We Do Correctly ✅

### 1. **Use Packed Sequences**

```python
packed_input = nn.utils.rnn.pack_padded_sequence(
    visit_vectors,
    lengths,
    batch_first=True,
    enforce_sorted=False  # ✅ Allows unsorted batch
)
```

**Why this is correct:**
- LSTM never processes padding
- Recurrence stops at each patient's true last visit
- Computationally efficient

### 2. **Extract True Lengths from Mask**

```python
lengths = sequence_mask.sum(dim=1).cpu()
```

**Input:** `sequence_mask` is `[batch_size, num_visits]` with 1 for real visits, 0 for padding

**Example:**
```python
sequence_mask = torch.tensor([
    [1, 1, 1, 0, 0],  # Patient 0: 3 visits
    [1, 1, 1, 1, 1],  # Patient 1: 5 visits
    [1, 1, 0, 0, 0],  # Patient 2: 2 visits
])
lengths = tensor([3, 5, 2])  # ✅ Correct
```

### 3. **Use Correct Final Hidden State**

```python
final_hidden = hidden[-1]  # ✅ NOT lstm_output[:, -1, :]
```

**Why this is correct:**
- `hidden[-1]` contains the hidden state at each patient's **true last visit**
- This is guaranteed by `pack_padded_sequence`
- No contamination from padding

### 4. **Handle Bidirectional LSTM Correctly**

```python
if self.bidirectional:
    final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
```

**Why this is correct:**
- `hidden[-2]` = forward direction final state
- `hidden[-1]` = backward direction final state
- Concatenating gives full bidirectional representation

### 5. **Support Optional Packing**

```python
if sequence_mask is not None:
    # Use packed sequences
else:
    # Fall back to regular LSTM
    lstm_output, (hidden, cell) = self.lstm(visit_vectors)
```

**Why this is good:**
- Allows usage without masks (e.g., for debugging)
- Graceful degradation

---

## Two-Level Masking Strategy

Our implementation uses **two levels of masking**, which is the correct approach:

### Level 1: Visit-Level Masking

```python
visit_mask: [batch_size, num_visits, max_codes_per_visit]
```

**Purpose:** Mask padding codes within each visit

**Handled by:** `VisitEncoder` (lines 246-252)

```python
visit_vectors = self.visit_encoder(code_embeddings_flat, visit_mask_flat)
```

### Level 2: Sequence-Level Masking

```python
sequence_mask: [batch_size, num_visits]
```

**Purpose:** Mask padding visits in the sequence

**Handled by:** `pack_padded_sequence` (lines 256-269)

**This is the correct architecture** because:
1. Visits can have variable numbers of codes
2. Patients can have variable numbers of visits
3. Both need separate masking

---

## What Could Be Improved 🔧

### Issue 1: No Explicit Loss Masking for Visit-Level Predictions

**Current implementation:** Only supports patient-level prediction (many-to-one)

**What's missing:** If we want visit-level predictions (many-to-many), we need to mask the loss:

```python
# This is NOT currently in the model
def compute_visit_level_loss(self, logits, labels, lengths):
    """
    Compute loss only on real visits, not padding.
    
    Args:
        logits: [batch_size, max_visits, num_classes]
        labels: [batch_size, max_visits]
        lengths: [batch_size] - number of real visits
    """
    batch_size, max_visits = logits.shape[:2]
    
    # Create mask
    mask = torch.arange(max_visits)[None, :] < lengths[:, None]
    
    # Masked loss
    loss = self.criterion(logits[mask], labels[mask])
    return loss
```

**Recommendation:** Add this to the `Trainer` class or as a utility function.

### Issue 2: No Validation of Lengths

**Current code:**
```python
lengths = sequence_mask.sum(dim=1).cpu()
```

**Potential issue:** If all sequences have length 0, `pack_padded_sequence` will fail

**Suggested fix:**
```python
lengths = sequence_mask.sum(dim=1).cpu()
if (lengths == 0).any():
    raise ValueError("Found sequences with zero length. All sequences must have at least 1 visit.")
```

### Issue 3: No Explicit Documentation of Masking Requirements

**Current docstring:**
```python
def forward(
    self,
    visit_codes: torch.Tensor,
    visit_mask: Optional[torch.Tensor] = None,
    sequence_mask: Optional[torch.Tensor] = None,
    ...
):
```

**Recommendation:** Add explicit documentation:

```python
"""
Args:
    visit_codes: [batch_size, num_visits, max_codes_per_visit]
        Medical codes for each visit
    visit_mask: [batch_size, num_visits, max_codes_per_visit]
        Binary mask: 1 for real codes, 0 for padding codes
        IMPORTANT: Must be provided for variable-length visits
    sequence_mask: [batch_size, num_visits]
        Binary mask: 1 for real visits, 0 for padding visits
        IMPORTANT: Must be provided for variable-length sequences
        If not provided, assumes all visits are real (no padding)
"""
```

---

## Comparison with Best Practices

Let's check against the best practices from the tutorial:

| Best Practice | Our Implementation | Status |
|--------------|-------------------|--------|
| Use `pack_padded_sequence` | ✅ Lines 259-264 | ✅ Pass |
| Use `h_n[-1]` for final state | ✅ Line 278 | ✅ Pass |
| Mask losses for visit-level | ❌ Not implemented | ⚠️ Missing |
| Define labels causally | N/A (model-level) | - |
| Never process padding | ✅ Via packing | ✅ Pass |
| Validate lengths > 0 | ❌ Not checked | ⚠️ Missing |

**Overall:** 3/4 implemented correctly, 2 minor improvements needed

---

## Test Coverage

Our tests verify variable-length handling:

### Test: `test_variable_length_sequences`

```python
def test_variable_length_sequences(self, model_config):
    """Test model handles variable-length sequences."""
    # Create variable-length masks
    sequence_mask[0, :5] = 1   # 5 visits
    sequence_mask[1, :8] = 1   # 8 visits
    sequence_mask[2, :3] = 1   # 3 visits
    sequence_mask[3, :6] = 1   # 6 visits
    
    output = model(visit_codes, visit_mask, sequence_mask)
    
    assert output['logits'].shape == (batch_size, 1)
    assert not torch.isnan(output['logits']).any()
```

**Status:** ✅ Passing

This confirms that:
- Variable lengths are handled without errors
- No NaN values are produced
- Output shape is correct

---

## Recommendations

### Priority 1: Add Loss Masking Utility

Add to `src/ehrsequencing/training/trainer.py`:

```python
def compute_masked_loss(
    criterion: nn.Module,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor
) -> torch.Tensor:
    """
    Compute loss only on real timesteps, masking padding.
    
    Args:
        criterion: Loss function
        predictions: [batch_size, max_timesteps, ...] predictions
        labels: [batch_size, max_timesteps, ...] labels
        lengths: [batch_size] number of real timesteps per sequence
    
    Returns:
        Masked loss (scalar)
    """
    batch_size, max_timesteps = predictions.shape[:2]
    
    # Create mask: [batch_size, max_timesteps]
    mask = torch.arange(max_timesteps, device=predictions.device)[None, :] < lengths[:, None]
    
    # Apply mask and compute loss
    if predictions.dim() == 3:
        # Multi-class: [B, T, K]
        mask = mask.unsqueeze(-1).expand_as(predictions)
    
    masked_predictions = predictions[mask]
    masked_labels = labels[mask]
    
    return criterion(masked_predictions, masked_labels)
```

### Priority 2: Add Length Validation

Add to `LSTMBaseline.forward()`:

```python
if sequence_mask is not None:
    lengths = sequence_mask.sum(dim=1).cpu()
    
    # Validate
    if (lengths == 0).any():
        raise ValueError(
            f"Found {(lengths == 0).sum()} sequences with zero length. "
            "All sequences must have at least 1 visit."
        )
    
    packed_input = nn.utils.rnn.pack_padded_sequence(...)
```

### Priority 3: Enhance Documentation

Update docstrings to explicitly document masking requirements and behavior.

---

## Conclusion

**Our LSTM baseline correctly handles variable-length sequences** using the industry-standard `pack_padded_sequence` approach. The implementation:

✅ **Strengths:**
- Proper use of packed sequences
- Correct extraction of final hidden states
- Two-level masking (codes and visits)
- Bidirectional LSTM support
- Tested with variable lengths

⚠️ **Minor gaps:**
- No visit-level loss masking (only needed for many-to-many tasks)
- No length validation
- Documentation could be more explicit

**Overall assessment:** Production-ready for patient-level prediction tasks. Needs minor additions for visit-level prediction tasks.

---

**Related Documents:**
- Tutorial: `docs/methods/variable-length-sequences.md`
- Implementation: `src/ehrsequencing/models/lstm_baseline.py`
- Tests: `tests/test_lstm_model.py`
