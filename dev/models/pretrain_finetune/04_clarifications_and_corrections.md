# BEHRT Clarifications and Corrections

**Last Updated:** 2026-02-03  
**Purpose:** Address common misconceptions about BEHRT

---

## Critical Clarifications

### ❌ COMMON MISCONCEPTION #1: BEHRT is Autoregressive

**FALSE.** BEHRT is **bidirectional**.

#### Key Differences

| Aspect | BEHRT (Bidirectional) | GPT (Autoregressive) |
|--------|----------------------|----------------------|
| **Architecture** | TransformerEncoder | TransformerDecoder |
| **Attention** | Full bidirectional | Causal (masked) |
| **Training** | MLM (mask random) | Next token prediction |
| **Context** | Past + Future | Past only |
| **Use case** | Representation learning | Generation |

#### Why BEHRT is Bidirectional

**1. Uses TransformerEncoder (not Decoder)**

```python
# From src/ehrsequencing/models/behrt.py:130-143
encoder_layer = nn.TransformerEncoderLayer(
    d_model=config.hidden_dim,
    nhead=config.num_heads,
    batch_first=True,
    norm_first=True
)
self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
```

**No causal masking!** The encoder allows each token to attend to all other tokens.

**2. MLM Objective Requires Bidirectional Context**

```python
# From examples/pretrain_finetune/train_behrt_demo.py:101
logits, loss = model(masked_codes, ages, visit_ids, attention_mask, labels)

# Example sequence:
# Original:   [250, 401, 780, 357, 560]
# Masked:     [250, [M], 780, [M], 560]
# Labels:     [-100, 401, -100, 357, -100]

# When predicting position 1 (401):
# Attends to: [250, X, 780, X, 560]  ← Can see future context!
```

**3. Attention Mask is for Padding, Not Causality**

```python
# From src/ehrsequencing/models/behrt.py:175-179
if attention_mask is not None:
    # Convert from (1=valid, 0=padding) to (True=padding, False=valid)
    src_key_padding_mask = ~attention_mask.bool()
    # This masks PADDING, not future tokens
```

#### Comparison: BEHRT vs GPT

**BEHRT Forward Pass:**
```python
# Sequence: [code1, code2, code3, code4, code5]
# All tokens can attend to all other tokens:
# code1 attends to: [code1, code2, code3, code4, code5]
# code2 attends to: [code1, code2, code3, code4, code5]
# code3 attends to: [code1, code2, code3, code4, code5]
# ...
```

**GPT Forward Pass (Autoregressive):**
```python
# Sequence: [code1, code2, code3, code4, code5]
# Each token only attends to previous tokens:
# code1 attends to: [code1]
# code2 attends to: [code1, code2]
# code3 attends to: [code1, code2, code3]
# code4 attends to: [code1, code2, code3, code4]
# code5 attends to: [code1, code2, code3, code4, code5]
```

#### When Would BEHRT Be Autoregressive?

**If we wanted autoregressive BEHRT, we would need to:**

1. Replace `TransformerEncoder` with `TransformerDecoder`
2. Add causal masking:
   ```python
   causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
   ```
3. Change objective from MLM to next-token prediction
4. Change model from `BEHRTForMLM` to `BEHRTForGeneration`

**But this is NOT what the current implementation does!**

---

### ❌ COMMON MISCONCEPTION #2: BEHRT Training Uses Survival Analysis

**FALSE.** BEHRT uses **Masked Language Modeling (MLM)**, NOT survival analysis.

#### What BEHRT Actually Does

**Training Objective: MLM (Self-Supervised Pre-training)**

```python
# From examples/pretrain_finetune/train_behrt_demo.py:4
# "4. MLM pre-training objective"

# Line 295
model = BEHRTForMLM(config)  # MLM head, not survival head

# Lines 101-104
logits, loss = model(masked_codes, ages, visit_ids, attention_mask, labels)
loss.backward()
optimizer.step()
```

**What MLM does:**
1. Takes sequence: `[250, 401, 780, 357, 560]`
2. Randomly masks 15%: `[250, [M], 780, [M], 560]`
3. Model predicts original codes
4. Loss: CrossEntropy between prediction and true code

**NOT:**
- Time-to-event prediction
- Survival curves
- Hazard ratios
- C-index optimization

#### Where is Survival Analysis?

**Survival analysis is in a DIFFERENT directory with DIFFERENT models:**

```
examples/survival_analysis/          ← LSTM models
├── train_lstm_demo.py              ← Discrete-time survival
├── train_lstm.py                   ← Discrete-time survival
└── README.md                       ← "discrete-time survival analysis"

examples/pretrain_finetune/         ← BEHRT models
├── train_behrt_demo.py             ← MLM pre-training
├── train_behrt_finetune.py         ← MLM with pretrained embeddings
└── README.md                       ← "Pre-training and fine-tuning"
```

#### BEHRT vs LSTM: Different Objectives

| Aspect | BEHRT (MLM) | LSTM (Survival) |
|--------|-------------|-----------------|
| **Model** | `BEHRTForMLM` | `SurvivalLSTM` |
| **Objective** | Predict masked codes | Predict time-to-event |
| **Loss** | CrossEntropyLoss | Negative log-likelihood |
| **Output** | Logits over vocab | Hazard probabilities |
| **Metric** | Accuracy, F1, Perplexity | C-index, Brier score |
| **Use case** | Representation learning | Risk prediction |

#### Could BEHRT Do Survival Analysis?

**Yes, but it would require:**

1. **Different model head:**
   ```python
   # Instead of BEHRTForMLM
   class BEHRTForSurvival(nn.Module):
       def __init__(self, config, num_time_bins):
           self.behrt = BEHRT(config)
           self.survival_head = nn.Sequential(
               nn.Linear(config.hidden_dim, config.hidden_dim),
               nn.GELU(),
               nn.Linear(config.hidden_dim, num_time_bins)
           )
   ```

2. **Different loss function:**
   ```python
   # Negative log-likelihood for discrete survival
   def survival_loss(hazard_probs, event_times, event_indicators):
       # ... survival analysis loss
   ```

3. **Different training script:**
   ```python
   # examples/survival_analysis/train_behrt_survival.py
   model = BEHRTForSurvival(config, num_time_bins=10)
   loss = survival_loss(hazard_probs, event_times, event_indicators)
   ```

**But this does NOT exist yet in the codebase!**

---

## Correct Understanding

### BEHRT Training Pipeline (Current Implementation)

**Phase 1: Pre-training (Unsupervised)**
```
Input: EHR sequences
  ↓
Mask 15% of codes randomly
  ↓
BEHRTForMLM predicts masked codes
  ↓
Loss: CrossEntropyLoss (MLM)
  ↓
Output: Pre-trained representations
```

**Phase 2: Fine-tuning (Supervised) - FUTURE WORK**

Could be adapted for:
- Classification: `BEHRTForSequenceClassification` (exists in code)
- Next visit prediction: `BEHRTForNextVisitPrediction` (exists in code)
- **Survival analysis: `BEHRTForSurvival` (does NOT exist yet)**

### Current Codebase Structure

```
Pre-training (BEHRT):
- examples/pretrain_finetune/train_behrt_demo.py       ← MLM objective
- examples/pretrain_finetune/train_behrt_finetune.py   ← MLM with embeddings
- src/ehrsequencing/models/behrt.py                    ← BEHRTForMLM

Survival Analysis (LSTM):
- examples/survival_analysis/train_lstm_demo.py        ← Survival objective
- examples/survival_analysis/train_lstm.py             ← Survival objective
- src/ehrsequencing/models/survival_lstm.py            ← SurvivalLSTM

Classification/Prediction (BEHRT):
- src/ehrsequencing/models/behrt.py                    ← BEHRTForSequenceClassification
                                                       ← BEHRTForNextVisitPrediction
                                                       ← (No training scripts yet)
```

---

## Summary

### ✅ Correct Statements

1. BEHRT is **bidirectional** (like BERT)
2. BEHRT training uses **MLM** (Masked Language Modeling)
3. Current BEHRT scripts do **NOT** do survival analysis
4. Survival analysis is done with **LSTM models** in separate directory
5. BEHRT uses `TransformerEncoder` (bidirectional attention)

### ❌ Incorrect Statements

1. ~~BEHRT is autoregressive~~ → It's bidirectional
2. ~~BEHRT training focuses on survival analysis~~ → It focuses on MLM
3. ~~train_behrt_demo.py does survival analysis~~ → It does MLM pre-training
4. ~~BEHRT uses causal masking~~ → It uses full bidirectional attention

### 🔮 Future Possibilities

**To make BEHRT do survival analysis, we would need to:**

1. Create `BEHRTForSurvival` model class
2. Implement survival loss function
3. Create training script `train_behrt_survival.py`
4. Use survival-specific metrics (C-index)

**This would be similar to how `BEHRTForSequenceClassification` extends the base BEHRT model for classification tasks.**

---

## References

**BEHRT Model:**
- `src/ehrsequencing/models/behrt.py` - BEHRT implementation
- Lines 130-143: TransformerEncoder (bidirectional)
- Lines 246-305: BEHRTForMLM (MLM objective)

**Training Scripts:**
- `examples/pretrain_finetune/train_behrt_demo.py` - MLM pre-training
- `examples/pretrain_finetune/train_behrt_finetune.py` - MLM fine-tuning
- `examples/survival_analysis/train_lstm_demo.py` - LSTM survival analysis

**Related Documentation:**
- `01_behrt_model_design.md` - BEHRT architecture (needs update)
- `02_training_guide.md` - Training workflows (needs update)
- `examples/survival_analysis/README.md` - Survival analysis overview

---

**Last Updated:** 2026-02-03  
**Status:** Clarifications based on code review
