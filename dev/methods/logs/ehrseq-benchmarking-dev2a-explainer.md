Some of the code had new updates. 

Please look into dev/models/pretrain_finetune/ to see if the tutorial documents need to be updated accordingly. 



Additional quesitons: 

1. Is BEHRT autoregressive?
2. Can you look into examples/pretrain_finetune/train_behrt_demo.py and 

examples/pretrain_finetune/train_behrt_finetune.py to see what the training objectives are? I think they all focus on discrete-time survival analysis

---

I'll investigate the training scripts and check if the documentation needs updates.

[6 tools called]

Now I have a complete picture. Let me answer your questions and provide documentation updates:

## Answering Your Questions

### 1. **Is BEHRT autoregressive?**

**NO.** BEHRT is **bidirectional**, not autoregressive. Here's why:

```python
# From train_behrt_demo.py, line 295-306
model = BEHRTForMLM(config).to(device)  # Uses MLM (Masked Language Modeling)

# MLM training objective
logits, loss = model(masked_codes, ages, visit_ids, attention_mask, labels)
```

**Key characteristics:**

| Feature      | BEHRT                | GPT (Autoregressive)   |
| ------------ | -------------------- | ---------------------- |
| Attention    | **Bidirectional**    | Causal (left-to-right) |
| Context      | Sees entire sequence | Only sees past tokens  |
| Masking      | Random masking       | Causal mask            |
| Objective    | MLM (predict masked) | Next token prediction  |
| Architecture | BERT-style           | GPT-style              |

**Why BEHRT is bidirectional:**
- Uses `nn.TransformerEncoder` (not decoder)
- No causal masking in attention
- Masked tokens can attend to both past AND future context
- Predicts randomly masked tokens, not next token

**In code:**
```python
# BEHRT sees full context (bidirectional)
sequence = [code1, [MASK], code3, code4, code5]
# When predicting [MASK], attends to: code1, code3, code4, code5

# Autoregressive (GPT-style) only sees past
sequence = [code1, code2, [PREDICT]]
# When predicting next token, attends to: code1, code2 only
```

### 2. **Training Objectives**

**You are INCORRECT** - these scripts do **NOT** focus on discrete-time survival analysis. Here's what they actually do:

#### `train_behrt_demo.py`
**Objective: Masked Language Modeling (MLM)**

```python
# Line 4: "4. MLM pre-training objective"
model = BEHRTForMLM(config)  # MLM head

# Training loop (lines 90-117)
masked_codes, ages, visit_ids, attention_mask, labels = batch
logits, loss = model(masked_codes, ages, visit_ids, attention_mask, labels)
# Predicts original code from masked token
```

**What it does:**
- Randomly masks 15% of medical codes
- Model predicts original codes from context
- Self-supervised pre-training (like BERT)
- NO survival analysis, NO time-to-event prediction

#### `train_behrt_finetune.py`
**Objective: Also MLM (with pretrained embeddings)**

```python
# Line 2: "BEHRT Fine-tuning Script with Pre-trained Embeddings"
# Line 8: "Requires fewer patients (1K-10K vs 100K+)"
model = BEHRTForMLM(config)  # Still MLM

# Lines 94-98: Same MLM training
logits, loss = model(masked_codes, ages, visit_ids, attention_mask, labels)
```

**What it does:**
- Loads pretrained embeddings (Med2Vec, Word2Vec)
- Fine-tunes BEHRT with frozen embeddings
- Still uses MLM objective
- NO survival analysis

### Where is Survival Analysis?

**Survival analysis is in a DIFFERENT directory:**

```
examples/survival_analysis/
├── train_lstm_demo.py          # ← Survival analysis with LSTM
├── train_lstm.py                # ← Survival analysis with LSTM
└── README.md                    # "Discrete-time survival analysis"
```

**Key difference:**
- `examples/pretrain_finetune/` → **BEHRT with MLM** (self-supervised)
- `examples/survival_analysis/` → **LSTM with survival objective** (time-to-event)

## Documentation Updates Needed

Let me create an update document:

[1 tool called]

Now let me update the README to include the clarifications document:

[4 tools called]

Perfect! I've completed my investigation and documentation updates. Here's my comprehensive summary:

## 📋 Summary of Findings

### Your Questions Answered

#### 1. **Is BEHRT autoregressive?**

**❌ NO.** BEHRT is **bidirectional** (like BERT, not GPT).

**Evidence:**
- Uses `nn.TransformerEncoder` (bidirectional attention)
- No causal masking in the attention mechanism
- MLM objective requires seeing full context (past + future)
- When predicting a masked token, it attends to ALL other tokens

**Key distinction:**
- **BEHRT (bidirectional):** `[code1, code2, [MASK], code4, code5]` → attends to all 5 positions
- **GPT (autoregressive):** `[code1, code2, code3, [PREDICT]]` → only attends to positions 1-3

#### 2. **Do the training scripts focus on survival analysis?**

**❌ NO.** They focus on **MLM (Masked Language Modeling)**, not survival analysis.

**What I found:**

| Script                                          | Objective    | Model          | Purpose         |
| ----------------------------------------------- | ------------ | -------------- | --------------- |
| `train_behrt_demo.py`                           | **MLM**      | `BEHRTForMLM`  | Pre-training    |
| `train_behrt_finetune.py`                       | **MLM**      | `BEHRTForMLM`  | Fine-tuning     |
| `examples/survival_analysis/train_lstm_demo.py` | **Survival** | `SurvivalLSTM` | Risk prediction |

**The confusion:**
- BEHRT scripts are in `examples/pretrain_finetune/` (MLM objective)
- Survival analysis scripts are in `examples/survival_analysis/` (LSTM models)
- **These are completely separate pipelines!**

---

## 📝 Documentation Updates

I've created a comprehensive clarification document and updated the documentation structure:

### New Document Created

**`04_clarifications_and_corrections.md`** (530+ lines)
- ⚠️ Addresses the two major misconceptions
- Detailed comparison: BEHRT vs GPT (bidirectional vs autoregressive)
- Detailed comparison: BEHRT MLM vs LSTM survival analysis
- Code examples showing why BEHRT is bidirectional
- Explains what BEHRT actually does vs what it doesn't do

### Updated Documents

**`README.md`** - Updated to include:
- Link to new clarifications document
- Added to "Quick Navigation" section
- Added to "Questions?" section
- Updated document history

---

## 🔑 Key Takeaways

### ✅ What BEHRT IS

1. **Bidirectional transformer** (like BERT)
2. Uses **MLM pre-training** (predict masked codes)
3. Trained with `BEHRTForMLM` model
4. Self-supervised learning from EHR sequences
5. Produces contextualized representations

### ❌ What BEHRT is NOT

1. ~~Autoregressive~~ → It's bidirectional
2. ~~Uses causal masking~~ → Full bidirectional attention
3. ~~Does survival analysis~~ → Does MLM
4. ~~Predicts time-to-event~~ → Predicts masked codes
5. ~~Uses survival loss~~ → Uses CrossEntropy loss

### 🔬 Current Codebase Structure

```
Pre-training (BEHRT):
└── examples/pretrain_finetune/
    ├── train_behrt_demo.py          ← MLM objective
    ├── train_behrt_finetune.py      ← MLM with pretrained embeddings
    └── [Survival analysis: NOT HERE]

Survival Analysis (LSTM):
└── examples/survival_analysis/
    ├── train_lstm_demo.py           ← Discrete-time survival
    ├── train_lstm.py                ← Discrete-time survival
    └── README.md                    ← "survival analysis"
```

---

## 🎯 Documentation Status

All documents in `dev/models/pretrain_finetune/`:

1. ✅ **01_behrt_model_design.md** - Complete (no changes needed)
2. ✅ **02_training_guide.md** - Complete (no changes needed)
3. ✅ **03_pretrained_embeddings_workflow.md** - Complete (no changes needed)
4. ✅ **04_clarifications_and_corrections.md** - **NEW** (addresses your questions)
5. ✅ **README.md** - Updated (includes new document)

The original documentation was accurate - the confusion stemmed from misconceptions about BEHRT's architecture and training objectives, which I've now clarified in document #4.

---

## 💡 If You Want BEHRT for Survival Analysis

Currently, BEHRT only does MLM pre-training. To use it for survival analysis, you would need to:

1. Create `BEHRTForSurvival` class (similar to `BEHRTForSequenceClassification`)
2. Implement survival loss function
3. Create new training script `train_behrt_survival.py`
4. Use survival-specific metrics (C-index)

This would be a future extension - currently only LSTM models do survival analysis in this codebase.

---

