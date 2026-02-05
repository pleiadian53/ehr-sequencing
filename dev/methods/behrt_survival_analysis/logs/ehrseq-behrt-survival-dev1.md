Prev: `/Users/pleiadian53/work/ehr-sequencing/dev/methods/logs/ehrseq-benchmarking-dev2.md`



# ✅ BEHRT for Survival Analysis - Complete Design

I've created a comprehensive design document at [dev/workflow/BEHRT_SURVIVAL_ANALYSIS_DESIGN.md](cci:7://file:///Users/pleiadian53/work/ehr-sequencing/dev/workflow/BEHRT_SURVIVAL_ANALYSIS_DESIGN.md:0:0-0:0) that answers your question.

---

## 📋 Summary: What's Needed

To use BEHRT for survival analysis and compare with LSTM models, you need to implement:

### **1. New Model: `BEHRTForSurvival`**

```python
# Architecture
BEHRT Encoder (pre-trained) 
  → Visit-level Aggregation 
  → Survival Prediction Head 
  → Hazards [batch, num_visits]
```

**Key components:**

- Load pre-trained BEHRT from MLM checkpoint
- Add survival prediction head (Linear + Sigmoid)
- Aggregate code-level representations to visit-level
- Output hazard at each visit

---

### **2. Data Adapter**

Convert between formats:

- **LSTM format:** Visit-grouped sequences
- **BEHRT format:** Flattened codes with visit boundaries

---

### **3. Training Pipeline**

**Two-stage approach:**

**Stage 1:** Pre-train BEHRT with MLM (already done)

```bash
python train_behrt_demo.py --realistic-data --epochs 100
```

**Stage 2:** Fine-tune for survival (new)

```bash
python train_behrt_survival.py \
    --pretrained-behrt checkpoints/behrt_mlm/best_model.pt \
    --task readmission \
    --freeze-encoder  # Optional
```

---

## 🎯 Why BEHRT Should Beat LSTM

1. **Pre-trained representations** (10-20% C-index gain expected)
   - BEHRT learns from large unlabeled data (MLM)
   - LSTM learns from scratch on small survival dataset

2. **Bidirectional context** (better representations)
   - BEHRT sees full sequence context
   - LSTM only sees past (causal)

3. **EHR-specific features** (better temporal modeling)
   - BEHRT uses age, visit, segment embeddings
   - LSTM only uses code embeddings

4. **Transfer learning** (faster convergence, less overfitting)
   - BEHRT transfers knowledge from pre-training
   - LSTM starts from random initialization

---

## 📁 Implementation Plan

### **Phase 1: Core Model (Week 1)**

- [ ] `src/ehrsequencing/models/behrt_survival.py` - BEHRTForSurvival class
- [ ] `src/ehrsequencing/data/behrt_survival_dataset.py` - Data adapter
- [ ] Visit aggregation logic (mean pooling)

### **Phase 2: Training (Week 2)**

- [ ] `examples/survival_analysis/train_behrt_survival.py` - Training script
- [ ] Load pre-trained BEHRT checkpoint
- [ ] Fine-tuning with LoRA support
- [ ] Evaluation metrics (C-index, calibration)

### **Phase 3: Benchmarking (Week 3)**

- [ ] `examples/survival_analysis/benchmark_survival_models.py` - Comparison
- [ ] BEHRT vs LSTM on readmission/mortality
- [ ] Ablation studies (frozen vs fine-tuned, with/without pre-training)
- [ ] Generate comparison reports

---

## 🔑 Key Design Decisions

### **Bidirectional vs Causal?**

✅ **Not a problem** - BEHRT generates representations (bidirectional OK), but survival head makes causal predictions (hazard at visit t uses only data through visit t)

### **Visit Aggregation?**

✅ **Mean pooling** (simple, effective) - aggregate BEHRT's code-level representations to visit-level

### **Fine-tuning Strategy?**

✅ **Three options:**

1. Freeze BEHRT, train only head (fast, less overfitting)
2. LoRA fine-tuning (efficient, good performance)
3. Full fine-tuning (best performance, slower)

---

## 📊 Comparison Framework

```python
# Models to compare
models = [
    DiscreteTimeSurvivalLSTM(),           # Baseline
    BEHRTForSurvival(frozen=True),        # Frozen encoder
    BEHRTForSurvival(lora=True),          # LoRA fine-tuning
    BEHRTForSurvival(full_finetune=True)  # Full fine-tuning
]

# Metrics
- C-index (discrimination)
- Calibration curves
- Time-dependent AUC (7, 14, 30 days)
- Training time, parameter count
```

---

The design document includes complete code examples, architecture diagrams, and implementation details. Ready to start implementing when you are! 🚀

**Markdown lints:** Cosmetic formatting warnings (blank lines, code block languages) - leaving as-is to focus on content.



---

Let's update our roadmap document: dev/workflow/ROADMAP.md

and move on to the next step, which I believe is the survival analysis using representation learning induced by BEHRT? The goal is to hopefully show that using BEHRT does have an advantage over LSTM-based models. 

---

The markdown lints are minor formatting issues. I'll move on and commit the changes.







## ✅ ROADMAP.md Updated - Phase 4 Active

### Summary

Successfully updated the project roadmap to reflect Phase 3 completion and transition to Phase 4 (BEHRT for Survival Analysis).

---

## 📋 What Changed

### Phase 3: COMPLETE (100%)

**Added accomplishments:**
- ✅ Transfer learning benchmarking (embedding fine-tuning, domain shift)
- ✅ Domain shift implementation with real distribution differences
- ✅ Fixed domain shift bug using deep copy approach
- ✅ Synthetic data refactored into `ehrsequencing.synthetic` package
- ✅ Comprehensive transfer learning tutorial documentation
- ✅ Updated benchmark scripts and testing roadmap

### Phase 4: IN PROGRESS (10%)

**Updated with detailed plan:**

**Key Question:**
> Does BEHRT's transformer-based representation learning provide advantages over LSTM-based models for survival analysis?

**Expected Advantages:**
- Better long-range dependency modeling
- More effective use of pre-trained embeddings
- Improved generalization through self-attention
- Faster convergence with transfer learning

**Implementation Tasks:**

1. **BEHRTForSurvival Model** (`src/ehrsequencing/models/behrt_survival.py`)
   - Extend BEHRT with survival prediction head
   - Visit-level aggregation from code-level representations
   - Support frozen/LoRA/full fine-tuning modes

2. **Dataset Adapter** (`src/ehrsequencing/data/behrt_survival_dataset.py`)
   - Convert visit-grouped sequences to BEHRT format
   - Maintain visit boundaries for aggregation
   - Generate survival labels

3. **Training Pipeline** (`examples/survival_analysis/train_behrt_survival.py`)
   - Load pre-trained BEHRT checkpoint
   - Multiple fine-tuning strategies
   - Evaluation with C-index, Brier score, calibration

4. **Benchmarking** (`examples/survival_analysis/benchmark_behrt_vs_lstm.py`)
   - Direct comparison on same data
   - Track performance, training time, convergence
   - Ablation studies

**Success Criteria:**
- ✅ Higher C-index than LSTM baseline
- ✅ Faster convergence with pre-training
- ✅ Better calibration
- ✅ LoRA achieves comparable performance with fewer parameters

---

## 🎯 Next Steps for Phase 4

You're correct - the next focus is **survival analysis using BEHRT's representation learning** to demonstrate advantages over LSTM-based models.

**Immediate Actions:**

1. **Implement BEHRTForSurvival** - Extend BEHRT with survival prediction head
2. **Create dataset adapter** - Convert existing survival data to BEHRT format
3. **Build training pipeline** - Support multiple fine-tuning strategies
4. **Run benchmarks** - Compare BEHRT vs LSTM on readmission/mortality tasks
5. **Document results** - Show clear advantages of transformer-based approach

The architecture is already designed (see `dev/workflow/BEHRT_SURVIVAL_ANALYSIS_DESIGN.md`), so we can proceed directly to implementation.

Ready to start implementing BEHRTForSurvival? 🚀



---

Before we move on to implementating BEHRT-based survival analysis workflow, can you write a tutorial document under docs/  on discrete-time survival analysis and how do we plan to compare BEHRT vs LSTM. 

In particular, please explain: 

1. how the label dataset is prepared (what are the labels)
2. how the loss is computed and what the optimization objective is. 


Potential locations to keep such documents: 

docs/applications/<topic> 
docs/<topic> 

The bottom line is that we want to keep the documents under a proper category for easy referencing. 

---



The markdown lints are minor formatting issues. I'll move on and commit the comprehensive tutorial document.







## ✅ Comprehensive Survival Analysis Tutorial Created

### Summary

Successfully created a detailed tutorial document on discrete-time survival analysis for EHR sequences, covering all requested topics.

---

## 📄 Document Location

**[docs/methods/discrete-time-survival-analysis.md](cci:7://file:///Users/pleiadian53/work/ehr-sequencing/docs/methods/discrete-time-survival-analysis.md:0:0-0:0)**

This follows the existing structure where [docs/methods/](cci:9://file:///Users/pleiadian53/work/ehr-sequencing/docs/methods:0:0-0:0) contains detailed methodological documentation.

---

## 📚 Content Overview

### 1. Label Preparation (Detailed)

**Dataset Structure:**
- Visit sequences with medical codes
- Event time (visit index when event occurred)
- Event indicator (1 = event, 0 = censored)

**Example with Real Data:**
```python
# Patient A (Readmission Event)
{
    'visits': [...],
    'event_time': 2,        # Readmitted at visit 2
    'event_indicator': 1    # Event occurred
}

# Patient B (Censored)
{
    'visits': [...],
    'event_time': 3,        # Last observed visit
    'event_indicator': 0    # No event (censored)
}
```

**Label Generation Process:**
1. Define event criteria (e.g., readmission within 30 days)
2. Identify event time (first visit where event occurs)
3. Create labels tensor for batch training

**Synthetic Labels:**
- Risk score computation based on comorbidities, visit frequency, code diversity
- Controlled risk-time correlation (r = -0.5)
- Realistic censoring (30%)

### 2. Loss Function and Optimization (Detailed)

**Negative Log-Likelihood:**

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \delta_i \log h_{t_i} + (1-\delta_i) \log S(t_i) \right]$$

**Loss Components:**
- **Event patients:** Maximize hazard at event time
- **Censored patients:** Maximize survival probability up to censoring

**Complete Implementation:**
- Full Python code with numerical stability (epsilon clamping)
- Masking for variable-length sequences
- Batch processing

**Optimization:**
- Adam optimizer with learning rate scheduling
- Gradient clipping (max_norm=1.0)
- Early stopping on validation C-index

### 3. BEHRT vs LSTM Comparison Framework

**Experimental Design:**
- Same data, same task, fair comparison
- Three BEHRT fine-tuning strategies: frozen, LoRA, full
- Comprehensive metrics: C-index, Brier score, training time, parameters

**Expected Results:**
```
LSTM Baseline:      C-index 0.70-0.75
BEHRT (Frozen):     C-index 0.72-0.77  (+2-5%)
BEHRT (LoRA):       C-index 0.75-0.80  (+5-10%)
BEHRT (Full):       C-index 0.76-0.82  (+8-12%)
```

**Four Hypotheses:**
1. Pre-training improves performance
2. BEHRT outperforms LSTM (5-10% higher C-index)
3. LoRA is parameter-efficient (80-90% fewer parameters)
4. BEHRT generalizes better (smaller train-val gap)

**Benchmark Script:**
```bash
python examples/survival_analysis/benchmark_behrt_vs_lstm.py \
    --num-patients 10000 \
    --epochs 100 \
    --output-dir experiments/behrt_vs_lstm/
```

### 4. Additional Content

- **Model architectures** with complete forward pass implementations
- **Data preparation** for BEHRT format (flattened sequences with visit boundaries)
- **Evaluation metrics** (C-index, Brier score, calibration plots)
- **Best practices** (hyperparameters, avoiding overfitting, clinical validation)
- **Implementation details** (variable-length sequences, early stopping)

---

## 🎯 Ready for Phase 4 Implementation

The tutorial provides everything needed to implement BEHRTForSurvival:

1. ✅ **Clear label format** - Event time + event indicator
2. ✅ **Loss function** - Complete implementation with code
3. ✅ **Model architecture** - BEHRT encoder + visit aggregation + hazard head
4. ✅ **Comparison framework** - Fair benchmarking approach
5. ✅ **Evaluation metrics** - C-index, Brier score, calibration

**Next Steps:**
1. Implement `src/ehrsequencing/models/behrt_survival.py`
2. Implement `src/ehrsequencing/data/behrt_survival_dataset.py`
3. Create `examples/survival_analysis/train_behrt_survival.py`
4. Create `examples/survival_analysis/benchmark_behrt_vs_lstm.py`
5. Run experiments and validate BEHRT advantages

Ready to start implementing BEHRTForSurvival? 🚀

---

@survival_analysis.md#L127-139 

As a follow-up question, this loss function is reasonable but do you think optimizing the model wrt to this negative log-likelihood on hazards directly optimizes toward increasing C-index? 

---

This is an excellent theoretical question about the relationship between the loss function and the evaluation metric.

## Short Answer

**No, optimizing negative log-likelihood (NLL) does not directly optimize C-index.** They measure different things:

- **NLL** measures **calibration** - how well predicted probabilities match true probabilities
- **C-index** measures **discrimination** - how well the model ranks patients by risk

## The Disconnect

### What NLL Optimizes

The negative log-likelihood loss:

$$\mathcal{L} = -\sum_{i=1}^{N} \left[ \delta_i \log h_{t_i} + (1-\delta_i) \log S(t_i) \right]$$

This penalizes:
1. **Low hazard predictions** when events occur (wants $h_{t_i} \to 1$)
2. **Low survival predictions** when censored (wants $S(t_i) \to 1$)

**Goal:** Accurate probability estimates (calibration)

### What C-index Measures

$$C = \frac{\text{# concordant pairs}}{\text{# comparable pairs}}$$

A pair $(i, j)$ is concordant if:
- Patient $i$ has event earlier than $j$ AND
- Patient $i$ has higher predicted risk than $j$

**Goal:** Correct relative ordering (discrimination)

### Why They're Different

**Example where NLL is good but C-index is bad:**

```
Patient A: True event at t=2, Predicted risk = 0.51
Patient B: True event at t=5, Predicted risk = 0.50
Patient C: Censored at t=10, Predicted risk = 0.49
```

- **NLL:** All predictions are well-calibrated (~0.5 matches true event rates)
- **C-index:** Poor! Risks are too similar to discriminate between patients

**Example where C-index is good but NLL is bad:**

```
Patient A: True event at t=2, Predicted risk = 0.9
Patient B: True event at t=5, Predicted risk = 0.6
Patient C: Censored at t=10, Predicted risk = 0.1
```

- **C-index:** Perfect! Correct ordering (A > B > C)
- **NLL:** Poor if true event rates are ~0.3 (predictions are overconfident)

## Why We Still Use NLL

Despite not directly optimizing C-index, NLL is the standard loss because:

### 1. Probabilistic Framework

NLL is derived from maximum likelihood estimation - a principled statistical approach:

$$\theta^* = \arg\max_{\theta} P(\text{data} | \theta) = \arg\min_{\theta} -\log P(\text{data} | \theta)$$

### 2. Differentiable

NLL has smooth gradients, enabling gradient-based optimization:

$$\frac{\partial \mathcal{L}}{\partial h_t} = -\frac{\delta}{h_t} + \frac{1-\delta}{1-h_t}$$

C-index is **not differentiable** - it's based on counting concordant pairs (discrete operation).

### 3. Calibration Matters Clinically

Well-calibrated predictions are crucial for clinical decision-making:
- "This patient has 20% risk of readmission" → actionable
- "This patient ranks 5th out of 100" → less actionable

### 4. Empirical Correlation

In practice, models with lower NLL **tend to** have higher C-index, though not guaranteed.

## Alternative: Ranking Losses

If you want to **directly optimize C-index**, you could use ranking losses:

### 1. Pairwise Ranking Loss

```python
def pairwise_ranking_loss(risk_scores, event_times, event_indicators):
    """
    Penalize incorrectly ordered pairs
    """
    loss = 0
    n_pairs = 0
    
    for i in range(len(risk_scores)):
        for j in range(len(risk_scores)):
            # Only compare if i had event and occurred before j
            if event_indicators[i] == 1 and event_times[i] < event_times[j]:
                # i should have higher risk than j
                margin_loss = torch.relu(risk_scores[j] - risk_scores[i] + margin)
                loss += margin_loss
                n_pairs += 1
    
    return loss / n_pairs if n_pairs > 0 else 0
```

### 2. Cox Partial Likelihood

The Cox proportional hazards model uses a ranking-based loss:

$$\mathcal{L}_{\text{Cox}} = -\sum_{i: \delta_i=1} \left[ \log \frac{\exp(r_i)}{\sum_{j: t_j \geq t_i} \exp(r_j)} \right]$$

This directly optimizes ranking by comparing each event to all at-risk patients.

### 3. Hybrid Approach

Combine NLL (calibration) + ranking loss (discrimination):

```python
loss = nll_loss + lambda_rank * ranking_loss
```

## Practical Recommendations

### For Your BEHRT vs LSTM Comparison

**Stick with NLL loss** because:

1. ✅ **Standard approach** - Comparable to literature
2. ✅ **Stable training** - Smooth gradients, well-understood
3. ✅ **Calibration important** - Clinical predictions need probabilities
4. ✅ **C-index as evaluation** - Use C-index to measure discrimination, not as loss

### Monitor Both Metrics

During training, track:
- **Training loss (NLL)** - Optimization objective
- **Validation C-index** - Discrimination performance
- **Calibration plots** - Predicted vs observed event rates

### If C-index Plateaus

If validation C-index stops improving while NLL decreases:
1. **Check calibration** - Model may be overconfident
2. **Try ranking loss** - Add pairwise ranking term
3. **Adjust learning rate** - May be stuck in local minimum
4. **Check data quality** - Ensure sufficient events and variation

## Summary

| Aspect                  | NLL Loss                           | C-index                    |
| ----------------------- | ---------------------------------- | -------------------------- |
| **Measures**            | Calibration (probability accuracy) | Discrimination (ranking)   |
| **Differentiable**      | ✅ Yes                              | ❌ No                       |
| **Direct optimization** | ✅ Yes                              | ❌ No (use as evaluation)   |
| **Clinical value**      | High (actionable probabilities)    | High (risk stratification) |
| **Relationship**        | Correlated but not equivalent      | -                          |

**Bottom line:** Use NLL as your loss function (standard, stable, principled), but evaluate with C-index (measures what you care about - discrimination). They measure complementary aspects of model quality.



---

I've grouped discrete time survival analysis-related documents under /Users/pleiadian53/work/ehr-sequencing/docs/methods/discrete_time_survival_analysis/ 

Other existing documents under docs/methods/ may be related as well. 

I added your answer to my question on NLL loss vs C-index under docs/methods/discrete_time_survival_analysis/pairwise_ranking_loss.md

pairwise_ranking_loss.md is a temporary docuement. Please help me rewrite it into a tutorial document with relevant details. 

---

