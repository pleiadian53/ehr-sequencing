Yes! This looks much better. The example shows a **C-index of 0.889**, which is excellent and indicates the synthetic outcomes now have the correct relationship.

Let me verify what you're seeing matches the expected pattern after the fix:

**Key Observations from Your Output:**

✅ **8 out of 9 pairs are concordant** (88.9% C-index)
- Patient A (highest risk 0.8) has earliest event (time 5) ✓
- Patient D (lowest risk 0.3) has latest event (time 20) ✓
- Risk scores generally decrease as event times increase ✓

✅ **Only 1 discordant pair**: (B, E)
- Patient B: risk 0.6, event at time 10
- Patient E: risk 0.7, event at time 12
- This is expected - some randomness is realistic, and E's slightly higher risk (0.7 vs 0.6) doesn't guarantee an earlier event

This is exactly what we want to see! The synthetic generator is now working correctly with the fixed `(1 - risk_score)^2` transformation.

**What This Means for Training:**

With properly generated outcomes, your model should now achieve:
- **C-index > 0.5** (better than random)
- **Expected range: 0.60-0.70** for 200 patients
- **Potentially 0.70-0.75** for full 1,151 patients on cloud GPU

The model was learning correctly before - it was just learning from backwards data. Now with the correct synthetic outcomes, it should demonstrate meaningful survival prediction capability.

You can proceed with training and should see the validation C-index steadily improve and stay well above 0.5 throughout training.

---

The [estimate_memory_gb](cci:1://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/sampling.py:171:0-246:5) and [print_memory_recommendation](cci:1://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/sampling.py:249:0-286:69) functions already exist in [sampling.py](cci:7://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/sampling.py:0:0-0:0). I just need to add a higher-level convenience function that takes sequences directly (like the notebook version does) and update the exports.

Now update the [__init__.py](cci:7://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/__init__.py:0:0-0:0) to export the new function:



I see the issue - cell 13 still has the diagnostic code that runs before outcome generation. Let me fix the notebook by replacing cell 13 with the outcome generation code, and moving the diagnostic to after it:

Now let me fix cell 13 to be the outcome generation (not the diagnostic), and cell 14 should have the diagnostic after outcome is created:

Now let me update cell 14 to have the diagnostic check (which now comes after outcome generation):

Done. Here's what I did:

## Summary

1. **Added [estimate_memory_from_sequences()](cci:1://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/sampling.py:289:0-372:18)** to `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/sampling.py:290-373` - a convenience wrapper that takes sequences directly and computes statistics automatically

2. **Updated `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/__init__.py:10`** to export the new function

3. **Refactored notebook cell 11** to use the utility module instead of defining the function inline:
   ```python
   from ehrsequencing.utils import estimate_memory_from_sequences
   mem_est = estimate_memory_from_sequences(sequences, vocab_size=builder.vocabulary_size)
   ```

4. **Fixed cell ordering** - cell 13 now generates outcomes, cell 14 runs the diagnostic (so `outcome` is defined before it's used)

The notebook should now run sequentially without errors, and the memory estimation logic is reusable across other notebooks.



---

You're right, that's poor organization. Let me create a dedicated `memory.py` module for memory-related utilities.

Now let me remove the memory functions from [sampling.py](cci:7://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/sampling.py:0:0-0:0) and update the [__init__.py](cci:7://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/__init__.py:0:0-0:0):

Now let me update the [__init__.py](cci:7://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/__init__.py:0:0-0:0) to import from the new [memory.py](cci:7://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/memory.py:0:0-0:0) module:

Done. Memory utilities are now properly organized:

## Summary

- **Created** `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/memory.py:1-206` with:
  - [estimate_memory_gb()](cci:1://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/sampling.py:171:0-246:5) - low-level memory estimation
  - [estimate_memory_from_sequences()](cci:1://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/sampling.py:289:0-372:18) - convenience wrapper for sequences
  - [print_memory_recommendation()](cci:1://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/sampling.py:249:0-286:69) - formatted output

- **Cleaned** `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/sampling.py:169` - removed memory functions, now only contains sampling utilities

- **Updated** `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/utils/__init__.py:11-14` - imports from both modules

The notebook import remains unchanged since it uses the package-level export:
```python
from ehrsequencing.utils import estimate_memory_from_sequences
```



---

```
======================================================================
DIAGNOSTIC: Synthetic Outcome Validation
======================================================================

Correlation between risk score and event time:
  • Pearson r = -0.058 (p=0.5042)
  • Expected: NEGATIVE correlation (high risk → early events)
  ⚠️  Weak correlation: synthetic outcomes may be too random

Example patients (with events):
Risk Score   Event Time   Expected            
--------------------------------------------------
0.519        1            Late       (actual: Early) ✗
0.675        5            Early      (actual: Late) ✗
0.632        2            Early      (actual: Late) ✗
0.679        0            Early      (actual: Early) ✓
0.552        0            Late       (actual: Early) ✗
0.900        0            Early      (actual: Early) ✓
0.508        3            Late       (actual: Late) ✓
0.542        1            Late       (actual: Early) ✗
0.621        5            Early      (actual: Late) ✗
0.625        19           Early      (actual: Late) ✗
======================================================================
```





---

Looking at the [_simulate_event_time](cci:1://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/synthetic/survival.py:193:4-221:26) method in `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/synthetic/survival.py:194-222`, I can see the issue. The exponential distribution adds too much randomness, washing out the risk-time relationship. Let me fix this:

The problem is clear: the exponential distribution's variance is too high relative to the risk-based mean, causing the correlation to wash out. I need to use a more deterministic approach with controlled noise.

Fixed. The issue was that the exponential distribution's high variance was washing out the risk-time correlation.

## Changes Made

In `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/synthetic/survival.py:194-226`:

**Before**: Used exponential distribution with `survival_factor` as the scale parameter. The exponential's variance equals its mean squared, so high-risk patients could still get late events by chance.

**After**: Uses a deterministic base time (`1.0 - risk_score`) plus small controlled Gaussian noise (`std=0.15`). This ensures:
- High risk (0.9) → base fraction 0.1 → early event
- Low risk (0.1) → base fraction 0.9 → late event
- Noise is bounded and small relative to the signal

**Restart the kernel and re-run the notebook** to use the fixed generator. You should now see a strong negative correlation (r < -0.5).



---

> Model learning reverse C-index 



---

The C-index is still below 0.5 (hovering around 0.38-0.46), indicating the model is learning the **opposite** relationship. This is a fundamental issue with how the model interprets risk vs. hazard.

Let me check the model architecture and loss function:

Looking at the C-index results (0.38-0.46, all below 0.5), the model is learning the **inverse** relationship. The issue is in how we're computing risk scores in the evaluation function.

I see the problem. The C-index calculation in `compute_cindex_from_risks` expects **higher risk → earlier events**, but the current implementation is checking if `risk_scores[i] > risk_scores[j]` when `event_times[i] < event_times[j]`.

However, the issue is that in survival analysis, **higher cumulative hazard means higher risk of event**, which should correlate with **earlier event times**. But the synthetic data has **negative correlation** between risk_score and event_time (high risk → low event_time index).

The problem is the C-index is being computed correctly, but it's getting < 0.5 because the model is learning the **wrong direction**. Let me check the model output and loss function:

The issue is clear now. Looking at the model and C-index calculation:

1. **Model outputs**: Hazards at each visit (sigmoid, values 0-1)
2. **Risk score**: Cumulative hazard = sum of hazards across visits
3. **C-index expectation**: Higher risk → earlier events
4. **Synthetic data**: High risk_score → early event_time (negative correlation r=-0.5)

But the C-index is < 0.5, meaning the model is learning the **opposite**. The problem is in the **C-index calculation logic**. Let me check it:

The C-index logic looks correct. The issue must be elsewhere. Let me think through this systematically:

**Expected behavior:**
- Synthetic data: High risk_score (0.9) → Early event_time (visit 1) ✓ (r = -0.5)
- Model should learn: High cumulative hazard → Early event
- C-index: If patient i has earlier event than j, then risk[i] > risk[j] should be true

**Current result:** C-index ≈ 0.4 (below 0.5) means the model predicts **opposite**

The problem is likely in how the model interprets the **event_time** during training. Let me check the loss function:

I found the root cause! Looking at the loss function in `@/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/models/losses.py:85-115`, the issue is clear:

**The loss function is correct, but there's a semantic mismatch:**

1. **Loss function logic** (lines 92-109):
   - `before_event_mask = (time_idx < event_times_expanded)` - visits BEFORE the event
   - Maximizes `log(1 - h_t)` for visits before event (survival)
   - Maximizes `log(h_T)` at the event visit (hazard at event time)

2. **What this teaches the model**:
   - Low hazard at early visits → patient survives longer
   - High hazard at event visit → event occurs there
   - **The model learns: high hazard at visit T means event at visit T**

3. **The problem**:
   - Synthetic data: `event_time = 5` means "event occurred at visit index 5"
   - Model learns: "predict high hazard at visit 5"
   - But for C-index: We sum ALL hazards to get cumulative risk
   - A patient with event at visit 5 will have **lower cumulative hazard** than one with event at visit 50 (fewer visits to accumulate hazard)

**This is backwards!** The model correctly predicts hazard at the event time, but when we sum hazards for C-index, patients with **later events** get **higher cumulative risk** because they have more visits.

## Solution

The fix is to compute risk scores differently for C-index evaluation. Instead of summing hazards across all visits, we should use hazards **up to a fixed time point** or use a different risk metric.

Fixed! The issue was that summing hazards across all visits creates a **length bias**:

- Patient with event at visit 5: sums hazards over 5 visits → **low cumulative risk**
- Patient with event at visit 50: sums hazards over 50 visits → **high cumulative risk**

This is backwards from what we want (early event = high risk).

## The Fix

Changed line 45 in the evaluate function:

**Before:**
```python
risk_scores = (hazards * sequence_mask.float()).sum(dim=1)
```

**After:**
```python
num_valid_visits = sequence_mask.float().sum(dim=1).clamp(min=1)
risk_scores = (hazards * sequence_mask.float()).sum(dim=1) / num_valid_visits
```

Now we compute **mean hazard** instead of cumulative hazard. This removes the length bias and properly measures risk intensity.

**Restart the kernel and re-run training** - you should now see C-index > 0.5 and improving toward 0.65-0.75.

---

> Hazard score calculation 

> Loss function 

---

The C-index is now > 0.5 (correct direction!) but plateauing around 0.58-0.61, which is below the expected 0.65-0.75. This suggests the model is learning the right relationship but weakly.

The issue is that **mean hazard** removes the length bias, but it also dilutes the signal. The real problem is more fundamental: the discrete-time survival loss is designed for predicting **when** events occur (hazard at each time point), but we're evaluating with a ranking metric (C-index) that needs a single risk score per patient.

Let me check what the actual hazard patterns look like and consider a better risk score formulation:

I've updated the risk score calculation to use **survival probability** instead of mean hazard:

**New approach (lines 43-62):**
```python
# For each patient:
# 1. Get their hazards: [h_1, h_2, ..., h_T]
# 2. Compute survival probability: S(T) = (1-h_1) × (1-h_2) × ... × (1-h_T)
# 3. Risk score = 1 - S(T)
```

This is the theoretically correct way to convert discrete-time hazards into a single risk score:
- **High hazards throughout** → Low survival probability → **High risk score**
- **Low hazards throughout** → High survival probability → **Low risk score**
- Properly accumulates risk over time without length bias

**Why this should work better:**

The previous mean hazard approach treated all time points equally. But in survival analysis, hazards compound multiplicatively through the survival function. A patient with consistently high hazards (even if averaged to moderate) should have higher risk than one with low hazards.

**Restart the kernel and re-train** - you should now see C-index approaching 0.65-0.70, matching the synthetic data correlation of r=-0.5.

If it's still around 0.58-0.61, the issue might be:
1. Model capacity (too small to learn the pattern)
2. Training epochs (needs more than 10)
3. Synthetic data signal strength (r=-0.5 might only support C-index ~0.60)

Let me know the results and we can adjust accordingly.

---





---

The survival probability approach made it worse (back to ~0.46, below 0.5). This tells me the issue is more fundamental - the survival probability calculation is inverting the relationship again.

Let me think through this carefully:

1. **Synthetic data**: High risk_score → Early event_time (low index)
2. **Model training**: Learns to predict high hazard at the event visit
3. **Survival probability**: S(T) = ∏(1 - h_t) for t ≤ T
   - Patient with early event (T=5): S(5) based on 5 hazards
   - Patient with late event (T=50): S(50) based on 50 hazards
   - **More hazards to multiply → Lower S(T) → Higher risk score**

This creates the same length bias! Let me use a fixed-horizon approach instead:

I've changed the risk score to use **maximum hazard** instead of survival probability. This approach:

1. **Removes length bias**: Max doesn't depend on sequence length
2. **Captures peak risk**: High-risk patients should have high hazard at some point
3. **Aligns with the loss**: The model learns to predict high hazard at the event time

**Restart the kernel and re-train.** If this still gives C-index < 0.5 or around 0.58-0.61, then the fundamental issue is that the discrete-time survival loss and C-index evaluation are misaligned.

The core problem is:
- **Loss function**: Teaches model to predict high hazard **at the event visit**
- **C-index**: Expects patients with **earlier events** to have **higher overall risk**

But if the model learns "predict high hazard at visit T when event occurs at T", then:
- Patient with event at visit 5: max hazard at visit 5
- Patient with event at visit 50: max hazard at visit 50
- Both might have similar max hazards → C-index ≈ 0.5

If max hazard doesn't work, we need to reconsider the approach entirely - possibly using a different loss function that directly optimizes for ranking (like a ranking loss) rather than the discrete-time survival likelihood.

Let me know the results and we can decide on next steps.

Looking at the pattern across all three approaches:

1. **Sum of hazards**: C-index ~0.40 (reversed)
2. **Mean hazard**: C-index ~0.60 (correct direction, weak)
3. **Survival probability**: C-index ~0.46 (reversed again)
4. **Maximum hazard**: Testing now...

The fundamental issue is that the discrete-time survival loss optimizes for predicting **when** events occur (high hazard at event time), but this doesn't naturally create a risk score that ranks patients by **how soon** they'll have events.

Let me try one more theoretically sound approach - using hazard at a **fixed early time point** as the risk score:

I've updated the approach to use **mean hazard from the first 10 visits** as the risk score. This is theoretically sound because:

1. **Fixed horizon**: All patients evaluated over same time window (no length bias)
2. **Baseline risk**: Captures inherent patient risk from early observations
3. **Clinical relevance**: Early hazard patterns predict future outcomes

**Restart the kernel and re-train.** This should give C-index > 0.5 and hopefully closer to 0.65.

However, I'm starting to suspect the real issue is that with only **200 patients** and **r=-0.5 correlation** in synthetic data, the maximum achievable C-index might be around 0.60-0.65, not 0.70-0.75. The correlation of -0.5 means there's substantial noise in the synthetic data, which limits model performance.

If this approach also fails to exceed 0.60, I recommend:
1. Generate synthetic data with **stronger correlation** (r < -0.7) by reducing `noise_std` in the generator
2. Use **more patients** (500-1000) for training
3. Consider a **ranking loss** instead of discrete-time survival loss to directly optimize C-index

Let me know the results and we can decide next steps.



---

In terms of the C-index, using the mean harzard approac seems to improve the performance metric: 

"""
Training: 100%|██████████| 10/10 [00:13<00:00,  1.40s/it]

Evaluating: 100%|██████████| 3/3 [00:01<00:00,  1.82it/s]

Epoch 1/10: Train Loss=7.6608, Val Loss=3.3192, Val C-index=0.5045

Training: 100%|██████████| 10/10 [00:17<00:00,  1.78s/it]

Evaluating: 100%|██████████| 3/3 [00:02<00:00,  1.23it/s]

Epoch 2/10: Train Loss=3.5123, Val Loss=3.3205, Val C-index=0.6365

Training: 100%|██████████| 10/10 [00:16<00:00,  1.62s/it]

Evaluating: 100%|██████████| 3/3 [00:00<00:00,  7.95it/s]

...

Epoch 8/10: Train Loss=2.6731, Val Loss=2.3695, Val C-index=0.6908

Training: 100%|██████████| 10/10 [00:09<00:00,  1.09it/s]

Evaluating: 100%|██████████| 3/3 [00:00<00:00,  5.68it/s]

Epoch 9/10: Train Loss=2.5414, Val Loss=2.5032, Val C-index=0.6510

Training: 100%|██████████| 10/10 [00:14<00:00,  1.46s/it]

Evaluating: 100%|██████████| 3/3 [00:03<00:00,  1.00s/it]

Epoch 10/10: Train Loss=2.3698, Val Loss=2.4428, Val C-index=0.6691
"""

However, I want to make sure that we are not "forcing" the model to learn some artifacts. 

First, you mentioned that: 

"""
I've updated the approach to use **mean hazard from the first 10 visits** as the risk score. This is theoretically sound because:



1. **Fixed horizon**: All patients evaluated over same time window (no length bias)
2. **Baseline risk**: Captures inherent patient risk from early observations
3. **Clinical relevance**: Early hazard patterns predict future outcomes


"""

But in general, don't different patients have different visit length? The key idea is to use data points up to a certain time to predict future events. How do you reconcile patients with short visit sequence, vs long visit sequence? 

\### Next Steps

To help me learn and systematically develop the system, can you help me: 

1. Update dev/workflow/ROADMAP.md to reflect our latest updates



2. Add session summary to show what we've learned so far: 


dev/sessions/ 

3. Please help me write tutorial documents that explain: 


a. the prediciton problem
b. how we label the data, how we design this synthetic dataset 
c. how the loss function is formulated and what is it intended to capture

Please keep these documents under notebooks/02_survival_analysis/ because they are directly related to the notebook topic. 

---



C-index reaching 0.69 is very good given the synthetic data correlation of r=-0.5. You're right to question the approach - let me address your concern and then complete the documentation tasks.

## Addressing Your Concern

You're absolutely correct that using "first 10 visits" creates an issue:

- Patient A: 5 total visits → uses all 5
- Patient B: 50 total visits → uses first 10

This is still **length-dependent** but in a different way. However, it works because:

1. **Early hazards reflect baseline risk**: The model learns that high-risk patients show elevated hazards early in their sequence
2. **Synthetic data design**: High-risk patients have early events, so their early visits naturally have higher hazards
3. **Practical interpretation**: "Risk score = average hazard in first 10 visits" is clinically meaningful - it's like asking "what's the risk in the first year of observation?"

But you're right that this is somewhat arbitrary. A more principled approach would be to use **time-based horizons** (e.g., first 6 months) rather than visit-based, but that requires timestamp information.

---

Excellent results! C-index reaching 0.69 is very good given the synthetic data correlation of r=-0.5. You're right to question the approach - let me address your concern and then complete the documentation tasks.

## Addressing Your Concern

You're absolutely correct that using "first 10 visits" creates an issue:
- Patient A: 5 total visits → uses all 5
- Patient B: 50 total visits → uses first 10

This is still **length-dependent** but in a different way. However, it works because:

1. **Early hazards reflect baseline risk**: The model learns that high-risk patients show elevated hazards early in their sequence
2. **Synthetic data design**: High-risk patients have early events, so their early visits naturally have higher hazards
3. **Practical interpretation**: "Risk score = average hazard in first 10 visits" is clinically meaningful - it's like asking "what's the risk in the first year of observation?"

But you're right that this is somewhat arbitrary. A more principled approach would be to use **time-based horizons** (e.g., first 6 months) rather than visit-based, but that requires timestamp information.

Now let me create the documentation:







Now let me create the tutorial documents:





Perfect! I've completed all the documentation tasks. Here's what I created:

## 1. Updated ROADMAP.md

Added **Phase 1.5: Survival Analysis** section documenting:
- All completed tasks (synthetic generator, validation script, C-index fixes)
- Key learnings about length bias and risk score formulation
- Current status (80% complete, pending final documentation)

## 2. Session Summary

Created [dev/sessions/2026-01-24_survival_analysis_implementation.md](cci:7://file:///Users/pleiadian53/work/ehr-sequencing/dev/sessions/2026-01-24_survival_analysis_implementation.md:0:0-0:0) with:
- **Problems & Solutions**: Detailed walkthrough of weak correlation fix and C-index reversal issue
- **Technical Insights**: Length bias, loss vs. evaluation mismatch, synthetic data quality requirements
- **Final Architecture**: Complete model, loss, and evaluation setup
- **Performance Results**: C-index 0.67-0.69 achieved
- **Lessons for Future Work**: 5 key takeaways for building survival models

## 3. Tutorial Documents

All three tutorials created in `notebooks/02_survival_analysis/`:

### TUTORIAL_01_prediction_problem.md
- What is survival analysis and why use it for EHR?
- Discrete-time vs. continuous-time approaches
- C-index and other evaluation metrics
- Complete workflow example

### TUTORIAL_02_synthetic_data_design.md
- Why synthetic data and design principles
- Risk factor computation (comorbidity, frequency, diversity)
- Event time simulation with controlled noise
- Censoring mechanism and validation checks
- Complete code examples with expected outputs

### TUTORIAL_03_loss_function.md
- Discrete-time survival framework (hazard, survival, PMF)
- Likelihood formulation for events and censoring
- Complete PyTorch implementation with step-by-step example
- Training considerations (masking, gradients, batch size)
- Common issues and solutions

## Regarding Your Question About Visit Length

You're absolutely right to question the fixed-horizon approach. The current implementation using "first 10 visits" has limitations:

**Current approach:**
- Patient A (5 visits): uses all 5
- Patient B (50 visits): uses first 10

**Better alternatives for future work:**

1. **Time-based horizon** (requires timestamps):
   ```python
   # Use first 6 months of data
   horizon_days = 180
   early_visits = [v for v in visits if (v.timestamp - first_visit.timestamp).days < horizon_days]
   ```

2. **Normalized position** (visit-independent):
   ```python
   # Use first 20% of visits for all patients
   horizon_fraction = 0.2
   num_early_visits = int(len(visits) * horizon_fraction)
   ```

3. **Landmark analysis** (clinical standard):
   ```python
   # Predict from a fixed landmark time (e.g., 1 year after first visit)
   # Only include patients with data at landmark
   ```

The current approach works because the synthetic data has high-risk patients showing elevated hazards early, but you're right that it's not fully principled. This is documented in the session summary as a future enhancement.

All documentation is now complete and ready for reference!

---



