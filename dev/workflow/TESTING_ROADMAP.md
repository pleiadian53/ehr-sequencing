# EHR-Sequencing Testing Roadmap

**Purpose:** Systematic validation of BEHRT models and benchmarking framework on A40/high-end pods

**Target Hardware:** A40 GPU (40GB VRAM) or higher

---

## 📁 Best Practices: Output Directory Management

**CRITICAL:** Always use **absolute paths** for `--output-dir` to avoid confusion about where outputs are saved.

### Why This Matters

The default `--output-dir` in scripts uses relative paths (e.g., `experiments/benchmark_embeddings`), which can create outputs in unexpected locations depending on where you run the command from.

### Recommended Practice

```bash
# ❌ BAD: Relative path (creates multiple experiments/ directories)
nohup python benchmark_pretrained_embeddings.py \
    --output-dir experiments/benchmark_embeddings \
    > ../../experiments/sessions/output.log 2>&1 &

# ✅ GOOD: Absolute path (always goes to same location)
nohup python benchmark_pretrained_embeddings.py \
    --output-dir /workspace/ehr-sequencing/experiments/benchmark_embeddings \
    > /workspace/ehr-sequencing/experiments/sessions/output.log 2>&1 &
```

### Output Structure

All experiments should output to:
```
/workspace/ehr-sequencing/experiments/
├── sessions/                           # nohup stdout/stderr logs
│   ├── embeddings_comparison_large.out
│   ├── behrt_vs_pyhealth_large.out
│   └── phase1_embeddings.out
├── benchmark_embeddings/               # Experiment results
│   ├── SUMMARY.txt
│   ├── summary.json
│   ├── training_curves.png
│   └── tracker_state.json
└── behrt_vs_pyhealth/                 # Another experiment
    └── ...
```

### Monitoring Jobs

```bash
# Check running jobs
ps aux | grep python

# Monitor output
tail -f /workspace/ehr-sequencing/experiments/sessions/your_job.out

# Check for errors
grep -i "error\|warning\|failed" /workspace/ehr-sequencing/experiments/sessions/*.out
```

---

## 🎯 Research Questions

1. **Do pre-trained embeddings (Med2Vec) improve performance?**
2. **Does BEHRT outperform generic transformers (PyHealth)?**
3. **How do we demonstrate clinical utility?**
4. **Which survival analysis use cases should we prioritize?**

---

## 📋 Test Suite Overview

```
Phase 1: Pre-trained Embeddings Validation (Q1)
Phase 2: BEHRT vs PyHealth Comparison (Q2)
Phase 3: Clinical Utility Demonstrations (Q3)
Phase 4: Survival Analysis Use Cases (Q4)
```

---

# Phase 1: Pre-trained Embeddings Validation

**Question:** Should we freeze or fine-tune pre-trained embeddings? Do embeddings transfer across datasets?

## Test 1.1: Embedding Fine-tuning Strategy (Large Scale)

**What it tests:** Compare freeze vs fine-tune strategies for pre-trained embeddings

**3-way comparison:**
1. Train from scratch (baseline)
2. Load pre-trained embeddings, freeze them (reduced capacity)
3. Load pre-trained embeddings, fine-tune them (transfer learning)

**Command:**
```bash
# On A40 pod
cd /workspace/ehr-sequencing/examples/pretrain_finetune

# Full-scale test (10K patients, 100 epochs)
# Now includes 3-way comparison: Scratch vs Frozen vs Fine-tuned
nohup python -u benchmark_embedding_finetuning.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --batch-size 128 \
    --output-dir /workspace/ehr-sequencing/experiments/embedding_finetuning \
    > /workspace/ehr-sequencing/experiments/sessions/embedding_finetuning_large.out 2>&1 &

# Monitor progress
tail -f /workspace/ehr-sequencing/experiments/sessions/embedding_finetuning_large.out
```

**Expected Runtime:** ~4-6 hours on A40

**What to look for:**
- Does freezing embeddings hurt performance? (Yes, expected)
- Does fine-tuning match or beat training from scratch?
- Training time differences

**Success Criteria:**
- Fine-tuned ≥ Scratch > Frozen (performance ranking)
- Fine-tuned should converge faster than scratch (fewer epochs)
- Frozen should show degraded performance (fewer trainable params)

**Note:** This uses the refactored script with shared benchmarking infrastructure from `src/ehrsequencing/benchmarks/`

---

## Test 1.2: Transfer Learning Across Datasets (Large Scale)

**What it tests:** Do embeddings learned on one dataset transfer to another?

**4-way comparison:**
1. Train on Source, test on Source (baseline)
2. Train on Source, test on Target (zero-shot transfer)
3. Train on Source, fine-tune on Target, test on Target (transfer learning)
4. Train on Target from scratch, test on Target (upper bound)

**Command:**
```bash
# On A40 pod
cd /workspace/ehr-sequencing/examples/pretrain_finetune

# Full-scale test
nohup python -u benchmark_transfer_learning.py \
    --model-size large \
    --source-patients 10000 \
    --target-patients 5000 \
    --epochs 100 \
    --finetune-epochs 20 \
    --batch-size 128 \
    --output-dir /workspace/ehr-sequencing/experiments/transfer_learning \
    > /workspace/ehr-sequencing/experiments/sessions/transfer_learning_large.out 2>&1 &

# Monitor progress
tail -f /workspace/ehr-sequencing/experiments/sessions/transfer_learning_large.out
```

**Expected Runtime:** ~6-8 hours on A40

**What to look for:**
- How much does zero-shot transfer degrade? (Source→Target vs Source→Source)
- Does fine-tuning recover performance? (Fine-tuned vs Target-from-scratch)
- Is transfer learning better than training from scratch on limited data?

**Success Criteria:**
- Zero-shot transfer should show degradation (domain shift)
- Fine-tuning should recover most performance (within 10% of target-from-scratch)
- Transfer learning should beat training from scratch when target data is limited

**This is the real test of embedding quality and transferability.**

---

## Test 1.3: Embedding Quality Analysis

**What it tests:** Detailed analysis of embedding quality and transferability

**Status:** ✅ Covered by Tests 1.1 and 1.2

**Analysis Approach:**

Embedding quality is evaluated through the existing benchmarks:

1. **Test 1.1 (Embedding Fine-tuning):**
   - Compares Scratch vs Frozen vs Fine-tuned embeddings
   - Measures embedding initialization impact
   - Output: `experiments/embedding_finetuning/`

2. **Test 1.2 (Transfer Learning):**
   - Tests embedding transferability across domains
   - Measures zero-shot vs fine-tuned performance
   - Output: `experiments/transfer_learning/`

**Embedding Quality Metrics:**

From Test 1.1 outputs:
- Training convergence speed (fine-tuned should be faster)
- Final performance (fine-tuned ≥ scratch > frozen)
- Embedding statistics in saved `.pt` files

From Test 1.2 outputs:
- Zero-shot transfer performance (measures generalization)
- Fine-tuning improvement (measures adaptability)
- Domain shift robustness

**Additional Analysis (Optional):**

For deeper embedding analysis, you can:

```python
# Load saved embeddings
import torch
embeddings = torch.load('experiments/embedding_finetuning/final_embeddings.pt')

# Analyze embedding space
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Visualize embeddings
tsne = TSNE(n_components=2)
embedding_2d = tsne.fit_transform(embeddings.cpu().numpy())
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1])
plt.savefig('embedding_visualization.png')
```

**Note:** A dedicated `benchmark_pretrained_embeddings.py` script for external embeddings (e.g., Med2Vec) is planned for future development but not yet implemented.

---

## Test 1.4: Quick Validation (Sanity Check)

**What it tests:** Fast validation that everything works before long runs

**Command:**
```bash
# Quick test with small model (1K patients, 10 epochs)
cd examples/pretrain_finetune

python benchmark_embedding_finetuning.py \
    --model-size small \
    --num-patients 1000 \
    --epochs 10 \
    --batch-size 32 \
    --output-dir /tmp/quick_validation
```

**Expected Runtime:** ~15-30 minutes

**Purpose:** 
- Verify scripts work before committing to long runs
- Test on local machine before deploying to pod
- Quick iteration during development

---

# Phase 2: BEHRT vs PyHealth Comparison

**Question:** Does BEHRT's EHR-specific design (age/visit embeddings) outperform generic transformers?

## Test 2.1: BEHRT vs PyHealth (Large Scale)

**What it tests:** Direct comparison on same data/task

**Command:**
```bash
cd /workspace/ehr-sequencing/examples/benchmarking

# Full comparison
nohup python -u benchmark_pyhealth.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --batch-size 128 \
    --realistic-data \
    > ../../experiments/sessions/behrt_vs_pyhealth_large.out 2>&1 &
```

**Expected Runtime:** ~6-8 hours (trains both models)

**What to look for:**
- **Accuracy gap:** BEHRT should outperform PyHealth by ≥5-10%
- **Convergence:** BEHRT should converge faster (fewer epochs)
- **Generalization:** BEHRT should have smaller train-val gap
- **Efficiency:** Compare training time and parameter count

**Success Criteria:**
- BEHRT achieves higher ROC-AUC (target: ≥0.75 vs ≥0.65 for PyHealth)
- BEHRT shows better validation accuracy
- Smaller overfitting (train-val gap)

---

## Test 2.2: Architecture Ablation Study

**What it tests:** Which BEHRT components contribute most to performance?

**Command:**
```bash
# Compare BEHRT variants
python benchmark_training_comparison.py \
    --num-patients 5000 \
    --epochs 50
```

**Variants to test:**
1. Full BEHRT (code + age + visit + segment embeddings)
2. BEHRT without age embeddings
3. BEHRT without visit embeddings
4. Generic transformer (code only, like PyHealth)

**Expected Runtime:** ~4-5 hours

**Analysis:** Quantify contribution of each EHR-specific feature

---

## Test 2.3: Model Size Comparison

**What it tests:** Performance vs efficiency trade-offs

**Command:**
```bash
# Small model
python benchmark_pyhealth.py \
    --model-size small \
    --num-patients 5000 \
    --epochs 50 \
    --batch-size 64

# Medium model
python benchmark_pyhealth.py \
    --model-size medium \
    --num-patients 5000 \
    --epochs 50 \
    --batch-size 128

# Large model
python benchmark_pyhealth.py \
    --model-size large \
    --num-patients 5000 \
    --epochs 50 \
    --batch-size 128
```

**Analysis:** Find optimal model size for deployment

---

# Phase 3: Clinical Utility Demonstrations

**Question:** How do we showcase value to clinicians?

## Test 3.1: Disease Prediction Accuracy

**Clinical Story:** "Can BEHRT predict future diagnoses better than existing methods?"

**Test:**
```bash
# Train on realistic disease patterns
python train_behrt_demo.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --realistic-data \
    --output-dir experiments/clinical_demo/disease_prediction
```

**Metrics to highlight:**
- **Top-5 Accuracy:** "BEHRT correctly predicts the actual diagnosis in its top 5 predictions X% of the time"
- **Early Detection:** "BEHRT can predict diagnosis Y visits before it's recorded"
- **Rare Disease Detection:** "BEHRT maintains accuracy even for rare conditions"

**Deliverable:** 
- Confusion matrix for common diseases
- Case studies: "Patient had X, Y, Z → BEHRT predicted A (correct)"

---

## Test 3.2: Patient Trajectory Modeling

**Clinical Story:** "Can BEHRT understand patient disease progression?"

**Test:**
```bash
# Focus on sequential patterns
python benchmark_pretrained_embeddings.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --realistic-data
```

**Analysis:**
- Visualize learned embeddings (diseases that cluster together)
- Show attention weights (which past visits matter most?)
- Demonstrate temporal understanding

**Clinician Value:**
- "Patients with diabetes + hypertension have 70% chance of cardiovascular event"
- "BEHRT identifies high-risk patients 6 months earlier"

---

## Test 3.3: Embedding Quality for Clinical Tasks

**Clinical Story:** "BEHRT learns clinically meaningful representations"

**Test:**
```bash
# Generate embeddings for downstream tasks
python benchmark_pretrained_embeddings.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --save-embeddings
```

**Downstream Tasks:**
1. **Patient Similarity:** Find similar patients for treatment recommendations
2. **Disease Clustering:** Discover disease subtypes
3. **Risk Stratification:** Identify high-risk cohorts

**Deliverable:**
- t-SNE plot showing disease clusters
- Case study: "These 3 patients have similar trajectories"

---

# Phase 4: Survival Analysis Use Cases

**Question:** Which survival analysis applications should we prioritize?

## Recommended Use Cases (Priority Order)

### 🥇 Priority 1: Hospital Readmission Prediction

**Why:** High clinical impact, well-defined outcome, abundant data

**Clinical Value:**
- Reduce 30-day readmission rates (CMS penalty avoidance)
- Identify high-risk patients for intervention
- Optimize discharge planning

**Test:**
```bash
cd examples/survival_analysis

# Train readmission model
python train_lstm.py \
    --task readmission \
    --time-horizon 30 \
    --num-patients 10000 \
    --epochs 100 \
    --use-behrt-embeddings
```

**Metrics:**
- C-index (discrimination)
- Calibration curves
- Time-dependent AUC at 7, 14, 30 days

**Clinician Story:**
- "This patient has 75% risk of readmission within 30 days → arrange follow-up"

---

### 🥈 Priority 2: Mortality Risk Prediction

**Why:** Universal outcome, critical for ICU/ED triage

**Clinical Value:**
- ICU resource allocation
- Goals-of-care discussions
- Clinical trial enrollment

**Test:**
```bash
python train_lstm.py \
    --task mortality \
    --time-horizon 365 \
    --num-patients 10000 \
    --epochs 100 \
    --use-behrt-embeddings
```

**Time Horizons:**
- In-hospital mortality
- 30-day mortality
- 1-year mortality

**Clinician Story:**
- "This patient has 40% 1-year mortality risk → consider palliative care referral"

---

### 🥉 Priority 3: Disease Onset Prediction

**Why:** Preventive medicine, early intervention

**Clinical Value:**
- Identify pre-diabetic patients
- Predict cardiovascular events
- Screen for cancer risk

**Test:**
```bash
python train_lstm.py \
    --task disease_onset \
    --target-disease diabetes \
    --time-horizon 365 \
    --num-patients 10000 \
    --epochs 100
```

**Diseases to Focus On:**
1. **Diabetes** (high prevalence, preventable)
2. **Heart Failure** (high cost, manageable)
3. **Stroke** (severe outcome, preventable)

**Clinician Story:**
- "This patient has 60% risk of developing diabetes in next year → lifestyle intervention"

---

### Priority 4: Treatment Response Prediction

**Why:** Personalized medicine, optimize therapy selection

**Clinical Value:**
- Predict drug efficacy before prescription
- Avoid adverse events
- Reduce trial-and-error

**Test:**
```bash
python train_lstm.py \
    --task treatment_response \
    --treatment antihypertensive \
    --time-horizon 90 \
    --num-patients 5000
```

**Clinician Story:**
- "This patient likely won't respond to Drug A → try Drug B first"

---

# 📊 Comprehensive Test Plan (Recommended Sequence)

## Week 1: Quick Validation
```bash
# Day 1: Sanity checks
./run_quick_tests.sh

# Day 2-3: Medium-scale validation
python benchmark_pretrained_embeddings.py --model-size medium --num-patients 5000 --epochs 50
python benchmark_pyhealth.py --model-size medium --num-patients 5000 --epochs 50
```

## Week 2: Large-Scale Experiments
```bash
# Day 1-2: Embedding comparison (Test 1.1)
nohup python benchmark_pretrained_embeddings.py \
    --model-size large --num-patients 10000 --epochs 100 \
    --external-embedding-path pretrained/med2vec_embeddings.pt &

# Day 3-4: BEHRT vs PyHealth (Test 2.1)
nohup python benchmark_pyhealth.py \
    --model-size large --num-patients 10000 --epochs 100 --realistic-data &

# Day 5: Architecture ablation (Test 2.2)
python benchmark_training_comparison.py --num-patients 5000 --epochs 50
```

## Week 3: Clinical Demonstrations
```bash
# Day 1-2: Disease prediction (Test 3.1)
python train_behrt_demo.py --model-size large --num-patients 10000 --realistic-data

# Day 3-4: Survival analysis - Readmission
cd examples/survival_analysis
python train_lstm.py --task readmission --num-patients 10000 --epochs 100

# Day 5: Analysis and visualization
python generate_clinical_report.py
```

---

# 🎯 Success Metrics Summary

## Question 1: Do Pre-trained Embeddings Help?

**Quantitative:**
- ✅ Convergence speed: ≥10% fewer epochs to best val loss
- ✅ Final performance: Within 5% of training from scratch
- ✅ Training time: ≥20% reduction

**Qualitative:**
- Embedding quality analysis shows semantic alignment
- Transfer learning works across different datasets

**Answer:** "Yes, if Med2Vec embeddings reduce training time by X hours and achieve comparable accuracy"

---

## Question 2: Does BEHRT Beat PyHealth?

**Quantitative:**
- ✅ Accuracy: BEHRT ≥5-10% higher than PyHealth
- ✅ ROC-AUC: BEHRT ≥0.75 vs PyHealth ≥0.65
- ✅ Generalization: Smaller train-val gap

**Qualitative:**
- BEHRT learns clinically meaningful patterns (age, visit context)
- Better handles temporal dependencies

**Answer:** "Yes, BEHRT's EHR-specific design provides X% improvement in disease prediction accuracy"

---

## Question 3: Clinical Utility

**Key Messages for Clinicians:**

1. **Accuracy:** "BEHRT predicts diagnoses with X% accuracy, comparable to specialist physicians"

2. **Early Detection:** "BEHRT identifies high-risk patients Y months earlier than current methods"

3. **Actionable Insights:** "BEHRT provides interpretable predictions with confidence scores"

4. **Practical Use Cases:**
   - Reduce hospital readmissions by X%
   - Improve ICU resource allocation
   - Enable personalized treatment selection

**Deliverables:**
- Clinical case studies with real patient trajectories
- ROC curves and calibration plots
- Cost-benefit analysis (reduced readmissions = $X saved)

---

## Question 4: Survival Analysis Focus

**Recommended Priority:**

1. **Hospital Readmission** (Start Here)
   - High impact, clear outcome, abundant data
   - Direct cost savings for hospitals
   - Regulatory incentive (CMS penalties)

2. **Mortality Prediction** (Second)
   - Critical for ICU/ED
   - Universal outcome
   - Ethical considerations well-studied

3. **Disease Onset** (Third)
   - Preventive medicine angle
   - Focus on diabetes, heart failure, stroke
   - Longer time horizon (harder to validate)

4. **Treatment Response** (Future)
   - Requires treatment data
   - Personalized medicine
   - Regulatory challenges (FDA approval)

---

# 📁 Expected Outputs

After completing all tests, you should have:

```
experiments/
├── embedding_comparison/
│   ├── SUMMARY.txt
│   ├── training_curves.png
│   ├── roc_curves.png
│   └── embedding_analysis.txt
│
├── behrt_vs_pyhealth/
│   ├── SUMMARY.txt
│   ├── comparison.json
│   ├── performance_metrics.png
│   └── winner_analysis.txt
│
├── clinical_demo/
│   ├── disease_prediction/
│   │   ├── confusion_matrix.png
│   │   ├── top5_accuracy.txt
│   │   └── case_studies.md
│   │
│   └── patient_trajectories/
│       ├── embeddings_tsne.png
│       ├── attention_visualization.png
│       └── clinical_insights.md
│
└── survival_analysis/
    ├── readmission/
    │   ├── c_index.txt
    │   ├── calibration_curves.png
    │   └── time_dependent_auc.png
    │
    └── mortality/
        ├── risk_stratification.png
        └── clinical_utility.md
```

---

# 🚀 Quick Start Script

Create `run_full_test_suite.sh`:

```bash
#!/bin/bash
# Full test suite for A40 pod

set -e

echo "Starting EHR-Sequencing Test Suite..."

# Phase 1: Embedding comparison
echo "Phase 1: Testing pre-trained embeddings..."
cd examples/pretrain_finetune
nohup python -u benchmark_pretrained_embeddings.py \
    --model-size large --num-patients 10000 --epochs 100 \
    --external-embedding-path pretrained/med2vec_embeddings.pt \
    > ../../experiments/sessions/phase1_embeddings.out 2>&1 &
PID1=$!

# Wait for Phase 1
wait $PID1
echo "Phase 1 complete!"

# Phase 2: BEHRT vs PyHealth
echo "Phase 2: BEHRT vs PyHealth comparison..."
cd ../benchmarking
nohup python -u benchmark_pyhealth.py \
    --model-size large --num-patients 10000 --epochs 100 --realistic-data \
    > ../../experiments/sessions/phase2_comparison.out 2>&1 &
PID2=$!

wait $PID2
echo "Phase 2 complete!"

# Phase 3: Clinical demo
echo "Phase 3: Clinical utility demonstration..."
cd ../pretrain_finetune
python train_behrt_demo.py \
    --model-size large --num-patients 10000 --realistic-data

echo "Phase 3 complete!"

# Phase 4: Survival analysis
echo "Phase 4: Survival analysis..."
cd ../survival_analysis
python train_lstm.py --task readmission --num-patients 10000 --epochs 100

echo "All phases complete! Check experiments/ for results."
```

---

# 💡 Tips for A40 Pod Testing

1. **Monitor GPU Usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Use tmux for long runs:**
   ```bash
   tmux new -s behrt_test
   # Run commands
   # Detach: Ctrl+B, D
   # Reattach: tmux attach -t behrt_test
   ```

3. **Transfer results back:**
   ```bash
   # On pod
   cd /workspace/ehr-sequencing
   tar -czf results.tar.gz experiments/
   
   # On local
   scp runpod-main:/workspace/ehr-sequencing/results.tar.gz .
   ```

4. **Check for errors:**
   ```bash
   grep -i "error\|warning\|failed" experiments/sessions/*.out
   ```

---

# 📚 Documentation to Generate

After tests complete, create:

1. **Technical Report:** `docs/BENCHMARK_RESULTS.md`
   - All quantitative results
   - Statistical significance tests
   - Performance comparisons

2. **Clinical Brief:** `docs/CLINICAL_UTILITY.md`
   - Non-technical summary for clinicians
   - Use cases and case studies
   - ROI analysis

3. **Paper Draft:** `docs/BEHRT_VS_BASELINE.md`
   - Methods, results, discussion
   - Figures and tables
   - Ready for submission

---

**Next Steps:**
1. Run quick validation tests (Test 1.3, ~30 min)
2. If successful, launch large-scale experiments (Tests 1.1, 2.1)
3. Analyze results and generate reports
4. Iterate based on findings

Good luck with your testing! 🚀

# EHR-Sequencing Testing Roadmap

**Purpose:** Systematic validation of BEHRT models and benchmarking framework on A40/high-end pods

**Target Hardware:** A40 GPU (40GB VRAM) or higher

---

## 📁 Best Practices: Output Directory Management

**CRITICAL:** Always use **absolute paths** for `--output-dir` to avoid confusion about where outputs are saved.

### Why This Matters

The default `--output-dir` in scripts uses relative paths (e.g., `experiments/benchmark_embeddings`), which can create outputs in unexpected locations depending on where you run the command from.

### Recommended Practice

```bash
# ❌ BAD: Relative path (creates multiple experiments/ directories)
nohup python benchmark_pretrained_embeddings.py \
    --output-dir experiments/benchmark_embeddings \
    > ../../experiments/sessions/output.log 2>&1 &

# ✅ GOOD: Absolute path (always goes to same location)
nohup python benchmark_pretrained_embeddings.py \
    --output-dir /workspace/ehr-sequencing/experiments/benchmark_embeddings \
    > /workspace/ehr-sequencing/experiments/sessions/output.log 2>&1 &
```

### Output Structure

All experiments should output to:
```
/workspace/ehr-sequencing/experiments/
├── sessions/                           # nohup stdout/stderr logs
│   ├── embeddings_comparison_large.out
│   ├── behrt_vs_pyhealth_large.out
│   └── phase1_embeddings.out
├── benchmark_embeddings/               # Experiment results
│   ├── SUMMARY.txt
│   ├── summary.json
│   ├── training_curves.png
│   └── tracker_state.json
└── behrt_vs_pyhealth/                 # Another experiment
    └── ...
```

### Monitoring Jobs

```bash
# Check running jobs
ps aux | grep python

# Monitor output
tail -f /workspace/ehr-sequencing/experiments/sessions/your_job.out

# Check for errors
grep -i "error\|warning\|failed" /workspace/ehr-sequencing/experiments/sessions/*.out
```

---

## 🎯 Research Questions

1. **Do pre-trained embeddings (Med2Vec) improve performance?**
2. **Does BEHRT outperform generic transformers (PyHealth)?**
3. **How do we demonstrate clinical utility?**
4. **Which survival analysis use cases should we prioritize?**

---

## 📋 Test Suite Overview

```
Phase 1: Pre-trained Embeddings Validation (Q1)
Phase 2: BEHRT vs PyHealth Comparison (Q2)
Phase 3: Clinical Utility Demonstrations (Q3)
Phase 4: Survival Analysis Use Cases (Q4)
```

---

# Phase 1: Pre-trained Embeddings Validation

**Question:** Does using pre-trained embeddings (Med2Vec) improve convergence speed and final performance?

## Test 1.1: 3-Way Embedding Comparison (Large Scale)

**What it tests:** Compare training from scratch vs learned embeddings vs Med2Vec embeddings

**Command:**
```bash
# On A40 pod
cd /workspace/ehr-sequencing/examples/pretrain_finetune

# Full-scale test (10K patients, 100 epochs)
# IMPORTANT: Use absolute path for --output-dir to avoid confusion
nohup python -u benchmark_pretrained_embeddings.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --batch-size 128 \
    --external-embedding-path pretrained/med2vec_embeddings.pt \
    --output-dir /workspace/ehr-sequencing/experiments/benchmark_embeddings \
    > /workspace/ehr-sequencing/experiments/sessions/embeddings_comparison_large.out 2>&1 &

# Monitor progress
tail -f /workspace/ehr-sequencing/experiments/sessions/embeddings_comparison_large.out
```

**Expected Runtime:** ~4-6 hours on A40

**What to look for:**
- Does Run 2 (learned embeddings) converge faster than Run 1 (scratch)?
- Does Run 3 (Med2Vec) show better initial performance?
- Which achieves highest final ROC-AUC?
- Training time differences

**Success Criteria:**
- Pre-trained embeddings should show ≥10% faster convergence (fewer epochs to best val loss)
- Final performance should be within 5% of training from scratch
- If Med2Vec helps: ROC-AUC improvement ≥0.02

---

## Test 1.2: Embedding Quality Analysis (Medium Scale)

**What it tests:** Detailed analysis of embedding quality and transferability

**Command:**
```bash
# Medium-scale for faster iteration
python benchmark_pretrained_embeddings.py \
    --model-size medium \
    --num-patients 5000 \
    --epochs 50 \
    --batch-size 128 \
    --external-embedding-path pretrained/med2vec_embeddings.pt
```

**Expected Runtime:** ~2-3 hours

**Analysis:**
1. Compare embedding statistics (see `pretrained/embedding_analysis.txt`)
2. Check if Med2Vec embeddings align with learned embeddings
3. Visualize embedding spaces (t-SNE/UMAP)

**Deliverable:** `experiments/embedding_comparison/SUMMARY.txt`

---

## Test 1.3: Quick Validation (Sanity Check)

**What it tests:** Fast validation that everything works

**Command:**
```bash
# Quick test (1K patients, 20 epochs)
python benchmark_pretrained_embeddings.py \
    --model-size small \
    --num-patients 1000 \
    --epochs 20 \
    --batch-size 32
```

**Expected Runtime:** ~15-30 minutes

**Purpose:** Verify scripts work before long runs

---

# Phase 2: BEHRT vs PyHealth Comparison

**Question:** Does BEHRT's EHR-specific design (age/visit embeddings) outperform generic transformers?

## Test 2.1: BEHRT vs PyHealth (Large Scale)

**What it tests:** Direct comparison on same data/task

**Command:**
```bash
cd /workspace/ehr-sequencing/examples/benchmarking

# Full comparison
# IMPORTANT: Use absolute path for --output-dir
nohup python -u benchmark_pyhealth.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --batch-size 128 \
    --realistic-data \
    --output-dir /workspace/ehr-sequencing/experiments/behrt_vs_pyhealth \
    > /workspace/ehr-sequencing/experiments/sessions/behrt_vs_pyhealth_large.out 2>&1 &
```

**Expected Runtime:** ~6-8 hours (trains both models)

**What to look for:**
- **Accuracy gap:** BEHRT should outperform PyHealth by ≥5-10%
- **Convergence:** BEHRT should converge faster (fewer epochs)
- **Generalization:** BEHRT should have smaller train-val gap
- **Efficiency:** Compare training time and parameter count

**Success Criteria:**
- BEHRT achieves higher ROC-AUC (target: ≥0.75 vs ≥0.65 for PyHealth)
- BEHRT shows better validation accuracy
- Smaller overfitting (train-val gap)

---

## Test 2.2: Architecture Ablation Study

**What it tests:** Which BEHRT components contribute most to performance?

**Command:**
```bash
# Compare BEHRT variants
python benchmark_training_comparison.py \
    --num-patients 5000 \
    --epochs 50
```

**Variants to test:**
1. Full BEHRT (code + age + visit + segment embeddings)
2. BEHRT without age embeddings
3. BEHRT without visit embeddings
4. Generic transformer (code only, like PyHealth)

**Expected Runtime:** ~4-5 hours

**Analysis:** Quantify contribution of each EHR-specific feature

---

## Test 2.3: Model Size Comparison

**What it tests:** Performance vs efficiency trade-offs

**Command:**
```bash
# Small model
python benchmark_pyhealth.py \
    --model-size small \
    --num-patients 5000 \
    --epochs 50 \
    --batch-size 64

# Medium model
python benchmark_pyhealth.py \
    --model-size medium \
    --num-patients 5000 \
    --epochs 50 \
    --batch-size 128

# Large model
python benchmark_pyhealth.py \
    --model-size large \
    --num-patients 5000 \
    --epochs 50 \
    --batch-size 128
```

**Analysis:** Find optimal model size for deployment

---

# Phase 3: Clinical Utility Demonstrations

**Question:** How do we showcase value to clinicians?

## Test 3.1: Disease Prediction Accuracy

**Clinical Story:** "Can BEHRT predict future diagnoses better than existing methods?"

**Test:**
```bash
# Train on realistic disease patterns
python train_behrt_demo.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --realistic-data \
    --output-dir experiments/clinical_demo/disease_prediction
```

**Metrics to highlight:**
- **Top-5 Accuracy:** "BEHRT correctly predicts the actual diagnosis in its top 5 predictions X% of the time"
- **Early Detection:** "BEHRT can predict diagnosis Y visits before it's recorded"
- **Rare Disease Detection:** "BEHRT maintains accuracy even for rare conditions"

**Deliverable:** 
- Confusion matrix for common diseases
- Case studies: "Patient had X, Y, Z → BEHRT predicted A (correct)"

---

## Test 3.2: Patient Trajectory Modeling

**Clinical Story:** "Can BEHRT understand patient disease progression?"

**Test:**
```bash
# Focus on sequential patterns
python benchmark_pretrained_embeddings.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --realistic-data
```

**Analysis:**
- Visualize learned embeddings (diseases that cluster together)
- Show attention weights (which past visits matter most?)
- Demonstrate temporal understanding

**Clinician Value:**
- "Patients with diabetes + hypertension have 70% chance of cardiovascular event"
- "BEHRT identifies high-risk patients 6 months earlier"

---

## Test 3.3: Embedding Quality for Clinical Tasks

**Clinical Story:** "BEHRT learns clinically meaningful representations"

**Test:**
```bash
# Generate embeddings for downstream tasks
python benchmark_pretrained_embeddings.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --save-embeddings
```

**Downstream Tasks:**
1. **Patient Similarity:** Find similar patients for treatment recommendations
2. **Disease Clustering:** Discover disease subtypes
3. **Risk Stratification:** Identify high-risk cohorts

**Deliverable:**
- t-SNE plot showing disease clusters
- Case study: "These 3 patients have similar trajectories"

---

# Phase 4: Survival Analysis Use Cases

**Question:** Which survival analysis applications should we prioritize?

## Recommended Use Cases (Priority Order)

### 🥇 Priority 1: Hospital Readmission Prediction

**Why:** High clinical impact, well-defined outcome, abundant data

**Clinical Value:**
- Reduce 30-day readmission rates (CMS penalty avoidance)
- Identify high-risk patients for intervention
- Optimize discharge planning

**Test:**
```bash
cd examples/survival_analysis

# Train readmission model
python train_lstm.py \
    --task readmission \
    --time-horizon 30 \
    --num-patients 10000 \
    --epochs 100 \
    --use-behrt-embeddings
```

**Metrics:**
- C-index (discrimination)
- Calibration curves
- Time-dependent AUC at 7, 14, 30 days

**Clinician Story:**
- "This patient has 75% risk of readmission within 30 days → arrange follow-up"

---

### 🥈 Priority 2: Mortality Risk Prediction

**Why:** Universal outcome, critical for ICU/ED triage

**Clinical Value:**
- ICU resource allocation
- Goals-of-care discussions
- Clinical trial enrollment

**Test:**
```bash
python train_lstm.py \
    --task mortality \
    --time-horizon 365 \
    --num-patients 10000 \
    --epochs 100 \
    --use-behrt-embeddings
```

**Time Horizons:**
- In-hospital mortality
- 30-day mortality
- 1-year mortality

**Clinician Story:**
- "This patient has 40% 1-year mortality risk → consider palliative care referral"

---

### 🥉 Priority 3: Disease Onset Prediction

**Why:** Preventive medicine, early intervention

**Clinical Value:**
- Identify pre-diabetic patients
- Predict cardiovascular events
- Screen for cancer risk

**Test:**
```bash
python train_lstm.py \
    --task disease_onset \
    --target-disease diabetes \
    --time-horizon 365 \
    --num-patients 10000 \
    --epochs 100
```

**Diseases to Focus On:**
1. **Diabetes** (high prevalence, preventable)
2. **Heart Failure** (high cost, manageable)
3. **Stroke** (severe outcome, preventable)

**Clinician Story:**
- "This patient has 60% risk of developing diabetes in next year → lifestyle intervention"

---

### Priority 4: Treatment Response Prediction

**Why:** Personalized medicine, optimize therapy selection

**Clinical Value:**
- Predict drug efficacy before prescription
- Avoid adverse events
- Reduce trial-and-error

**Test:**
```bash
python train_lstm.py \
    --task treatment_response \
    --treatment antihypertensive \
    --time-horizon 90 \
    --num-patients 5000
```

**Clinician Story:**
- "This patient likely won't respond to Drug A → try Drug B first"

---

# 📊 Comprehensive Test Plan (Recommended Sequence)

## Week 1: Quick Validation
```bash
# Day 1: Sanity checks
./run_quick_tests.sh

# Day 2-3: Medium-scale validation
python benchmark_pretrained_embeddings.py --model-size medium --num-patients 5000 --epochs 50
python benchmark_pyhealth.py --model-size medium --num-patients 5000 --epochs 50
```

## Week 2: Large-Scale Experiments
```bash
# Day 1-2: Embedding comparison (Test 1.1)
nohup python benchmark_pretrained_embeddings.py \
    --model-size large --num-patients 10000 --epochs 100 \
    --external-embedding-path pretrained/med2vec_embeddings.pt \
    --output-dir /workspace/ehr-sequencing/experiments/benchmark_embeddings \
    > /workspace/ehr-sequencing/experiments/sessions/test1.1.out 2>&1 &

# Day 3-4: BEHRT vs PyHealth (Test 2.1)
nohup python benchmark_pyhealth.py \
    --model-size large --num-patients 10000 --epochs 100 --realistic-data \
    --output-dir /workspace/ehr-sequencing/experiments/behrt_vs_pyhealth \
    > /workspace/ehr-sequencing/experiments/sessions/test2.1.out 2>&1 &

# Day 5: Architecture ablation (Test 2.2)
python benchmark_training_comparison.py --num-patients 5000 --epochs 50
```

## Week 3: Clinical Demonstrations
```bash
# Day 1-2: Disease prediction (Test 3.1)
python train_behrt_demo.py --model-size large --num-patients 10000 --realistic-data

# Day 3-4: Survival analysis - Readmission
cd examples/survival_analysis
python train_lstm.py --task readmission --num-patients 10000 --epochs 100

# Day 5: Analysis and visualization
python generate_clinical_report.py
```

---

# 🎯 Success Metrics Summary

## Question 1: Do Pre-trained Embeddings Help?

**Quantitative:**
- ✅ Convergence speed: ≥10% fewer epochs to best val loss
- ✅ Final performance: Within 5% of training from scratch
- ✅ Training time: ≥20% reduction

**Qualitative:**
- Embedding quality analysis shows semantic alignment
- Transfer learning works across different datasets

**Answer:** "Yes, if Med2Vec embeddings reduce training time by X hours and achieve comparable accuracy"

---

## Question 2: Does BEHRT Beat PyHealth?

**Quantitative:**
- ✅ Accuracy: BEHRT ≥5-10% higher than PyHealth
- ✅ ROC-AUC: BEHRT ≥0.75 vs PyHealth ≥0.65
- ✅ Generalization: Smaller train-val gap

**Qualitative:**
- BEHRT learns clinically meaningful patterns (age, visit context)
- Better handles temporal dependencies

**Answer:** "Yes, BEHRT's EHR-specific design provides X% improvement in disease prediction accuracy"

---

## Question 3: Clinical Utility

**Key Messages for Clinicians:**

1. **Accuracy:** "BEHRT predicts diagnoses with X% accuracy, comparable to specialist physicians"

2. **Early Detection:** "BEHRT identifies high-risk patients Y months earlier than current methods"

3. **Actionable Insights:** "BEHRT provides interpretable predictions with confidence scores"

4. **Practical Use Cases:**
   - Reduce hospital readmissions by X%
   - Improve ICU resource allocation
   - Enable personalized treatment selection

**Deliverables:**
- Clinical case studies with real patient trajectories
- ROC curves and calibration plots
- Cost-benefit analysis (reduced readmissions = $X saved)

---

## Question 4: Survival Analysis Focus

**Recommended Priority:**

1. **Hospital Readmission** (Start Here)
   - High impact, clear outcome, abundant data
   - Direct cost savings for hospitals
   - Regulatory incentive (CMS penalties)

2. **Mortality Prediction** (Second)
   - Critical for ICU/ED
   - Universal outcome
   - Ethical considerations well-studied

3. **Disease Onset** (Third)
   - Preventive medicine angle
   - Focus on diabetes, heart failure, stroke
   - Longer time horizon (harder to validate)

4. **Treatment Response** (Future)
   - Requires treatment data
   - Personalized medicine
   - Regulatory challenges (FDA approval)

---

# 📁 Expected Outputs

After completing all tests, you should have:

```
experiments/
├── embedding_comparison/
│   ├── SUMMARY.txt
│   ├── training_curves.png
│   ├── roc_curves.png
│   └── embedding_analysis.txt
│
├── behrt_vs_pyhealth/
│   ├── SUMMARY.txt
│   ├── comparison.json
│   ├── performance_metrics.png
│   └── winner_analysis.txt
│
├── clinical_demo/
│   ├── disease_prediction/
│   │   ├── confusion_matrix.png
│   │   ├── top5_accuracy.txt
│   │   └── case_studies.md
│   │
│   └── patient_trajectories/
│       ├── embeddings_tsne.png
│       ├── attention_visualization.png
│       └── clinical_insights.md
│
└── survival_analysis/
    ├── readmission/
    │   ├── c_index.txt
    │   ├── calibration_curves.png
    │   └── time_dependent_auc.png
    │
    └── mortality/
        ├── risk_stratification.png
        └── clinical_utility.md
```

---

# 🚀 Quick Start Script

Create `run_full_test_suite.sh`:

```bash
#!/bin/bash
# Full test suite for A40 pod

set -e

echo "Starting EHR-Sequencing Test Suite..."

# Phase 1: Embedding comparison
echo "Phase 1: Testing pre-trained embeddings..."
cd examples/pretrain_finetune
nohup python -u benchmark_pretrained_embeddings.py \
    --model-size large --num-patients 10000 --epochs 100 \
    --external-embedding-path pretrained/med2vec_embeddings.pt \
    --output-dir /workspace/ehr-sequencing/experiments/benchmark_embeddings \
    > /workspace/ehr-sequencing/experiments/sessions/phase1_embeddings.out 2>&1 &
PID1=$!

# Wait for Phase 1
wait $PID1
echo "Phase 1 complete!"

# Phase 2: BEHRT vs PyHealth
echo "Phase 2: BEHRT vs PyHealth comparison..."
cd ../benchmarking
nohup python -u benchmark_pyhealth.py \
    --model-size large --num-patients 10000 --epochs 100 --realistic-data \
    --output-dir /workspace/ehr-sequencing/experiments/behrt_vs_pyhealth \
    > /workspace/ehr-sequencing/experiments/sessions/phase2_comparison.out 2>&1 &
PID2=$!

wait $PID2
echo "Phase 2 complete!"

# Phase 3: Clinical demo
echo "Phase 3: Clinical utility demonstration..."
cd ../pretrain_finetune
python train_behrt_demo.py \
    --model-size large --num-patients 10000 --realistic-data

echo "Phase 3 complete!"

# Phase 4: Survival analysis
echo "Phase 4: Survival analysis..."
cd ../survival_analysis
python train_lstm.py --task readmission --num-patients 10000 --epochs 100

echo "All phases complete! Check experiments/ for results."
```

---

# 💡 Tips for A40 Pod Testing

1. **Monitor GPU Usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Use tmux for long runs:**
   ```bash
   tmux new -s behrt_test
   # Run commands
   # Detach: Ctrl+B, D
   # Reattach: tmux attach -t behrt_test
   ```

3. **Transfer results back:**
   ```bash
   # On pod
   cd /workspace/ehr-sequencing
   tar -czf results.tar.gz experiments/
   
   # On local
   scp runpod-main:/workspace/ehr-sequencing/results.tar.gz .
   ```

4. **Check for errors:**
   ```bash
   grep -i "error\|warning\|failed" experiments/sessions/*.out
   ```

---

# 📚 Documentation to Generate

After tests complete, create:

1. **Technical Report:** `docs/BENCHMARK_RESULTS.md`
   - All quantitative results
   - Statistical significance tests
   - Performance comparisons

2. **Clinical Brief:** `docs/CLINICAL_UTILITY.md`
   - Non-technical summary for clinicians
   - Use cases and case studies
   - ROI analysis

3. **Paper Draft:** `docs/BEHRT_VS_BASELINE.md`
   - Methods, results, discussion
   - Figures and tables
   - Ready for submission

---

**Next Steps:**
1. Run quick validation tests (Test 1.3, ~30 min)
2. If successful, launch large-scale experiments (Tests 1.1, 2.1)
3. Analyze results and generate reports
4. Iterate based on findings

Good luck with your testing! 🚀
