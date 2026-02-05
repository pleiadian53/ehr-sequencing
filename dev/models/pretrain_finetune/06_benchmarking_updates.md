# Benchmarking Implementation Updates

**Last Updated:** 2026-02-03  
**Purpose:** Document changes to benchmarking scripts and infrastructure

---

## Overview of Changes

The benchmarking infrastructure has been **refactored and improved** based on insights from initial experiments. The models themselves (`src/ehrsequencing/models/`) remain unchanged, but the benchmarking design is now more rigorous and addresses the right questions.

---

## What Changed

### 1. Script Restructuring

**Old Structure:**
```
examples/pretrain_finetune/
└── benchmark_pretrained_embeddings.py  # Single script, 3-way comparison
```

**New Structure:**
```
examples/pretrain_finetune/
├── benchmark_embedding_finetuning.py   # Tests: freeze vs fine-tune
└── benchmark_transfer_learning.py      # Tests: domain transfer
```

### 2. Shared Infrastructure

**New:** `src/ehrsequencing/benchmarks/training.py`

**Purpose:** Eliminate code duplication across benchmarking scripts

**Benefits:**
- Consistent training logic
- Unified metric computation
- Easier to maintain
- Handles both `BEHRTForMLM` and standard model formats

**Functions moved to shared infrastructure:**
- `train_epoch()` - Training loop
- `evaluate()` - Evaluation with metrics
- `compute_metrics()` - Metric calculation
- `BenchmarkTracker` - Experiment tracking

---

## New Benchmark Design

### Benchmark 1: Embedding Fine-tuning Strategy

**File:** `benchmark_embedding_finetuning.py`

**Question:** Should we freeze or fine-tune pretrained embeddings?

**3-way comparison:**

| Run | Initialization | Embeddings | Expected Performance |
|-----|----------------|------------|---------------------|
| 1. **Scratch** | Random | Trainable | Baseline |
| 2. **Frozen** | Pretrained | **Frozen** | Worst (reduced capacity) |
| 3. **Fine-tuned** | Pretrained | **Trainable** | Best (initialized + adaptable) |

**Expected ranking:** Fine-tuned ≥ Scratch > Frozen

**Usage:**
```bash
python examples/pretrain_finetune/benchmark_embedding_finetuning.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --batch-size 128 \
    --embedding-path pretrained/embeddings.pt
```

**Insights from initial experiments:**
- Frozen embeddings reduce trainable parameters by ~40%
- Frozen embeddings can't adapt to new transformer initialization
- Always fine-tune unless preventing catastrophic forgetting

### Benchmark 2: Transfer Learning

**File:** `benchmark_transfer_learning.py`

**Question:** Do embeddings transfer across different datasets/distributions?

**4-way comparison:**

| Run | Train Data | Test Data | Adaptation | Purpose |
|-----|-----------|-----------|------------|---------|
| 1. **Source baseline** | Source | Source | None | Upper bound on source |
| 2. **Zero-shot** | Source | Target | None | Measure domain shift |
| 3. **Transfer learning** | Source | Target | Fine-tune on target | Transfer benefit |
| 4. **Target baseline** | Target | Target | None | Upper bound on target |

**Expected results:**
- Zero-shot shows degradation (domain shift exists)
- Transfer learning recovers most performance
- Transfer beats target-from-scratch with limited data

**Usage:**
```bash
python examples/pretrain_finetune/benchmark_transfer_learning.py \
    --model-size large \
    --source-patients 10000 \
    --target-patients 5000 \
    --epochs 100 \
    --finetune-epochs 20 \
    --batch-size 128
```

**This is the gold standard test for embedding quality.**

---

## Why the Redesign?

### Problem with Original Design

**Original script:** `benchmark_pretrained_embeddings.py`

**Claimed to test:** "Pre-training vs Fine-tuning with Pre-trained Embeddings"

**Actually tested:** "Does freezing embeddings hurt performance?"

**Results from initial pod run:**
```
RUN 1 (Scratch):  Val Acc 36.35%, Val Loss 1.8347 ✅
RUN 2 (Frozen):   Val Acc 31.58%, Val Loss 2.0899 ❌
```

**Why RUN 2 performed worse:**
1. Frozen embeddings → can't adapt to new transformer
2. Same dataset → no transfer learning benefit
3. 40% fewer trainable parameters → reduced capacity

**Conclusion:** The benchmark answered "Does freezing hurt?" (Yes) instead of "Do pre-trained embeddings help?" (Need transfer learning test)

### New Design Addresses Real Questions

**Benchmark 1 answers:**
- Should embeddings be frozen or fine-tuned?
- How much does freezing reduce performance?
- Is there benefit to pre-initialization?

**Benchmark 2 answers:**
- Do embeddings generalize across datasets?
- Is transfer learning effective?
- How much fine-tuning is needed for adaptation?

---

## Impact on Tutorial Documentation

### Concepts That Remain the Same

✅ **BEHRT architecture** - No changes
✅ **Embedding design** (code + age + visit + position) - No changes
✅ **MLM pre-training objective** - No changes
✅ **LoRA application** - No changes
✅ **Training workflows** - No changes
✅ **Model implementations** - No changes

### What Changed in Tutorials

❌ **Old references:**
- `benchmark_pretrained_embeddings.py` → Now `benchmark_embedding_finetuning.py`
- "3-way comparison" → Now "2 separate benchmarks"
- Incorrect interpretation of freeze vs fine-tune

✅ **Updated references:**
- Two benchmark scripts with clear purposes
- Correct understanding of freeze vs fine-tune
- Transfer learning as quality measure

---

## Updated Workflow Examples

### Workflow 1: Fine-tuning Strategy Comparison

**Purpose:** Determine optimal embedding training strategy

```bash
# Compare: scratch vs frozen vs fine-tuned
python examples/pretrain_finetune/benchmark_embedding_finetuning.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --batch-size 128 \
    --embedding-path pretrained/med2vec_embeddings.pt
```

**Expected output:**
```
Results Summary:
  Scratch:     45% accuracy, 50 epochs to converge
  Frozen:      38% accuracy, 60 epochs (reduced capacity)
  Fine-tuned:  52% accuracy, 30 epochs (best performance)

Conclusion: Fine-tuning gives +7% absolute improvement
Recommendation: Always fine-tune pretrained embeddings
```

### Workflow 2: Transfer Learning Validation

**Purpose:** Validate that embeddings transfer across distributions

```bash
# Test: do embeddings transfer to new data?
python examples/pretrain_finetune/benchmark_transfer_learning.py \
    --model-size large \
    --source-patients 10000 \
    --target-patients 5000 \
    --epochs 100 \
    --finetune-epochs 20 \
    --batch-size 128
```

**Expected output:**
```
Results Summary:
  Source baseline:    48% accuracy on source data
  Zero-shot:          35% accuracy on target data (-13% degradation)
  Transfer learning:  45% accuracy on target data (+10% vs zero-shot)
  Target baseline:    47% accuracy on target data

Conclusion: Transfer learning recovers 92% of target baseline
Domain shift: 13% performance drop without adaptation
Transfer benefit: 10% improvement vs zero-shot
```

---

## How to Interpret Results

### Benchmark 1: Fine-tuning Strategy

**Good results:**
- Fine-tuned > Scratch by 5-15%
- Fine-tuned converges faster (20-50% fewer epochs)
- Frozen shows degradation (proof that freezing hurts)

**Poor results:**
- Fine-tuned ≤ Scratch (embeddings don't help)
- Fine-tuned converges slower (poor initialization)
- Frozen = Fine-tuned (embeddings already optimal, unlikely)

### Benchmark 2: Transfer Learning

**Good transfer learning:**
- Zero-shot degradation < 20% (domains are similar)
- Transfer learning recovers > 80% of target baseline
- Transfer beats target-from-scratch with limited data

**Poor transfer learning:**
- Zero-shot degradation > 40% (domains too different)
- Transfer learning fails to recover performance
- Target-from-scratch outperforms transfer (embeddings don't transfer)

---

## Updated Best Practices

### When to Freeze Embeddings

**Freeze embeddings when:**
- ❌ Almost never! (Reduces performance in most cases)
- ✅ Preventing catastrophic forgetting (very specific use case)
- ✅ Extremely limited compute (faster training)
- ✅ Embeddings proven optimal for target task

**Default recommendation:** Always fine-tune embeddings

### When to Use Transfer Learning

**Transfer learning helps when:**
- ✅ Limited target domain data (< 1000 patients)
- ✅ Source and target domains are related
- ✅ Source domain has more diverse/larger dataset
- ✅ Quick adaptation needed

**Transfer learning doesn't help when:**
- ❌ Target domain very different from source
- ❌ Plenty of target domain data (> 10K patients)
- ❌ Source domain too narrow/specific

---

## Technical Details

### Shared Infrastructure API

**Training function:**
```python
from ehrsequencing.benchmarks.training import train_epoch

# Works with both model types
loss, accuracy = train_epoch(
    model,           # BEHRTForMLM or standard model
    dataloader,      # Training data
    optimizer,       # PyTorch optimizer
    device          # 'cuda' or 'cpu'
)
```

**Evaluation function:**
```python
from ehrsequencing.benchmarks.training import evaluate

metrics = evaluate(
    model,
    dataloader,
    device,
    vocab_size=1000
)
# Returns: {
#     'loss': float,
#     'accuracy': float,
#     'top_5_accuracy': float,
#     'macro_f1': float,
#     'weighted_f1': float,
#     'perplexity': float
# }
```

**Model compatibility:**
- `BEHRTForMLM`: Returns `(logits, loss)`
- `BEHRTForNextVisitPrediction`: Returns `(logits, loss)`
- Standard models: Return `outputs`

The shared functions handle both formats automatically.

---

## Migration Guide

### If You Have Old Benchmark Results

**Old results from `benchmark_pretrained_embeddings.py` are still valid!**

**They answer:** "Does freezing embeddings hurt?" (Yes, by ~5% accuracy)

**To get complete picture:**
1. Re-run with new `benchmark_embedding_finetuning.py` (adds RUN 3)
2. Run `benchmark_transfer_learning.py` to test actual transfer

### If You're Writing New Benchmarks

**Use shared infrastructure:**
```python
from ehrsequencing.benchmarks.training import train_epoch, evaluate
from ehrsequencing.benchmarks import BenchmarkTracker

# No need to reimplement training/evaluation logic
# Focus on experiment design
```

**Follow naming convention:**
- `benchmark_*.py` for benchmark scripts
- Clear purpose in filename
- Document what question it answers

---

## Summary

### Key Changes

1. **Two benchmarks** instead of one
2. **Shared infrastructure** eliminates duplication
3. **Correct interpretation** of freeze vs fine-tune
4. **Transfer learning** as quality measure

### Concepts Unchanged

- BEHRT model architecture
- Embedding design (summation)
- MLM training objective
- LoRA application
- General training workflows

### Action Items for Users

✅ **Update references** from old to new script names
✅ **Use both benchmarks** for complete understanding
✅ **Fine-tune by default** (don't freeze unless specific reason)
✅ **Test transfer learning** to validate embedding quality

---

## References

**Implementation:**
- `examples/pretrain_finetune/benchmark_embedding_finetuning.py`
- `examples/pretrain_finetune/benchmark_transfer_learning.py`
- `src/ehrsequencing/benchmarks/training.py`

**Documentation:**
- `dev/workflow/EMBEDDING_BENCHMARKS_PLAN.md` - Detailed refactoring plan
- `dev/workflow/EMBEDDING_BENCHMARKS_REFACTORING_SUMMARY.md` - Implementation summary
- `dev/models/pretrain_finetune/` - This directory (tutorial docs)

---

**Last Updated:** 2026-02-03  
**Status:** Refactoring complete, documentation updated
