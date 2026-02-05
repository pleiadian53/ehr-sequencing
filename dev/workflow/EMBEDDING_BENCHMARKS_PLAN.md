# Embedding Benchmarks Refactoring Plan

## Completed ✅

### 1. Shared Infrastructure Updates
- **Updated** `src/ehrsequencing/benchmarks/training.py`:
  - `train_epoch()` now handles both model types: `(logits, loss)` and `outputs`
  - `evaluate()` now handles both model types
  - Compatible with `BEHRTForMLM` and standard models

### 2. Refactored benchmark_pretrained_embeddings.py
- **Removed** duplicate functions:
  - `train_epoch` → imported from `ehrsequencing.benchmarks`
  - `evaluate` → imported from `ehrsequencing.benchmarks`
  - `compute_metrics` → imported from `ehrsequencing.benchmarks`
  - `compute_roc_curve` → imported from `ehrsequencing.benchmarks`
  - `compute_pr_curve` → imported from `ehrsequencing.benchmarks`
- **Kept** custom visualization class `CustomBenchmarkVisualizer`
- **Uses** shared `BenchmarkTracker` from `ehrsequencing.benchmarks`

## Current Understanding

### What the Current Benchmark Actually Tests

**Current script**: `benchmark_pretrained_embeddings.py`

**What it claims to test**: "Pre-training vs Fine-tuning with Pre-trained Embeddings"

**What it actually tests**: "Does freezing embeddings hurt performance?"

**Results from pod run**:
- RUN 1 (Scratch): Val Acc 36.35%, Val Loss 1.8347
- RUN 2 (Frozen): Val Acc 31.58%, Val Loss 2.0899
- **Winner**: Training from scratch (because frozen embeddings can't adapt)

**Why RUN 2 performed worse**:
1. Embeddings frozen → can't adapt to new transformer initialization
2. Same dataset → no transfer learning benefit
3. 40% fewer trainable parameters → reduced model capacity

## Next Steps 🎯

### 1. Rename & Refocus Current Script

**New name**: `benchmark_embedding_finetuning.py`

**New purpose**: Compare freeze vs fine-tune strategies

**3-way comparison**:
1. **Scratch**: Train everything from random initialization
2. **Frozen**: Load pre-trained embeddings, freeze them, train transformer
3. **Fine-tuned**: Load pre-trained embeddings, fine-tune them with transformer

**Key change**: Add RUN 3 that fine-tunes embeddings instead of freezing

**Expected result**: Fine-tuned > Scratch > Frozen

### 2. Create Transfer Learning Script

**New file**: `benchmark_transfer_learning.py`

**Purpose**: Test if embeddings transfer across datasets/time periods

**Design**:
```python
# Dataset A: Train embeddings (e.g., 2010-2015 patients)
# Dataset B: Fine-tune on different data (e.g., 2016-2020 patients)

Experiments:
1. Train on A, test on A (baseline)
2. Train on A, test on B (no adaptation)
3. Train on A, fine-tune on B, test on B (transfer learning)
4. Train on B from scratch, test on B (upper bound)
```

**Questions answered**:
- Do embeddings generalize across time periods?
- Is transfer learning better than training from scratch?
- How much fine-tuning is needed?

### 3. Update TESTING_ROADMAP.md

**Changes needed**:
- Rename Test 1.1 to "Embedding Fine-tuning Strategy Comparison"
- Add new Test 1.2: "Transfer Learning Across Datasets"
- Update descriptions to reflect actual experiments
- Remove Med2Vec requirement (optional, not critical)

## Implementation Details

### benchmark_embedding_finetuning.py Changes

```python
# RUN 1: Scratch (unchanged)
model1 = BEHRTForMLM(config)
# Train everything

# RUN 2: Frozen (unchanged)
model2 = BEHRTForMLM(config)
load_embeddings(model2, embedding_path)
freeze_embeddings(model2)  # Freeze
# Train only transformer

# RUN 3: Fine-tuned (NEW)
model3 = BEHRTForMLM(config)
load_embeddings(model3, embedding_path)
# DON'T freeze - allow fine-tuning
# Train everything (embeddings + transformer)
```

### benchmark_transfer_learning.py Structure

```python
# Generate two datasets with different distributions
dataset_A = generate_realistic_dataset(patients=5000, seed=42)
dataset_B = generate_realistic_dataset(patients=5000, seed=123)

# RUN 1: Train on A, test on A
# RUN 2: Train on A, test on B (no adaptation)
# RUN 3: Train on A, fine-tune on B, test on B
# RUN 4: Train on B, test on B (from scratch)
```

## Timeline

1. ✅ Refactor shared infrastructure (DONE)
2. ⏳ Rename & update current script (IN PROGRESS)
3. ⏳ Create transfer learning script
4. ⏳ Update TESTING_ROADMAP.md
5. ⏳ Test locally
6. ⏳ Run on pod

## Notes

- Markdown lints in `BEHRT_SURVIVAL_ANALYSIS_DESIGN.md` are cosmetic, ignore
- The refactoring makes all benchmarking scripts consistent
- Transfer learning is the real test of embedding quality
- Current results are valid but answer a different question than intended
