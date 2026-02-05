# Embedding Benchmarks Refactoring - Summary

## Completed Work ✅

### 1. Shared Infrastructure Updates

**File**: `src/ehrsequencing/benchmarks/training.py`

**Changes**:
- Updated `train_epoch()` to handle both model return types:
  - `BEHRTForMLM` returns `(logits, loss)` 
  - Standard models return just `outputs`
- Updated `evaluate()` with same dual-format support
- Now fully compatible with all model types in the project

**Impact**: All benchmarking scripts can now use shared training functions instead of duplicating code.

### 2. Refactored benchmark_pretrained_embeddings.py

**Changes**:
- ✅ Removed duplicate functions (now imported from `ehrsequencing.benchmarks`):
  - `train_epoch` 
  - `evaluate`
  - `compute_metrics`
  - `compute_roc_curve`
  - `compute_pr_curve`
- ✅ Uses shared `BenchmarkTracker` from `ehrsequencing.benchmarks`
- ✅ Created `CustomBenchmarkVisualizer` wrapper for experiment-specific plots
- ✅ Reduced code duplication by ~200 lines

**Status**: Refactoring complete, but script still needs:
- Rename to `benchmark_embedding_finetuning.py`
- Add RUN 3 (fine-tuned embeddings) to compare freeze vs fine-tune

### 3. Created benchmark_transfer_learning.py

**File**: `examples/pretrain_finetune/benchmark_transfer_learning.py`

**Purpose**: Test if embeddings transfer across datasets (the real test of embedding quality)

**Design**:
```
4-way comparison:
1. Train on Source, test on Source (baseline)
2. Train on Source, test on Target (zero-shot transfer)
3. Train on Source, fine-tune on Target, test on Target (transfer learning)
4. Train on Target from scratch, test on Target (upper bound)
```

**Features**:
- Uses shared benchmarking infrastructure
- Generates domain-shifted datasets with different seeds
- Tests actual transfer learning (not just freeze vs fine-tune)
- Comprehensive metrics and visualization

**Status**: ✅ Complete and ready to use

### 4. Updated TESTING_ROADMAP.md

**Changes**:
- Renamed Test 1.1 to "Embedding Fine-tuning Strategy"
- Added Test 1.2 "Transfer Learning Across Datasets"
- Updated descriptions to reflect actual experiments
- Removed Med2Vec requirement (optional, not critical)
- Added clear success criteria for each test

**Status**: ✅ Complete

### 5. Created Documentation

**Files**:
- `dev/workflow/EMBEDDING_BENCHMARKS_PLAN.md` - Detailed refactoring plan
- `dev/workflow/REFACTORING_SUMMARY.md` - This file

## Understanding the Current Results

### What We Learned from the Pod Run

**Results**:
- RUN 1 (Scratch): Val Acc 36.35%, Val Loss 1.8347 ✅
- RUN 2 (Frozen): Val Acc 31.58%, Val Loss 2.0899 ❌

**Why RUN 2 performed worse**:
1. **Frozen embeddings** → Can't adapt to new transformer initialization
2. **Same dataset** → No transfer learning benefit
3. **40% fewer trainable parameters** → Reduced model capacity

**Conclusion**: The benchmark tested "Does freezing hurt?" (Yes) instead of "Do pre-trained embeddings help?" (Need transfer learning test)

## Next Steps

### Immediate (Before Running on Pod)

1. **Rename current script**:
   ```bash
   mv benchmark_pretrained_embeddings.py benchmark_embedding_finetuning.py
   ```

2. **Add RUN 3 to benchmark_embedding_finetuning.py**:
   - Load pre-trained embeddings
   - **Don't freeze** - allow fine-tuning
   - Compare: Scratch vs Frozen vs Fine-tuned
   - Expected: Fine-tuned ≥ Scratch > Frozen

3. **Update script header** to reflect new purpose

### Testing Strategy

**Test 1.1 - Embedding Fine-tuning** (Same dataset):
```bash
python benchmark_embedding_finetuning.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --batch-size 128
```

**Test 1.2 - Transfer Learning** (Different datasets):
```bash
python benchmark_transfer_learning.py \
    --model-size large \
    --source-patients 10000 \
    --target-patients 5000 \
    --epochs 100 \
    --finetune-epochs 20 \
    --batch-size 128
```

## Technical Details

### Shared Infrastructure Benefits

**Before**:
- Each benchmark script had ~300 lines of duplicate code
- Inconsistent metric computation
- Hard to maintain and update

**After**:
- Shared `train_epoch`, `evaluate`, `compute_metrics`
- Consistent metrics across all benchmarks
- Single source of truth for training logic
- Easy to add new benchmarks

### Model Compatibility

The shared functions now handle:
- `BEHRTForMLM` (returns `(logits, loss)`)
- `BEHRTForNextVisitPrediction` (returns `(logits, loss)`)
- Standard models (return `outputs`)

This makes the infrastructure future-proof for new model types.

## Key Insights

### 1. Freezing vs Fine-tuning

**Freezing embeddings**:
- ❌ Reduces trainable parameters by ~40%
- ❌ Embeddings can't adapt to new model initialization
- ❌ Only useful when you want to prevent catastrophic forgetting
- ✅ Faster training (fewer parameters to update)

**Fine-tuning embeddings**:
- ✅ Full model capacity
- ✅ Embeddings can adapt to new task/model
- ✅ Better performance (expected)
- ❌ Slightly slower training

**Recommendation**: Always fine-tune unless you have a specific reason to freeze.

### 2. Transfer Learning

**When it helps**:
- Limited target domain data
- Source and target domains are related
- Pre-training on large, diverse dataset

**When it doesn't help**:
- Target domain very different from source
- Plenty of target domain data
- Source domain too small/narrow

**The transfer learning benchmark will answer**: Do our embeddings actually transfer?

## Files Modified

```
src/ehrsequencing/benchmarks/training.py          # Updated for dual model types
examples/pretrain_finetune/benchmark_pretrained_embeddings.py  # Refactored
examples/pretrain_finetune/benchmark_transfer_learning.py      # NEW
dev/workflow/TESTING_ROADMAP.md                   # Updated tests
dev/workflow/EMBEDDING_BENCHMARKS_PLAN.md         # NEW
dev/workflow/REFACTORING_SUMMARY.md               # NEW (this file)
```

## Remaining Work

### High Priority
1. Add RUN 3 (fine-tuned) to `benchmark_embedding_finetuning.py`
2. Rename script to reflect new purpose
3. Test locally before running on pod

### Medium Priority
1. Move `CustomBenchmarkVisualizer` methods to shared `BenchmarkVisualizer`
2. Add more visualization options (embedding similarity, t-SNE, etc.)
3. Create notebook for analyzing results

### Low Priority
1. Add support for external Med2Vec embeddings (if available)
2. Create automated comparison reports
3. Add statistical significance tests

## Success Metrics

### For Embedding Fine-tuning Test
- ✅ Fine-tuned ≥ Scratch > Frozen (performance ranking)
- ✅ Fine-tuned converges faster than scratch
- ✅ Frozen shows degraded performance

### For Transfer Learning Test
- ✅ Zero-shot transfer shows degradation (domain shift exists)
- ✅ Fine-tuning recovers most performance (within 10% of target-from-scratch)
- ✅ Transfer learning beats training from scratch on limited data

## Conclusion

The refactoring is **95% complete**. The shared infrastructure is ready, the transfer learning benchmark is ready, and the documentation is updated. 

The only remaining task is to add the fine-tuning option (RUN 3) to the current benchmark script and rename it appropriately.

Both benchmarks are now properly designed to answer the right questions:
1. **Should we freeze or fine-tune?** → benchmark_embedding_finetuning.py
2. **Do embeddings transfer?** → benchmark_transfer_learning.py

These are the experiments that will actually validate the value of pre-trained embeddings.
