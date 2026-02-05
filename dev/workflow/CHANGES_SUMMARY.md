# Changes Summary - Embedding Benchmarks Refactoring

**Date**: February 3, 2026

## Completed Tasks ✅

### 1. Renamed Script
- **Old**: `benchmark_pretrained_embeddings.py`
- **New**: `benchmark_embedding_finetuning.py`
- **Reason**: Better reflects the actual purpose - comparing freeze vs fine-tune strategies

### 2. Updated Script Header
- Changed from "Pre-training vs Fine-tuning" to "Embedding Fine-tuning Strategy Comparison"
- Clarified the 3-way comparison:
  1. Train from scratch (baseline)
  2. Load pre-trained embeddings, FREEZE them (reduced capacity)
  3. Load pre-trained embeddings, FINE-TUNE them (transfer learning)
- Added expected results section
- Updated usage examples with new script name

### 3. Added RUN 3 - Fine-tuned Embeddings
- **RUN 1**: Train from scratch (unchanged)
- **RUN 2**: Load embeddings, freeze them (unchanged)
- **RUN 3**: Load embeddings, fine-tune them (NEW)
  - Uses same embeddings as RUN 2 but allows fine-tuning
  - `freeze=False` and `train_embeddings=True`
  - Full model capacity (all parameters trainable)
- **RUN 4**: External embeddings (optional, unchanged)

### 4. Updated TESTING_ROADMAP.md
- Test 1.1 now references `benchmark_embedding_finetuning.py`
- Added note about 3-way comparison in command section
- Maintained all other test descriptions

### 5. Renamed Summary File
- **Old**: `REFACTORING_SUMMARY.md`
- **New**: `EMBEDDING_BENCHMARKS_REFACTORING_SUMMARY.md`
- **Reason**: More specific about what it documents (embedding benchmarks refactoring)

## Expected Results

When running the updated script, you should see:

**Performance Ranking**: Fine-tuned ≥ Scratch > Frozen

**Why**:
- **Fine-tuned**: Benefits from pre-trained initialization + full capacity
- **Scratch**: Full capacity but random initialization
- **Frozen**: Pre-trained initialization but reduced capacity (40% fewer params)

## Files Modified

1. `examples/pretrain_finetune/benchmark_embedding_finetuning.py` (renamed + updated)
2. `dev/workflow/TESTING_ROADMAP.md` (updated script name)
3. `dev/workflow/EMBEDDING_BENCHMARKS_REFACTORING_SUMMARY.md` (renamed)

## Next Steps

Ready to run on pod:

```bash
cd /workspace/ehr-sequencing/examples/pretrain_finetune

nohup python -u benchmark_embedding_finetuning.py \
    --model-size large \
    --num-patients 10000 \
    --epochs 100 \
    --batch-size 128 \
    --output-dir /workspace/ehr-sequencing/experiments/embedding_finetuning \
    > /workspace/ehr-sequencing/experiments/sessions/embedding_finetuning_large.out 2>&1 &
```

This will now produce a proper 3-way comparison showing the value of fine-tuning vs freezing embeddings.
