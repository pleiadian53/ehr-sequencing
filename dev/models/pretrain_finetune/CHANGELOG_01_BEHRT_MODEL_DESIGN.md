# Changelog: 01_behrt_model_design.md

**Date:** 2026-02-03  
**Reason:** Updated embedding freeze defaults based on benchmark findings

---

## Changes Made

### 1. Updated Default: `freeze=False` (Line 337)

**Before:**
```python
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    pretrained_embeddings=embeddings,
    freeze=True  # Freeze embeddings during training
)
```

**After:**
```python
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    pretrained_embeddings=embeddings,
    freeze=False  # Fine-tune embeddings (recommended default)
)
```

### 2. Added Best Practice Section

**New section after Word2Vec example (lines 359-395):**

Added comprehensive guidance on freeze vs fine-tune decision:

**Key points:**
- ✅ **Fine-tune by default** (`freeze=False`)
- ✅ Better performance (5-15% improvement)
- ✅ Faster convergence (20-50% fewer epochs)
- ❌ **Freeze only for specific use cases**
- ❌ Freezing reduces performance (40% fewer trainable params)

**When to freeze:**
- Prevent catastrophic forgetting
- Extremely limited compute
- Embeddings proven optimal

**When to fine-tune (default):**
- Standard training
- Want best performance
- Transfer learning

**References:** `benchmark_embedding_finetuning.py` and `06_benchmarking_updates.md`

### 3. Updated Summary Section (Line 984)

**Before:**
```
3. **Pretrained Embeddings**: Support for Med2Vec, Word2Vec, and custom embeddings via `initialize_embedding_layer()`
```

**After:**
```
3. **Pretrained Embeddings**: Support for Med2Vec, Word2Vec, and custom embeddings via `initialize_embedding_layer(..., freeze=False)` (fine-tuning recommended by default)
```

### 4. Added Benchmarking Reference (Lines 1035-1038)

**Before:**
```
**Next:** See `02_training_guide.md` for detailed training instructions and best practices.
```

**After:**
```
**Next:**
- `02_training_guide.md` - Detailed training instructions and best practices
- `03_pretrained_embeddings_workflow.md` - Complete pretrained embeddings workflow
- `06_benchmarking_updates.md` - Freeze vs fine-tune comparison (benchmark results)
```

---

## Why These Changes?

### Problem

Original documentation showed `freeze=True` as the default example, which contradicts:
1. **Benchmark findings** - Fine-tuning outperforms freezing by 5-15%
2. **Implementation reality** - Most examples already use `freeze=False`
3. **Best practices** - Transfer learning works best with fine-tuning

### Solution

Updated documentation to:
1. Use `freeze=False` as default
2. Explain when to freeze vs fine-tune
3. Reference benchmark evidence
4. Guide users to correct decision

---

## Impact

### What Changed

✅ Default example now shows `freeze=False`  
✅ Added clear guidance on freeze vs fine-tune  
✅ Referenced benchmark results  
✅ Consistent with other examples in the doc

### What Stayed the Same

✔️ All model implementations (code unchanged)  
✔️ Architecture explanations (concepts unchanged)  
✔️ LoRA application (workflow unchanged)  
✔️ Other pretrained embedding examples (already correct)

---

## Verification

All `freeze` parameters in the document:

| Line | Value | Status | Context |
|------|-------|--------|---------|
| 337 | `False` | ✅ Updated | Med2Vec example |
| 380 | `False` | ✅ Correct | Workflow example |
| 811 | `False` | ✅ Correct | Approach 1 |
| 829 | `False` | ✅ Correct | Approach 2 |
| 849 | `False` | ✅ Correct | Approach 3 |
| 1008 | `False` | ✅ Correct | Pattern 2 |

**All freeze parameters now consistently use `False` (fine-tuning).**

---

## References

- **Benchmark script:** `examples/pretrain_finetune/benchmark_embedding_finetuning.py`
- **Documentation:** `dev/models/pretrain_finetune/06_benchmarking_updates.md`
- **Workflow plan:** `dev/workflow/EMBEDDING_BENCHMARKS_PLAN.md`

---

## Summary

**Change:** Updated default from `freeze=True` to `freeze=False` with clear best practice guidance.

**Reason:** Benchmarking shows fine-tuning embeddings significantly outperforms freezing them.

**Impact:** Documentation now aligns with evidence-based best practices and implementation reality.

**Status:** ✅ Complete and verified
