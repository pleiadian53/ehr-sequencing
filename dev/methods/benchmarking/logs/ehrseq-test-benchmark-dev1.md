The script `benchmark_pretrained_embeddings.py` doesn't exist yet. Let me check what embedding-related benchmarking scripts we have:







I see there's a duplicate "Test 1.3" section. Let me fix that:

The markdown lints are minor formatting issues (trailing spaces and list spacing). I'll move on and commit the changes.







## ✅ TESTING_ROADMAP.md Updated

### Summary

Fixed the issue with Test 1.3 referencing a non-existent script.

### Changes Made

**Test 1.3: Embedding Quality Analysis**
- ✅ Clarified that embedding quality is already covered by Tests 1.1 and 1.2
- ✅ Explained how each test evaluates embedding quality:
  - Test 1.1: Scratch vs Frozen vs Fine-tuned comparison
  - Test 1.2: Transfer learning and domain shift robustness
- ✅ Added optional Python code for deeper embedding visualization (t-SNE/UMAP)
- ✅ Noted that `benchmark_pretrained_embeddings.py` is planned but not yet implemented

**Test 1.4: Quick Validation** (renamed from duplicate Test 1.3)
- ✅ Updated to reference correct existing script: [benchmark_embedding_finetuning.py](cci:7://file:///Users/pleiadian53/work/ehr-sequencing/examples/pretrain_finetune/benchmark_embedding_finetuning.py:0:0-0:0)
- ✅ Added proper command with small model settings for quick validation
- ✅ Clarified purpose: sanity check before long runs

### Current Test Structure

```
Phase 1: Pre-training and Fine-tuning
├── Test 1.1: Embedding Fine-tuning (benchmark_embedding_finetuning.py) ✅
├── Test 1.2: Transfer Learning (benchmark_transfer_learning.py) ✅
├── Test 1.3: Embedding Quality Analysis (covered by 1.1 & 1.2) ✅
└── Test 1.4: Quick Validation (benchmark_embedding_finetuning.py --small) ✅
```

All tests now reference **existing, working scripts**. No broken references remain.



---

