> On A40 pod

```
nohup python train_behrt_demo.py --demo-data > experiments/nohup_demo.out 2>&1 & tail -f experiments/nohup_demo.out
```

---





## Summary of Overfitting Diagnosis & Fixes

**Problem Confirmed:**
- Train accuracy: 60.8% vs Val accuracy: 32.2% (**-28.6% gap**)
- Train loss: 1.23 vs Val loss: 2.06 (**+67% gap**)
- Severe overfitting - model memorizing training data

**Root Causes Identified:**
1. ❌ **Random synthetic data** (no learnable patterns - just noise)
2. ❌ **Low dropout** (0.1 too weak for large model)
3. ⚠️ Model capacity vs dataset size mismatch

**Fixes Implemented:**
1. ✅ Increased default dropout: **0.1 → 0.2**
2. ✅ Changed default data: **random → demo data** (70%+ accuracy expected)
3. ✅ Updated docstring and usage examples

**Ready to commit when you return!** 🎯





---

