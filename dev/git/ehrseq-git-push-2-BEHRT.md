git commit -m "Phase 3: BEHRT transformers with LoRA and experiment tracking

Key milestones:
- Implemented BEHRT (BERT for EHR) with 3 size configs (small/medium/large)
- Added temporal embeddings (age, visit, position) for EHR sequences
- Integrated LoRA for 90-99% parameter reduction in fine-tuning
- Built comprehensive experiment tracking for ephemeral GPU pods
- Created benchmarking framework for model comparison
- Added demo training script with automatic checkpointing and visualization

Features:
- BEHRT: Transformer encoder with MLM, NVP, and classification heads
- LoRA: Efficient fine-tuning with tiny checkpoints (2MB vs 60MB)
- Tracking: Auto-save checkpoints, plots, metrics, and summaries
- Supports local dev (M1 16GB) and cloud training (A40)

Updated roadmap: Phase 2 optional, Phase 3 prioritized (2026 approach)"



---

git commit -m "Fix LoRA compatibility with PyTorch MultiheadAttention

Added weight and bias property accessors to LinearWithLoRA to expose
underlying linear layer attributes. PyTorch's MultiheadAttention expects
these attributes to be accessible on out_proj layer.

Tested successfully with train_behrt_demo.py on M1 MacBook Pro."

---

git commit -m "Fix LoRA to freeze all non-LoRA parameters

CRITICAL FIX: Previously 92.3% of parameters were trainable despite using LoRA.
Now properly freezes embeddings, encoder base weights, and MLM head.

Changes:

- Added freeze_base parameter (default: True) to apply_lora_to_behrt()
- Freezes ALL parameters before applying LoRA
- Ensures only LoRA adapters (lora_A, lora_B) are trainable
- Expected result: <10% trainable parameters instead of 92%

This should dramatically reduce overfitting by limiting trainable capacity."

---

git commit -m "Document complete LoRA debugging journey in TROUBLESHOOTING.md

Added section on the critical embedding/head freezing bug that was discovered
after the initial fix. Documents the complete debugging process:

1. Initial issue: 92% trainable params (embeddings/head not frozen)
2. First fix: Freeze all params with freeze_base=True
3. Second issue: Model not learning (loss stuck at 6.9)
4. Root cause: Embeddings and MLM head frozen at random values
5. Final fix: train_embeddings=True and train_head=True parameters

Results after complete fix:
- Training loss: 7.0 → 5.6 (vs 7.0 → 6.8 before)
- Training accuracy: ~5% (vs ~0.1% before)
- Embeddings trainable: 109,824 params
- MLM head trainable: 145,768 params

Key lesson: When training from scratch, embeddings and task heads MUST be
trainable. Only freeze them when fine-tuning pre-trained models."

---

git commit -m "Add realistic synthetic EHR data generator for showcasing BEHRT

Created a sophisticated synthetic data generator with learnable medical patterns
to properly showcase BEHRT+LoRA capabilities, addressing the limitation of
random synthetic data that only leads to overfitting.

Features:
- 8 chronic disease patterns (diabetes, hypertension, asthma, depression, etc.)
- Realistic disease progression: diagnosis → treatment → monitoring
- Co-morbidity patterns (40% diabetics have hypertension, etc.)
- Age-related disease prevalence
- Temporal progression with realistic visit spacing
- Routine care codes

Expected performance improvement:
- Random data: ~5% train acc, ~0.1% val acc (overfitting)
- Realistic data: ~40% train acc, ~30% val acc (generalization!)

Usage:
  python examples/pretrain_finetune/train_behrt_demo.py \\
      --model_size large \\
      --use_lora \\
      --lora_rank 16 \\
      --num_patients 5000 \\
      --realistic_data  # ← New flag!

Files:
- src/ehrsequencing/data/realistic_synthetic.py: Generator implementation
- src/ehrsequencing/data/README.md: Comprehensive documentation
- examples/pretrain_finetune/train_behrt_demo.py: Added --realistic_data flag"

---

git commit -m "Add comprehensive benchmark script for pre-training vs fine-tuning

Created a production-ready benchmark script to compare BEHRT training workflows
on A40 pod, maximizing GPU utilization with realistic synthetic data.

Features:
- Runs both workflows sequentially on same dataset
- Comprehensive metrics: ROC-AUC, PR-AUC, Average Precision
- Performance curves: ROC, Precision-Recall
- Training curves: Loss and accuracy over time
- Statistical comparison with winner analysis
- Automatic summary table generation

Outputs (saved to experiments/benchmark_embeddings/):
1. training_curves_comparison.png - Loss/accuracy over epochs
2. performance_metrics_comparison.png - Bar chart of metrics
3. roc_curves_comparison.png - ROC curves with AUC scores
4. pr_curves_comparison.png - Precision-Recall curves
5. SUMMARY.txt - Detailed comparison table with winner analysis
6. summary.json - Machine-readable results

Workflow:
1. Generate realistic synthetic data once (shared)
2. Train model from scratch (learn embeddings)
3. Save learned embeddings
4. Train model with frozen pre-trained embeddings
5. Compare performance metrics and curves

Usage on A40 pod:
  python benchmark_pretrained_embeddings.py \\
      --model_size large \\
      --num_patients 10000 \\
      --epochs 100 \\
      --batch_size 128

Expected insights:
- Fine-tuning converges faster (fewer epochs to best val loss)
- Pre-training may achieve slightly better final performance
- Fine-tuning requires fewer trainable parameters
- Both should generalize well with realistic data"

---

git commit -m "Rename examples/encoders to examples/pretrain_finetune for clarity

Better directory naming that clearly indicates the purpose: comparing
pre-training from scratch vs fine-tuning with pre-trained embeddings.

Changes:
- Renamed examples/encoders -> examples/pretrain_finetune
- Updated all references in Python scripts and documentation
- Clarified benchmark workflow in benchmark_pretrained_embeddings.py
- Updated README title to reflect new focus

The benchmark script workflow:
1. Run 1: Trains BEHRT from scratch, learns embeddings
2. Saves learned embeddings
3. Run 2: Loads embeddings from Run 1, freezes them, trains only LoRA+head
4. Compares performance metrics (ROC-AUC, PR-AUC, AP)

This simulates real-world scenario:
- Pre-train embeddings on large dataset (100K patients)
- Fine-tune on smaller task-specific dataset (5K patients)"



---

```
git add -A && git commit -m "Add 3-way comparison support to benchmark script

Enhancements:
- Added --external_embedding_path parameter for Med2Vec/external embeddings
- Supports 2-way comparison (default) or 3-way comparison (with external embeddings)
- Run 3 loads external embeddings, freezes them, trains only LoRA + head
- All runs use same dataset for fair comparison
- Updated docstring with clear usage examples

This answers:
- Does using pre-trained embeddings help?
- Self-learned vs external (Med2Vec) embeddings - which is better?
- Can we skip expensive pre-training with good external embeddings?

Usage:
# 2-way: python benchmark_pretrained_embeddings.py --num_patients 10000
# 3-way: python benchmark_pretrained_embeddings.py --num_patients 10000 --external_embedding_path pretrained/med2vec.pt"
```

---

```
git add -A && git commit -m "Add high-signal demo data and comprehensive metrics

Major enhancements for compelling demos:

1. New demo_synthetic.py:
   - Very strong, deterministic disease patterns
   - 70-85% accuracy achievable (vs 30-60% realistic, 0.1% random)
   - Clear diagnosis → treatment → monitoring sequences
   - Perfect for showcasing BEHRT learning capabilities

2. New metrics.py:
   - Comprehensive MLM metrics beyond accuracy
   - Top-K accuracy (clinical relevance)
   - Macro/Weighted F1 (handles class imbalance)
   - Precision/Recall (per-code and aggregate)
   - Perplexity (language modeling quality)
   - Per-code metrics for detailed analysis

3. Updated train_behrt_demo.py:
   - Added --demo_data flag for high-signal demos
   - Integrated comprehensive metrics in training loop
   - Display Accuracy, Top-5, F1 during training
   - Updated docstring with data options and metrics explanation

Why these metrics matter for medical code prediction:
- Accuracy misleading due to class imbalance (common codes dominate)
- F1 treats rare but important codes equally
- Top-K more clinically relevant (top 5 differential diagnoses)
- Perplexity measures overall prediction quality

Usage:
# High-signal demo (70%+ accuracy)
python train_behrt_demo.py --model_size large --demo_data

# Realistic evaluation (30-60% accuracy)
python train_behrt_demo.py --model_size large --realistic_data"
```

