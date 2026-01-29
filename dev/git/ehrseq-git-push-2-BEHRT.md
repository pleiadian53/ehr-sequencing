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

