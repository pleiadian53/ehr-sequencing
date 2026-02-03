# BEHRT Pretrain/Finetune Documentation

**Last Updated:** 2026-02-02  
**Location:** `dev/models/pretrain_finetune/`  
**Status:** Private development notes (not for GitHub)

## Overview

This directory contains comprehensive internal documentation for the BEHRT model implementation, training workflows, and pretrained embeddings integration in the EHR-sequencing project.

## Documentation Structure

### 📘 [01_behrt_model_design.md](./01_behrt_model_design.md)

**Comprehensive model architecture walkthrough**

**Topics covered:**
- BEHRT architecture and components
- Model size configurations (small/medium/large)
- Embedding design (code + age + visit + position)
- Task-specific heads (MLM, NVP, Classification)
- LoRA (Low-Rank Adaptation) implementation
- Pretrained embeddings integration
- Parameter counting and efficiency

**Key questions answered:**
- ✅ Does BEHRT use pretrained models from HuggingFace?
- ✅ How to provide pretrained embeddings (e.g., from Med2Vec)?
- ✅ How to apply LoRA to a foundation model?
- ✅ What are the model design patterns?

**Read this first if:**
- New to the codebase
- Want to understand model architecture
- Need to modify model components
- Integrating pretrained embeddings

---

### 🚀 [02_training_guide.md](./02_training_guide.md)

**Practical training instructions and best practices**

**Topics covered:**
- Quick start commands
- Command-line usage patterns
- Data generation (demo vs. realistic)
- Hyperparameter tuning guide
- Training strategies (from scratch, with embeddings, fine-tuning)
- Monitoring and debugging
- Common issues and solutions

**Key sections:**
- Auto-resource detection
- Model size selection
- Learning rate scheduling
- Early stopping
- Experiment tracking
- Troubleshooting OOM errors

**Read this if:**
- Ready to train models
- Need to tune hyperparameters
- Encountering training issues
- Want to optimize training workflow

---

### 🔗 [03_pretrained_embeddings_workflow.md](./03_pretrained_embeddings_workflow.md)

**Complete guide for using pretrained embeddings**

**Topics covered:**
- Why use pretrained embeddings?
- Med2Vec integration (step-by-step)
- Word2Vec integration
- Custom embeddings
- Embedding analysis and visualization
- Complete workflow examples
- Best practices

**Key workflows:**
- Loading Med2Vec embeddings
- Initializing BEHRT with pretrained weights
- Comparing random vs. pretrained initialization
- Monitoring embedding drift
- Saving/loading trained embeddings

**Read this if:**
- Have Med2Vec or Word2Vec embeddings
- Want faster convergence
- Need transfer learning
- Analyzing embedding quality

---

### ⚠️ [04_clarifications_and_corrections.md](./04_clarifications_and_corrections.md)

**Critical clarifications about BEHRT**

**Topics covered:**
- **Is BEHRT autoregressive?** NO - it's bidirectional
- **Does BEHRT do survival analysis?** NO - it does MLM
- BEHRT vs GPT architecture comparison
- BEHRT vs LSTM objective differences
- Common misconceptions and corrections

**Key clarifications:**
- BEHRT uses bidirectional attention (like BERT)
- Training objective is MLM (Masked Language Modeling)
- Survival analysis is done with LSTM models separately
- BEHRT uses TransformerEncoder, not Decoder

**READ THIS FIRST if:**
- Confused about BEHRT's architecture
- Think BEHRT is autoregressive
- Think BEHRT does survival analysis
- Need to understand BEHRT vs other models

---

### 🔬 [05_embedding_summation_and_quality_analysis.md](./05_embedding_summation_and_quality_analysis.md)

**Deep dive into embedding design and quality metrics**

**Topics covered:**
- **Why sum embeddings?** Mathematical justification
- **Does summation preserve information?** Yes, with proof
- Med2Vec evaluation methodology
- BEHRT evaluation standards
- Medical embedding quality metrics
- How to assess embedding quality

**Key analyses:**
- Mathematical proof that summation preserves visit/position information
- Fourier analysis perspective on embedding superposition
- Industry standards for medical embedding evaluation
- Why 31% MLM accuracy is excellent performance
- How to evaluate Med2Vec embedding quality
- Intrinsic vs extrinsic evaluation metrics

**Read this if:**
- Curious about embedding architecture choices
- Want to understand why summation works
- Need to evaluate embedding quality
- Comparing different initialization strategies
- Understanding your benchmark results

---

## Quick Navigation

### By Task

**I want to...**

- **Understand the model** → Read `01_behrt_model_design.md`
- **Clarify misconceptions** → Read `04_clarifications_and_corrections.md` ⚠️
- **Train a model** → Read `02_training_guide.md`
- **Use pretrained embeddings** → Read `03_pretrained_embeddings_workflow.md`
- **Modify the architecture** → Read `01_behrt_model_design.md` (sections 1-2)
- **Apply LoRA** → Read `01_behrt_model_design.md` (section 4)
- **Debug training issues** → Read `02_training_guide.md` (section 7)
- **Tune hyperparameters** → Read `02_training_guide.md` (section 5)
- **Compare initialization strategies** → Read `03_pretrained_embeddings_workflow.md` (section 6)

### By Experience Level

**Beginner (new to project):**
1. Start with `01_behrt_model_design.md` (Overview, Model Architecture)
2. Run quick start from `02_training_guide.md`
3. Explore `03_pretrained_embeddings_workflow.md` if you have embeddings

**Intermediate (familiar with basics):**
1. Deep dive into specific sections of `01_behrt_model_design.md`
2. Explore training strategies in `02_training_guide.md`
3. Experiment with pretrained embeddings from `03_pretrained_embeddings_workflow.md`

**Advanced (modifying codebase):**
1. Reference implementation details in `01_behrt_model_design.md`
2. Use as reference while coding
3. Consult best practices from all documents

---

## Key Files Referenced

### Source Code

```
src/ehrsequencing/models/
├── behrt.py                    # BEHRT model implementations
├── embeddings.py               # Temporal embedding layers
├── lora.py                     # LoRA adaptation
└── pretrained_embeddings.py    # Embedding utilities
```

### Training Scripts

```
examples/pretrain_finetune/
├── train_behrt_demo.py         # Main training script
├── train_behrt_finetune.py     # Fine-tuning script
└── benchmark_pretrained_embeddings.py
```

### Utilities

```
src/ehrsequencing/
├── data.py                     # Data generation
├── utils/
│   ├── experiment_tracker.py   # Experiment logging
│   ├── metrics.py              # Evaluation metrics
│   └── resource_manager.py     # Auto-resource detection
```

---

## Quick Start Example

### Minimal Training Example

```bash
# Navigate to project root
cd /Users/pleiadian53/work/ehr-sequencing

# Run with auto-detection (recommended)
python examples/pretrain_finetune/train_behrt_demo.py
```

### Training with Med2Vec Embeddings

```python
from ehrsequencing.models.behrt import BEHRTConfig, BEHRTForMLM
from ehrsequencing.models.lora import apply_lora_to_behrt
from ehrsequencing.models.pretrained_embeddings import (
    load_med2vec_embeddings,
    initialize_embedding_layer
)

# 1. Load embeddings
embeddings = load_med2vec_embeddings('med2vec.pt', 1000, 256)

# 2. Create model
config = BEHRTConfig.large(vocab_size=1000)
model = BEHRTForMLM(config)

# 3. Initialize embeddings
model.behrt.embeddings.code_embedding = initialize_embedding_layer(
    model.behrt.embeddings.code_embedding,
    embeddings,
    freeze=False
)

# 4. Apply LoRA
model = apply_lora_to_behrt(model, rank=16, train_embeddings=True, train_head=True)

# 5. Train...
```

---

## Key Concepts

### BEHRT Architecture

```
Input Codes → Embeddings → Transformer → Task Head → Output
             ↓
    Code + Age + Visit + Position
```

### LoRA Adaptation

```
Original: h = Wx
LoRA: h = Wx + (BA)x
      where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
```

**Benefits:**
- 90-99% reduction in trainable parameters
- Maintains performance
- Faster training
- Smaller checkpoints

### Training Patterns

| Pattern | Scenario | Embeddings | Head | LoRA Rank |
|---------|----------|------------|------|-----------|
| From Scratch | No pretrained weights | Trainable | Trainable | 16 |
| With Med2Vec | Pretrained embeddings | Trainable | Trainable | 16 |
| Fine-tuning | Pretrained BEHRT | Frozen | Trainable | 8 |

---

## Common Workflows

### Workflow 1: Standard Pre-training

```bash
# Auto-detect resources, use demo data
python examples/pretrain_finetune/train_behrt_demo.py
```

**Output:** `experiments/behrt_<size>_mlm_lora<rank>/`

### Workflow 2: Custom Configuration

```bash
# Specify model size and data
python examples/pretrain_finetune/train_behrt_demo.py \
    --model-size large \
    --batch-size 128 \
    --num-patients 10000 \
    --realistic-data
```

### Workflow 3: With Pretrained Embeddings

See detailed instructions in `03_pretrained_embeddings_workflow.md`

---

## Performance Expectations

### Model Capacity

| Model | Parameters | Memory | Training Time | Accuracy |
|-------|-----------|---------|---------------|----------|
| Small | ~2M | 1GB | Fast | 70-75% |
| Medium | ~10M | 4GB | Medium | 75-80% |
| Large | ~26M | 12GB | Slow | 80-85% |

### Data Quality Impact

| Data Type | Accuracy | Top-5 | Convergence |
|-----------|----------|-------|-------------|
| Demo | 70-85% | 85-95% | 20-30 epochs |
| Realistic | 30-60% | 60-80% | 40-60 epochs |

### Initialization Impact

| Initialization | Epochs | Final Acc | Time Saved |
|----------------|--------|-----------|------------|
| Random | 50-100 | 75% | Baseline |
| Med2Vec | 20-30 | 82% | 50% faster |
| Word2Vec | 30-50 | 78% | 30% faster |

---

## Tips and Best Practices

### ✅ Do's

1. **Start with auto-detection** - Let the script configure parameters
2. **Use demo data first** - Verify training works before realistic data
3. **Enable LoRA by default** - Efficient and effective
4. **Monitor early epochs** - Catch issues quickly
5. **Save checkpoints** - Use experiment tracking
6. **Keep embeddings trainable** - When training from scratch
7. **Use pretrained embeddings** - When available

### ❌ Don'ts

1. **Don't skip validation** - Always use validation set
2. **Don't ignore OOM errors** - Reduce batch size or model size
3. **Don't train without tracking** - Use ExperimentTracker
4. **Don't freeze embeddings** - When training from scratch
5. **Don't use too high LR** - Start with 1e-4
6. **Don't skip early stopping** - Prevents overfitting
7. **Don't modify all hyperparameters at once** - Change one at a time

---

## Troubleshooting

### Common Issues

| Issue | Solution | Document |
|-------|----------|----------|
| OOM error | Reduce batch size | `02_training_guide.md` (section 9) |
| Not learning | Check data/LR | `02_training_guide.md` (section 7) |
| Overfitting | Increase dropout | `02_training_guide.md` (section 8) |
| Slow convergence | Use pretrained embeddings | `03_pretrained_embeddings_workflow.md` |
| Loss is NaN | Lower LR, clip gradients | `02_training_guide.md` (section 9) |

### Debug Checklist

- [ ] Data generation working?
- [ ] Masking applied correctly?
- [ ] Loss function configured?
- [ ] Learning rate appropriate?
- [ ] GPU memory sufficient?
- [ ] Validation set separate?
- [ ] Metrics improving?

---

## Related Documentation

### Public Documentation (docs/)

- `docs/BEHRT/README.md` - Public BEHRT overview
- `docs/pretrain_finetune/` - Public training guides
- `examples/pretrain_finetune/README.md` - Public examples

### Private Notes (dev/)

- `dev/models/pretrain_finetune/` - **This directory**
- `dev/methods/` - Training methodology notes
- `dev/experiments/` - Experiment logs

---

## Questions?

### Clarifications

- "Is BEHRT autoregressive?" → `04_clarifications_and_corrections.md` (section 1) ⚠️
- "Does BEHRT do survival analysis?" → `04_clarifications_and_corrections.md` (section 2) ⚠️
- "BEHRT vs GPT?" → `04_clarifications_and_corrections.md` (section 1)
- "BEHRT vs LSTM?" → `04_clarifications_and_corrections.md` (section 2)

### Embedding Design

- "Why sum embeddings?" → `05_embedding_summation_and_quality_analysis.md` (section 1) 🔬
- "Does summation preserve information?" → `05_embedding_summation_and_quality_analysis.md` (section 1)
- "How to evaluate embedding quality?" → `05_embedding_summation_and_quality_analysis.md` (section 2)
- "Is Med2Vec good quality?" → `05_embedding_summation_and_quality_analysis.md` (section 2)
- "Why is 31% accuracy good?" → `05_embedding_summation_and_quality_analysis.md` (section 2)

### Model Design

- "How does the embedding layer work?" → `01_behrt_model_design.md` (section 2)
- "What is LoRA?" → `01_behrt_model_design.md` (section 4)
- "How to apply LoRA?" → `01_behrt_model_design.md` (section 4.3)

### Training

- "How to train?" → `02_training_guide.md` (section 1)
- "What hyperparameters?" → `02_training_guide.md` (section 5)
- "Model not learning?" → `02_training_guide.md` (section 7)

### Embeddings

- "Use Med2Vec?" → `03_pretrained_embeddings_workflow.md` (section 2)
- "Load embeddings?" → `03_pretrained_embeddings_workflow.md` (section 2.2)
- "Compare initialization?" → `03_pretrained_embeddings_workflow.md` (section 6.2)

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-02 | Initial documentation |
| 1.1 | 2026-02-03 | Added clarifications document addressing misconceptions |
| 1.2 | 2026-02-03 | Added embedding summation analysis and quality metrics |

---

## Feedback

These are internal development notes. For questions or improvements:
1. Add notes to specific document sections
2. Create new documents as needed
3. Update this README when adding new files

---

**Happy training! 🚀**
