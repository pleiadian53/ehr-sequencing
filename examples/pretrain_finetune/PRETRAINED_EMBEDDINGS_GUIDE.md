# Pre-trained Embeddings Guide for BEHRT

This guide covers how to use pre-trained medical code embeddings with BEHRT, including sources, best practices, and general principles for foundation model fine-tuning.

---

## Table of Contents

1. [Why Use Pre-trained Embeddings?](#why-use-pre-trained-embeddings)
2. [Sources of Pre-trained Medical Embeddings](#sources-of-pre-trained-medical-embeddings)
3. [Downloading from HuggingFace](#downloading-from-huggingface)
4. [Using Pre-trained Embeddings with BEHRT](#using-pre-trained-embeddings-with-behrt)
5. [General Best Practices](#general-best-practices)
6. [Foundation Model Fine-tuning Tips](#foundation-model-fine-tuning-tips)
7. [Troubleshooting](#troubleshooting)

---

## Why Use Pre-trained Embeddings?

### Benefits

**1. Reduced Data Requirements**
- Pre-trained embeddings encode medical knowledge from large corpora
- Can achieve good performance with 10-100x less data
- Example: 1K-10K patients vs 100K+ patients needed for training from scratch

**2. Faster Convergence**
- Model starts with meaningful representations
- Fewer epochs needed to reach optimal performance
- Reduced training time and compute costs

**3. Better Generalization**
- Pre-trained on diverse medical data
- Captures medical code relationships (e.g., diabetes → insulin)
- Less prone to overfitting on small datasets

**4. Transfer Learning**
- Leverage knowledge from related tasks
- Domain adaptation from general medical codes to specific conditions
- Cross-institution knowledge transfer

### When to Use Pre-trained vs Train from Scratch

| Scenario | Recommendation |
|----------|----------------|
| **Small dataset** (<10K patients) | ✅ Use pre-trained |
| **Large dataset** (>100K patients) | Either (pre-trained still faster) |
| **Limited compute** | ✅ Use pre-trained |
| **Novel medical codes** not in pre-training | ❌ Train from scratch |
| **Domain shift** (different code system) | ⚠️ Evaluate both |
| **Quick prototyping** | ✅ Use pre-trained |

---

## Sources of Pre-trained Medical Embeddings

### 1. HuggingFace Hub 🤗

**Popular Medical Embeddings:**

- **ClinicalBERT**: Pre-trained on MIMIC-III clinical notes
  - Model: `emilyalsentzer/Bio_ClinicalBERT`
  - Use case: Clinical text embeddings

- **BioBERT**: Pre-trained on PubMed and PMC
  - Model: `dmis-lab/biobert-v1.1`
  - Use case: Biomedical literature

- **Med2Vec**: Medical code embeddings
  - Search: "med2vec" or "medical code embeddings"
  - Use case: ICD/CPT code embeddings

- **CEHR-BERT**: EHR-specific BERT
  - Search: "ehr bert" or "electronic health records"
  - Use case: Structured EHR data

**How to Search:**
```python
from huggingface_hub import list_models

# Search for medical embeddings
models = list_models(filter="medical")
for model in models:
    print(f"{model.modelId}: {model.downloads} downloads")
```

### 2. Published Research

**Med2Vec (Choi et al., 2016)**
- Paper: "Multi-layer Representation Learning for Medical Concepts"
- GitHub: https://github.com/mp2893/med2vec
- Pre-trained on MIMIC-III

**GRAM (Choi et al., 2017)**
- Paper: "GRAM: Graph-based Attention Model for Healthcare Representation Learning"
- Incorporates medical ontology structure

**BEHRT (Li et al., 2020)**
- Paper: "BEHRT: Transformer for Electronic Health Records"
- Pre-trained on large EHR datasets

### 3. Public Datasets for Training Your Own

If pre-trained embeddings aren't available:

- **MIMIC-III/IV**: Critical care database (requires credentialing)
- **eICU**: Multi-center ICU database
- **UK Biobank**: Large-scale biomedical database
- **All of Us**: NIH research program

---

## Downloading from HuggingFace

### Method 1: Using `huggingface_hub` (Recommended)

```python
from huggingface_hub import hf_hub_download
import torch

# Download pre-trained embeddings
embedding_path = hf_hub_download(
    repo_id="username/medical-code-embeddings",
    filename="embeddings.pt",
    cache_dir="./pretrained"
)

# Load embeddings
embeddings = torch.load(embedding_path)
print(f"Loaded embeddings: {embeddings.shape}")
```

### Method 2: Using `transformers` Library

```python
from transformers import AutoModel, AutoTokenizer

# Download pre-trained model
model_name = "emilyalsentzer/Bio_ClinicalBERT"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Extract embeddings
embeddings = model.embeddings.word_embeddings.weight.data
print(f"Vocabulary size: {embeddings.shape[0]}")
print(f"Embedding dimension: {embeddings.shape[1]}")

# Save for later use
torch.save(embeddings, "pretrained/clinical_bert_embeddings.pt")
```

### Method 3: Manual Download

```bash
# Install git-lfs first
git lfs install

# Clone repository
git clone https://huggingface.co/username/medical-code-embeddings

# Embeddings are in the repo
ls medical-code-embeddings/
```

### Method 4: Using Our Utility

```python
from ehrsequencing.models.pretrained_embeddings import load_embeddings

# Load from HuggingFace cache or local path
embeddings, metadata = load_embeddings("pretrained/embeddings.pt")
```

---

## Using Pre-trained Embeddings with BEHRT

### Step 1: Download or Create Embeddings

**Option A: Download from HuggingFace**
```python
from huggingface_hub import hf_hub_download

embedding_path = hf_hub_download(
    repo_id="medical-ai/icd10-embeddings",
    filename="embeddings.pt"
)
```

**Option B: Train Your Own (Med2Vec)**
```bash
# See Phase 2 implementation for Med2Vec training
python examples/code_embeddings/train_med2vec.py \
    --data_path data/ehr_sequences.pkl \
    --output_path pretrained/med2vec_embeddings.pt
```

### Step 2: Verify Embedding Compatibility

```python
from ehrsequencing.models.pretrained_embeddings import load_embeddings, print_embedding_statistics

# Load and inspect
embeddings, metadata = load_embeddings("pretrained/embeddings.pt")
print_embedding_statistics(embeddings, "Pre-trained Embeddings")

# Check shape
vocab_size = embeddings.shape[0]
embedding_dim = embeddings.shape[1]
print(f"Vocab size: {vocab_size}, Embedding dim: {embedding_dim}")
```

### Step 3: Fine-tune BEHRT with Pre-trained Embeddings

```bash
python examples/pretrain_finetune/train_behrt_finetune.py \
    --model_size large \
    --embedding_path pretrained/embeddings.pt \
    --use_lora \
    --lora_rank 16 \
    --num_patients 5000 \
    --epochs 50 \
    --realistic_data
```

### Step 4: Evaluate Performance

```bash
# Run benchmark to compare with training from scratch
python examples/pretrain_finetune/benchmark_pretrained_embeddings.py \
    --model_size large \
    --num_patients 10000 \
    --epochs 100
```

---

## General Best Practices

### 1. Vocabulary Alignment

**Problem:** Pre-trained embeddings may have different vocabulary than your data.

**Solutions:**

**A. Exact Match (Ideal)**
```python
# Ensure your data uses the same code system
# E.g., if embeddings are ICD-10, use ICD-10 codes in your data
```

**B. Mapping (Common)**
```python
# Map your codes to pre-trained vocabulary
code_mapping = {
    'local_code_1': 'pretrained_code_42',
    'local_code_2': 'pretrained_code_17',
    # ...
}

# Initialize embeddings for unmapped codes randomly
unmapped_codes = set(your_vocab) - set(pretrained_vocab)
print(f"Unmapped codes: {len(unmapped_codes)}/{len(your_vocab)}")
```

**C. Partial Initialization**
```python
from ehrsequencing.models.pretrained_embeddings import initialize_embedding_layer

# Initialize with pre-trained where available
# Random initialization for new codes
embedding_layer = nn.Embedding(your_vocab_size, embedding_dim)

# Copy pre-trained embeddings for matched codes
for your_idx, pretrained_idx in code_mapping.items():
    embedding_layer.weight.data[your_idx] = pretrained_embeddings[pretrained_idx]
```

### 2. Freezing Strategy

**Full Freeze (Recommended for Small Data)**
```python
# Freeze all embeddings
model = apply_lora_to_behrt(
    model,
    train_embeddings=False,  # Freeze embeddings
    train_head=True           # Train task head
)
```

**Partial Freeze (For Medium Data)**
```python
# Freeze pre-trained codes, train new codes
for idx in pretrained_code_indices:
    model.embeddings.code_embedding.weight.data[idx].requires_grad = False

for idx in new_code_indices:
    model.embeddings.code_embedding.weight.data[idx].requires_grad = True
```

**Gradual Unfreezing (For Large Data)**
```python
# Epoch 1-10: Freeze embeddings
# Epoch 11-20: Unfreeze with low LR
# Epoch 21+: Full training

if epoch < 10:
    freeze_embeddings(model)
elif epoch < 20:
    unfreeze_embeddings(model, lr=1e-5)
else:
    unfreeze_embeddings(model, lr=1e-4)
```

### 3. Learning Rate Selection

**Rule of Thumb:**
- **Frozen embeddings**: Use standard LR (1e-4)
- **Fine-tuning embeddings**: Use lower LR (1e-5 to 1e-6)
- **LoRA adapters**: Can use higher LR (1e-4 to 1e-3)

**Discriminative Learning Rates:**
```python
optimizer = torch.optim.AdamW([
    {'params': lora_params, 'lr': 1e-4},
    {'params': embedding_params, 'lr': 1e-5},
    {'params': head_params, 'lr': 1e-4}
])
```

### 4. Embedding Dimension Mismatch

**Problem:** Pre-trained embeddings have different dimension than your model.

**Solutions:**

**A. Adjust Model (Preferred)**
```python
# Match model dimension to pre-trained embeddings
config = BEHRTConfig.large(vocab_size=1000)
config.embedding_dim = pretrained_embeddings.shape[1]  # Use pre-trained dim
model = BEHRTForMLM(config)
```

**B. Project Embeddings (If Necessary)**
```python
# Project pre-trained embeddings to model dimension
projection = nn.Linear(pretrained_dim, model_dim, bias=False)
projected_embeddings = projection(pretrained_embeddings)
```

### 5. Monitoring Transfer Learning

**Track These Metrics:**
```python
# 1. Embedding drift (how much embeddings change)
initial_embeddings = model.embeddings.code_embedding.weight.data.clone()

# After training
embedding_drift = (model.embeddings.code_embedding.weight.data - initial_embeddings).norm()
print(f"Embedding drift: {embedding_drift:.4f}")

# 2. Performance on pre-training task
# Evaluate on original pre-training objective

# 3. Performance on downstream task
# Your specific task (classification, prediction, etc.)
```

---

## Foundation Model Fine-tuning Tips

### 1. The Fine-tuning Hierarchy

**Least to Most Aggressive:**

1. **Feature Extraction** (Freeze Everything)
   - Freeze: All layers
   - Train: Only task head
   - Use when: Very small data, similar domain

2. **LoRA Fine-tuning** (Our Approach)
   - Freeze: Base model
   - Train: LoRA adapters + task head
   - Use when: Small-medium data, limited compute

3. **Partial Fine-tuning**
   - Freeze: Lower layers
   - Train: Upper layers + task head
   - Use when: Medium data, some compute

4. **Full Fine-tuning**
   - Freeze: Nothing
   - Train: All parameters
   - Use when: Large data, lots of compute

### 2. Catastrophic Forgetting Prevention

**Problem:** Model forgets pre-trained knowledge during fine-tuning.

**Solutions:**

**A. Regularization**
```python
# L2 regularization toward pre-trained weights
initial_params = {name: param.clone() for name, param in model.named_parameters()}

def regularization_loss(model, initial_params, lambda_reg=0.01):
    reg_loss = 0
    for name, param in model.named_parameters():
        if name in initial_params:
            reg_loss += ((param - initial_params[name]) ** 2).sum()
    return lambda_reg * reg_loss
```

**B. Elastic Weight Consolidation (EWC)**
```python
# Protect important weights from changing
# See: https://arxiv.org/abs/1612.00796
```

**C. Progressive Unfreezing**
```python
# Unfreeze layers gradually
# Start with task head, then upper layers, then all
```

### 3. Domain Adaptation

**When Pre-trained Domain ≠ Target Domain:**

**Strategy 1: Two-stage Fine-tuning**
```bash
# Stage 1: Adapt to target domain (unsupervised)
python train_behrt_demo.py \
    --embedding_path pretrained/general_medical.pt \
    --data_path target_domain_data.pkl \
    --task mlm \
    --epochs 20

# Stage 2: Fine-tune on task (supervised)
python train_behrt_finetune.py \
    --embedding_path stage1_embeddings.pt \
    --task classification \
    --epochs 50
```

**Strategy 2: Multi-task Learning**
```python
# Train on both pre-training task and target task
loss = alpha * pretraining_loss + (1 - alpha) * task_loss
```

### 4. Hyperparameter Search

**Priority Order:**

1. **Learning Rate** (Most Important)
   - Try: [1e-5, 5e-5, 1e-4, 5e-4]
   - Use lower for fine-tuning, higher for LoRA

2. **LoRA Rank**
   - Try: [4, 8, 16, 32]
   - Higher rank = more capacity but slower

3. **Batch Size**
   - Larger is better (up to memory limit)
   - Use gradient accumulation if needed

4. **Dropout**
   - Try: [0.1, 0.2, 0.3]
   - Higher for smaller datasets

5. **Weight Decay**
   - Try: [0.0, 0.01, 0.1]
   - Prevents overfitting

### 5. Evaluation Best Practices

**Don't Just Look at Loss:**

```python
# 1. Task-specific metrics
# Classification: Accuracy, F1, AUROC, AUPRC
# Regression: MAE, RMSE, R²

# 2. Calibration
# Are predicted probabilities well-calibrated?

# 3. Fairness
# Performance across demographic groups

# 4. Robustness
# Performance on out-of-distribution data

# 5. Efficiency
# Inference time, memory usage
```

---

## Troubleshooting

### Issue 1: Poor Performance Despite Pre-trained Embeddings

**Possible Causes:**
- Domain mismatch (pre-trained on different data)
- Vocabulary mismatch (different code systems)
- Embeddings are outdated or low quality
- Task is too different from pre-training

**Solutions:**
1. Check embedding quality with visualization (t-SNE, UMAP)
2. Verify vocabulary overlap
3. Try training from scratch as baseline
4. Use domain-adaptive pre-training

### Issue 2: Embeddings Not Loading

**Common Errors:**
```python
# Shape mismatch
RuntimeError: size mismatch, expected [1000, 128], got [1000, 256]

# Solution: Adjust model config
config.embedding_dim = 256  # Match pre-trained
```

```python
# File format issue
# Solution: Ensure embeddings are saved correctly
torch.save({
    'embeddings': embeddings,
    'vocab_size': vocab_size,
    'embedding_dim': embedding_dim
}, 'embeddings.pt')
```

### Issue 3: Overfitting Despite Pre-trained Embeddings

**Solutions:**
1. Increase dropout
2. Add weight decay
3. Use early stopping
4. Reduce LoRA rank
5. Freeze more layers

### Issue 4: Slow Convergence

**Solutions:**
1. Increase learning rate (but not too high)
2. Use learning rate warmup
3. Try different optimizer (AdamW, Lion)
4. Check if embeddings are actually being used (not re-initialized)

---

## Example Workflows

### Workflow 1: Using HuggingFace Embeddings

```bash
# 1. Download embeddings
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('medical-ai/icd10-embeddings', 'embeddings.pt')
print(f'Downloaded to: {path}')
"

# 2. Fine-tune BEHRT
python examples/pretrain_finetune/train_behrt_finetune.py \
    --embedding_path ~/.cache/huggingface/hub/.../embeddings.pt \
    --model_size large \
    --num_patients 5000 \
    --realistic_data

# 3. Evaluate
python examples/pretrain_finetune/benchmark_pretrained_embeddings.py \
    --model_size large \
    --num_patients 10000
```

### Workflow 2: Creating Custom Embeddings

```bash
# 1. Train Med2Vec on your data
python examples/code_embeddings/train_med2vec.py \
    --data_path data/ehr_sequences.pkl \
    --embedding_dim 128 \
    --epochs 100 \
    --output_path pretrained/custom_embeddings.pt

# 2. Use with BEHRT
python examples/pretrain_finetune/train_behrt_finetune.py \
    --embedding_path pretrained/custom_embeddings.pt \
    --model_size medium \
    --num_patients 2000
```

### Workflow 3: Comparing Multiple Embeddings

```python
# Compare different pre-trained embeddings
embeddings_to_test = [
    'pretrained/med2vec.pt',
    'pretrained/clinical_bert.pt',
    'pretrained/custom.pt',
    None  # Train from scratch
]

for emb_path in embeddings_to_test:
    if emb_path:
        cmd = f"python train_behrt_finetune.py --embedding_path {emb_path}"
    else:
        cmd = "python train_behrt_demo.py"
    
    os.system(cmd)
```

---

## Resources

### Papers
- **Med2Vec**: Choi et al. (2016) - Multi-layer Representation Learning for Medical Concepts
- **BEHRT**: Li et al. (2020) - BEHRT: Transformer for Electronic Health Records
- **ClinicalBERT**: Alsentzer et al. (2019) - Publicly Available Clinical BERT Embeddings
- **LoRA**: Hu et al. (2021) - Low-Rank Adaptation of Large Language Models

### Code Repositories
- **HuggingFace Hub**: https://huggingface.co/models?filter=medical
- **Med2Vec**: https://github.com/mp2893/med2vec
- **BEHRT**: https://github.com/deepmedicine/BEHRT

### Datasets
- **MIMIC-III/IV**: https://mimic.mit.edu/
- **eICU**: https://eicu-crd.mit.edu/
- **All of Us**: https://www.researchallofus.org/

---

## Summary

**Key Takeaways:**

1. ✅ **Use pre-trained embeddings when possible** - Faster, better, less data needed
2. ✅ **Freeze embeddings for small data** - Prevents overfitting
3. ✅ **Use LoRA for efficient fine-tuning** - 90%+ parameter reduction
4. ✅ **Monitor embedding drift** - Ensure transfer learning is working
5. ✅ **Benchmark against training from scratch** - Verify pre-trained helps
6. ✅ **Check vocabulary alignment** - Critical for success
7. ✅ **Use realistic data for evaluation** - Random data doesn't generalize

**Next Steps:**

1. Download or train pre-trained embeddings
2. Run `train_behrt_finetune.py` with your embeddings
3. Run `benchmark_pretrained_embeddings.py` to compare
4. Analyze results and iterate

Happy fine-tuning! 🚀
