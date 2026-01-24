# Pretrained Embeddings for Medical Codes

## Overview

Pretrained embeddings capture semantic relationships between medical codes learned from large corpora. Using them instead of random initialization can significantly improve model performance, especially with limited training data.

**Expected Impact**: +0.05-0.10 C-index improvement

## Available Pretrained Embeddings

### 1. Med2Vec (Recommended)

**What it is**: Skip-gram embeddings trained on visit co-occurrence patterns

**Advantages**:
- Learns from EHR visit structure (codes that co-occur have similar embeddings)
- Captures clinical relationships (e.g., diabetes codes cluster together)
- Relatively easy to train on your own data

**How to get**:
- Train on your own EHR data using Med2Vec implementation
- Use publicly available Med2Vec embeddings (if available)

**Training Med2Vec**:
```python
from med2vec import Med2Vec

# Prepare visit data: List[List[str]] (list of visits, each visit is list of codes)
visits = [
    ['250.00', '401.9', '272.4'],  # Visit 1
    ['250.00', '585.9'],            # Visit 2
    ...
]

# Train Med2Vec
model = Med2Vec(
    num_codes=len(vocab),
    embedding_dim=128,
    num_visits=len(visits)
)

model.train(visits, epochs=100)
model.save('med2vec_embeddings.pkl')
```

### 2. CUI2Vec

**What it is**: UMLS Concept embeddings trained on clinical notes

**Advantages**:
- Trained on large clinical text corpora
- Captures semantic relationships from natural language
- Publicly available

**Requirements**:
- Map ICD/CPT codes to UMLS CUIs
- Download CUI2Vec embeddings

**Download**:
```bash
# CUI2Vec embeddings
wget https://figshare.com/ndownloader/files/10959626 -O cui2vec_pretrained.txt
```

### 3. Clinical BERT Embeddings

**What it is**: Contextualized embeddings from BERT models trained on clinical text

**Advantages**:
- State-of-the-art performance
- Captures context-dependent meanings
- Multiple variants (BioBERT, ClinicalBERT, PubMedBERT)

**Disadvantages**:
- More complex to use (requires tokenization)
- Larger model size
- May be overkill for simple code embeddings

## Using Pretrained Embeddings

### Option 1: Load from File

```python
from ehrsequencing.embeddings import PretrainedEmbedding

# Load embeddings
embedding = PretrainedEmbedding.from_file(
    embedding_path='med2vec_embeddings.pkl',
    vocab=builder.vocab,
    embedding_dim=128,
    freeze=True  # Don't update during training
)

# Use in model
model = DiscreteTimeSurvivalLSTM(
    vocab_size=builder.vocabulary_size,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.3,
    pretrained_embedding=embedding  # Pass pretrained embeddings
)
```

### Option 2: Med2Vec Specific

```python
from ehrsequencing.embeddings import Med2VecEmbedding

embedding = Med2VecEmbedding.from_med2vec_checkpoint(
    checkpoint_path='med2vec_model.pkl',
    vocab=builder.vocab,
    freeze=True
)
```

### Option 3: Train Your Own Med2Vec

```python
# 1. Extract visits from your data
visits_by_patient = defaultdict(list)
for visit in visits:
    visits_by_patient[visit.patient_id].append(visit)

# 2. Convert to code lists
all_visits = []
for patient_visits in visits_by_patient.values():
    for visit in patient_visits:
        codes = visit.get_all_codes()
        all_visits.append(codes)

# 3. Train Med2Vec
from med2vec import Med2Vec

model = Med2Vec(
    num_codes=len(builder.vocab),
    embedding_dim=128,
    num_visits=len(all_visits)
)

model.train(all_visits, epochs=100)

# 4. Extract embeddings
code_embeddings = model.get_code_embeddings()

# 5. Create embedding layer
embedding = PretrainedEmbedding(
    vocab=builder.vocab,
    embedding_dim=128,
    pretrained_weights=code_embeddings,
    freeze=False  # Allow fine-tuning
)
```

## Freezing vs. Fine-tuning

### Frozen Embeddings (Recommended for Small Data)

```python
embedding = PretrainedEmbedding.from_file(
    embedding_path='embeddings.pkl',
    vocab=vocab,
    embedding_dim=128,
    freeze=True  # Don't update during training
)
```

**Advantages**:
- Prevents overfitting with small data
- Faster training
- Preserves pretrained knowledge

**Disadvantages**:
- Can't adapt to task-specific patterns

### Fine-tuned Embeddings (Recommended for Large Data)

```python
embedding = PretrainedEmbedding.from_file(
    embedding_path='embeddings.pkl',
    vocab=vocab,
    embedding_dim=128,
    freeze=False  # Update during training
)

# Or unfreeze after initial training
embedding.unfreeze()
```

**Advantages**:
- Adapts to task-specific patterns
- Better performance with sufficient data

**Disadvantages**:
- Risk of overfitting with small data
- Slower training

### Hybrid Approach (Best of Both)

```python
# Phase 1: Train with frozen embeddings
embedding.freeze = True
train_model(model, epochs=10)

# Phase 2: Fine-tune embeddings
embedding.unfreeze()
train_model(model, epochs=5, lr=0.0001)  # Lower learning rate
```

## Handling Unknown Codes

Pretrained embeddings may not cover all codes in your vocabulary:

```python
# Check coverage
embedding = PretrainedEmbedding.from_file(...)
# Logs: "Pretrained embedding coverage: 75.3% (583/776)"
```

**Strategies**:

1. **Random initialization for unknown codes** (default)
   - Unknown codes get random embeddings
   - Can be learned during training

2. **Average of known embeddings**
   ```python
   # Replace random init with average
   unknown_mask = (embedding.weight == 0).all(dim=1)
   known_embeddings = embedding.weight[~unknown_mask]
   avg_embedding = known_embeddings.mean(dim=0)
   embedding.weight[unknown_mask] = avg_embedding
   ```

3. **Hierarchical fallback**
   - Map ICD-10 codes to ICD-9 if ICD-10 not found
   - Use code prefixes (e.g., '250' for all diabetes codes)

## Expected Performance

| Setup | C-index | Notes |
|-------|---------|-------|
| Random embeddings, 106 patients | 0.45-0.52 | Baseline (current) |
| Random embeddings, 1000 patients | 0.60-0.70 | More data helps |
| Pretrained embeddings, 106 patients | 0.50-0.58 | Small improvement |
| Pretrained embeddings, 1000 patients | 0.65-0.75 | Best combination |
| Pretrained + fine-tuned, 1000 patients | 0.70-0.80 | Optimal |

## Implementation Checklist

- [ ] Generate/download more training data (1000+ patients)
- [ ] Train Med2Vec on your data OR download pretrained embeddings
- [ ] Load pretrained embeddings using `PretrainedEmbedding.from_file()`
- [ ] Update survival LSTM to accept pretrained embeddings
- [ ] Train with frozen embeddings first
- [ ] Optionally fine-tune embeddings with lower learning rate
- [ ] Evaluate C-index improvement
- [ ] Document embedding source and parameters

## Troubleshooting

### Low Coverage (<50%)

**Problem**: Most codes not in pretrained embeddings

**Solutions**:
- Train Med2Vec on your own data
- Use code mapping (ICD-10 → ICD-9, detailed → general)
- Use hierarchical embeddings (code prefixes)

### No Performance Improvement

**Problem**: Pretrained embeddings don't help

**Possible causes**:
- Embeddings trained on different code system
- Task-specific patterns not captured by embeddings
- Data too small to benefit from pretraining

**Solutions**:
- Try fine-tuning instead of freezing
- Train task-specific embeddings
- Get more training data

### Overfitting with Fine-tuning

**Problem**: Training loss decreases but validation C-index doesn't improve

**Solutions**:
- Use frozen embeddings
- Add regularization (weight decay, dropout)
- Reduce learning rate for embedding layer
- Use hybrid approach (freeze → fine-tune)

## References

- **Med2Vec**: Choi et al. (2016). "Multi-layer Representation Learning for Medical Concepts"
- **CUI2Vec**: Beam et al. (2018). "Clinical Concept Embeddings Learned from Massive Sources"
- **Clinical BERT**: Alsentzer et al. (2019). "Publicly Available Clinical BERT Embeddings"

## Next Steps

1. **Get more data** (see `data_generation_guide.md`)
2. **Train or download pretrained embeddings**
3. **Update model to use pretrained embeddings**
4. **Retrain and evaluate**
5. **Compare with random embedding baseline**
