# Pre-trained Models Setup

**Date:** January 20, 2026  
**Focus:** Setting up and using pre-trained models (CEHR-BERT, Med-BERT) for EHR sequencing

---

## Overview

This guide covers downloading, configuring, and using pre-trained foundation models for medical code embeddings. These models provide contextualized embeddings that can significantly improve performance compared to training from scratch.

---

## Available Pre-trained Models

### CEHR-BERT (Recommended)

**Paper:** "CEHR-BERT: Incorporating temporal information from structured EHR data to improve prediction tasks"

**Details:**
- **Vocabulary**: ~20,000 medical codes (ICD-9/10, LOINC, RxNorm, CPT)
- **Architecture**: BERT-base (12 layers, 768 hidden dim)
- **Training Data**: MIMIC-III, eICU
- **Embedding Dim**: 768
- **Context Window**: 512 tokens

**Use Cases:**
- Visit-grouped sequence representation
- Disease progression modeling
- Patient similarity
- Risk prediction

### Med-BERT

**Paper:** "Med-BERT: Pre-trained contextualized embeddings on large-scale structured electronic health records"

**Details:**
- **Vocabulary**: ~15,000 medical codes
- **Architecture**: BERT-base
- **Training Data**: Multiple EHR datasets
- **Embedding Dim**: 768

### ClinicalBERT

**Paper:** "Publicly Available Clinical BERT Embeddings"

**Details:**
- **Focus**: Clinical notes (text)
- **Less suitable** for structured codes
- **Use**: Text-based features (if available)

---

## Installation

### Option 1: Hugging Face (Recommended)

```bash
# Install transformers
pip install transformers

# Download CEHR-BERT
python -c "
from transformers import AutoModel, AutoTokenizer

model_name = 'cehrbert/cehrbert-base'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained('checkpoints/cehrbert')
tokenizer.save_pretrained('checkpoints/cehrbert')
"
```

### Option 2: Direct Download

```bash
# Create directory
mkdir -p checkpoints/cehrbert

# Download from model repository
# (URL depends on specific model release)
wget https://example.com/cehrbert-base.tar.gz
tar -xzf cehrbert-base.tar.gz -C checkpoints/cehrbert/
```

### Option 3: Custom Pre-training (Advanced)

See [custom-pretraining.md](custom-pretraining.md) for training your own embeddings.

---

## Usage in EHR Sequencing

### Basic Usage: Load Pre-trained Embeddings

```python
from ehrsequencing.embeddings import CEHRBERTWrapper
import torch

# Load pre-trained model
cehrbert = CEHRBERTWrapper.from_pretrained('checkpoints/cehrbert')

# Get code embeddings
code_ids = torch.tensor([42, 123, 456])  # ICD codes mapped to IDs
code_embeddings = cehrbert.encode_codes(code_ids)
# Output: [3, 768] tensor

# Get visit embedding (aggregate codes)
visit_codes = torch.tensor([[42, 123, 456, 0, 0]])  # Padded
visit_embedding = cehrbert.encode_visit(visit_codes)
# Output: [1, 768] tensor
```

### Integration with LSTM Model

```python
from ehrsequencing.models.lstm import LSTMProgressionModel
from ehrsequencing.embeddings import CEHRBERTWrapper

# Load pre-trained embeddings
cehrbert = CEHRBERTWrapper.from_pretrained('checkpoints/cehrbert')
pretrained_embeddings = cehrbert.get_code_embeddings()  # [vocab_size, 768]

# Create model
model = LSTMProgressionModel(
    vocab_size=20000,
    code_embed_dim=768,  # Match CEHR-BERT
    visit_embed_dim=512,
    hidden_dim=512,
    num_stages=5
)

# Replace embeddings with pre-trained
model.encoder.code_embeddings = torch.nn.Embedding.from_pretrained(
    pretrained_embeddings,
    freeze=False  # Allow fine-tuning
)

# Optional: Freeze embeddings (no fine-tuning)
# model.encoder.code_embeddings.weight.requires_grad = False
```

### Fine-tuning vs Frozen Embeddings

**Fine-tuning (Recommended):**
```python
# Allow gradients to flow through embeddings
model.encoder.code_embeddings.weight.requires_grad = True

# Use lower learning rate for embeddings
optimizer = torch.optim.Adam([
    {'params': model.encoder.code_embeddings.parameters(), 'lr': 1e-5},
    {'params': [p for n, p in model.named_parameters() 
                if 'code_embeddings' not in n], 'lr': 1e-3}
])
```

**Frozen Embeddings:**
```python
# Freeze embeddings (faster training, less memory)
model.encoder.code_embeddings.weight.requires_grad = False

# Only train other parameters
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
```

---

## CEHR-BERT Wrapper Implementation

```python
# src/ehrsequencing/embeddings/cehrbert_wrapper.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict

class CEHRBERTWrapper(nn.Module):
    """
    Wrapper for CEHR-BERT pre-trained model.
    
    Provides convenient methods for encoding codes and visits.
    """
    
    def __init__(
        self,
        model_path: str,
        freeze_embeddings: bool = False,
        pooling: str = 'mean'  # 'mean', 'cls', 'max'
    ):
        super().__init__()
        
        # Load pre-trained model
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Freeze if requested
        if freeze_embeddings:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.pooling = pooling
        self.embed_dim = self.model.config.hidden_size
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load pre-trained CEHR-BERT."""
        return cls(model_path, **kwargs)
    
    def get_code_embeddings(self) -> torch.Tensor:
        """
        Extract code embeddings from CEHR-BERT.
        
        Returns:
            embeddings: [vocab_size, embed_dim]
        """
        # Get embedding layer
        embeddings = self.model.embeddings.word_embeddings.weight.data
        return embeddings.clone()
    
    def encode_codes(
        self,
        code_ids: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Encode individual codes.
        
        Args:
            code_ids: [batch, num_codes] or [num_codes]
            return_attention: Return attention weights
        
        Returns:
            embeddings: [batch, num_codes, embed_dim] or [num_codes, embed_dim]
        """
        if code_ids.dim() == 1:
            code_ids = code_ids.unsqueeze(0)
        
        # Get embeddings directly (no context)
        embeddings = self.model.embeddings.word_embeddings(code_ids)
        
        if return_attention:
            # Run through transformer for contextualized embeddings
            outputs = self.model(input_ids=code_ids, output_attentions=True)
            return outputs.last_hidden_state, outputs.attentions
        
        return embeddings.squeeze(0) if code_ids.size(0) == 1 else embeddings
    
    def encode_visit(
        self,
        visit_codes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a visit (sequence of codes).
        
        Args:
            visit_codes: [batch, max_codes_per_visit]
            attention_mask: [batch, max_codes_per_visit]
        
        Returns:
            visit_embedding: [batch, embed_dim]
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (visit_codes != 0).long()
        
        # Run through BERT
        outputs = self.model(
            input_ids=visit_codes,
            attention_mask=attention_mask
        )
        
        # Pool to get visit embedding
        if self.pooling == 'cls':
            # Use [CLS] token
            visit_embedding = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == 'mean':
            # Mean pooling (excluding padding)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            visit_embedding = sum_embeddings / sum_mask
        elif self.pooling == 'max':
            # Max pooling
            visit_embedding = torch.max(outputs.last_hidden_state, dim=1)[0]
        
        return visit_embedding
    
    def encode_patient(
        self,
        patient_visits: torch.Tensor,
        visit_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode patient trajectory (sequence of visits).
        
        Args:
            patient_visits: [batch, num_visits, max_codes_per_visit]
            visit_mask: [batch, num_visits, max_codes_per_visit]
        
        Returns:
            visit_embeddings: [batch, num_visits, embed_dim]
        """
        batch_size, num_visits, max_codes = patient_visits.shape
        
        # Encode each visit
        visit_embeddings = []
        for i in range(num_visits):
            visit_codes = patient_visits[:, i, :]
            mask = visit_mask[:, i, :] if visit_mask is not None else None
            visit_emb = self.encode_visit(visit_codes, mask)
            visit_embeddings.append(visit_emb)
        
        return torch.stack(visit_embeddings, dim=1)
```

---

## Vocabulary Mapping

### Map Your Codes to CEHR-BERT Vocabulary

```python
# src/ehrsequencing/data/vocab.py

import json
from typing import Dict, List

class VocabularyMapper:
    """Map project codes to CEHR-BERT vocabulary."""
    
    def __init__(self, cehrbert_vocab_path: str):
        # Load CEHR-BERT vocabulary
        with open(cehrbert_vocab_path, 'r') as f:
            self.cehrbert_vocab = json.load(f)
        
        # Create reverse mapping
        self.code_to_id = {code: idx for idx, code in enumerate(self.cehrbert_vocab)}
        self.unk_id = self.code_to_id.get('[UNK]', 0)
    
    def map_code(self, code: str) -> int:
        """Map a single code to CEHR-BERT ID."""
        return self.code_to_id.get(code, self.unk_id)
    
    def map_codes(self, codes: List[str]) -> List[int]:
        """Map multiple codes."""
        return [self.map_code(code) for code in codes]
    
    def get_coverage(self, dataset_codes: List[str]) -> float:
        """Calculate vocabulary coverage."""
        mapped = sum(1 for code in dataset_codes if code in self.code_to_id)
        return mapped / len(dataset_codes) if dataset_codes else 0.0
```

---

## Memory Considerations

### CEHR-BERT Memory Usage

**Model Size:**
- CEHR-BERT base: ~440MB (110M parameters)
- Embeddings only: ~60MB (20K vocab × 768 dim × 4 bytes)

**During Training:**
- Full model: ~2GB (with optimizer states)
- Frozen embeddings: ~500MB

### Optimization Strategies

```python
# 1. Use frozen embeddings (save memory)
model.encoder.code_embeddings.weight.requires_grad = False

# 2. Use smaller batch size
config.batch_size = 16  # Instead of 32

# 3. Use gradient checkpointing
config.use_checkpoint = True

# 4. Extract embeddings once, save to disk
embeddings = cehrbert.get_code_embeddings()
torch.save(embeddings, 'checkpoints/cehrbert_embeddings.pt')

# Later: Load directly
embeddings = torch.load('checkpoints/cehrbert_embeddings.pt')
model.encoder.code_embeddings = nn.Embedding.from_pretrained(embeddings)
```

---

## Evaluation

### Compare Pre-trained vs Random Initialization

```python
# Train two models
model_pretrained = LSTMProgressionModel(...)
model_pretrained.load_pretrained_embeddings('checkpoints/cehrbert')

model_random = LSTMProgressionModel(...)
# Uses random initialization

# Compare performance
results_pretrained = evaluate(model_pretrained, test_data)
results_random = evaluate(model_random, test_data)

print(f"Pre-trained AUC: {results_pretrained['auc']:.3f}")
print(f"Random AUC: {results_random['auc']:.3f}")
```

---

## Troubleshooting

### Model Not Found

```bash
# Check model path
ls checkpoints/cehrbert/

# Re-download
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('cehrbert/cehrbert-base')
model.save_pretrained('checkpoints/cehrbert')
"
```

### Vocabulary Mismatch

```python
# Check vocabulary size
print(f"Model vocab: {model.encoder.code_embeddings.num_embeddings}")
print(f"CEHR-BERT vocab: {cehrbert.model.config.vocab_size}")

# Resize if needed
model.encoder.code_embeddings = nn.Embedding(
    num_embeddings=cehrbert.model.config.vocab_size,
    embedding_dim=768
)
```

### Out of Memory

```bash
# Use frozen embeddings
model.encoder.code_embeddings.weight.requires_grad = False

# Or extract embeddings only
embeddings = cehrbert.get_code_embeddings()
torch.save(embeddings, 'cehrbert_embeddings.pt')
# Then load without full model
```

---

## Related Documentation

- [INSTALL.md](../../INSTALL.md) - Installation guide
- [Resource-Aware Models](../../docs/implementation/resource-aware-models.md) - Model configs
- [Visit-Grouped Sequences](../../docs/implementation/visit-grouped-sequences.md) - Implementation plan

---

**Last Updated:** January 20, 2026
