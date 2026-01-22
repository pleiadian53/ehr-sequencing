# Resource-Aware Model Configurations

**Date:** January 20, 2026  
**Focus:** Small/Medium/Large model presets for M1 MacBook (16GB) and RunPod deployment

---

## Overview

This document defines resource-aware model configurations for the EHR Sequencing project, optimized for different hardware constraints:

- **SMALL**: M1 MacBook Pro 16GB (local development, fast iteration)
- **MEDIUM**: RunPod with 24GB GPU (A10, RTX 4090)
- **LARGE**: Cloud instances with 40GB+ GPU (A40, A100)

**Key Principle:** All models are **logically equivalent** but scaled for available resources. This enables:
- Fast iteration on M1 MacBook
- Realistic training on RunPod
- Production-scale deployment on cloud

---

## Part 1: LSTM Baseline for Visit-Grouped Sequences

### Should We Use LSTM as a Baseline?

**Answer: YES - LSTM is an excellent baseline for visit-grouped sequences.**

#### Rationale

**1. Natural Fit for Visit Sequences**
```python
# Visit sequence is naturally sequential
patient_trajectory = [visit1, visit2, visit3, ..., visitN]

# LSTM processes sequences step-by-step
hidden_state = lstm(visit_embeddings)
```

**2. Computational Efficiency**
- Much faster than Transformers for long sequences
- Lower memory footprint
- Suitable for M1 MacBook development

**3. Strong Baseline Performance**
- LSTMs have proven effective for EHR sequences
- Captures temporal dependencies
- Easier to interpret than Transformers

**4. Comparison Point**
- Establishes baseline performance
- Compare against Transformer-based models
- Validate that complexity is justified

#### LSTM vs Transformer for Visit Sequences

| Aspect | LSTM | Transformer |
|--------|------|-------------|
| **Sequence Length** | Efficient for long sequences (50+ visits) | Quadratic complexity O(n²) |
| **Memory** | O(n) - Linear | O(n²) - Quadratic |
| **Training Speed** | Fast | Slower |
| **Long-range Deps** | Limited (vanishing gradients) | Excellent (attention) |
| **Interpretability** | Moderate (hidden states) | High (attention weights) |
| **M1 MacBook** | ✅ Runs well | ⚠️ Slower, more memory |
| **Best For** | Baseline, fast iteration | Production, best performance |

**Recommendation:** Implement both LSTM and Transformer, use LSTM as baseline.

---

## Part 2: LSTM Baseline Architecture

### Two-Level LSTM for Visit-Grouped Sequences

```python
import torch
import torch.nn as nn

class LSTMVisitEncoder(nn.Module):
    """
    Two-level LSTM for visit-grouped sequences.
    
    Level 1: Encode codes within each visit
    Level 2: Model sequence of visits
    """
    
    def __init__(
        self,
        vocab_size: int,
        code_embed_dim: int = 128,
        visit_embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        
        # Code embeddings (can use pre-trained)
        self.code_embeddings = nn.Embedding(vocab_size, code_embed_dim, padding_idx=0)
        
        # Level 1: Within-visit LSTM
        self.visit_lstm = nn.LSTM(
            input_size=code_embed_dim,
            hidden_size=code_embed_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )
        
        # Project to visit embedding
        self.visit_projection = nn.Linear(code_embed_dim, visit_embed_dim)
        
        # Level 2: Visit sequence LSTM
        self.sequence_lstm = nn.LSTM(
            input_size=visit_embed_dim + 2,  # +2 for time features
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim * (2 if bidirectional else 1)
    
    def encode_visit(self, visit_codes, visit_mask):
        """
        Encode a single visit.
        
        Args:
            visit_codes: [batch, max_codes_per_visit]
            visit_mask: [batch, max_codes_per_visit]
        
        Returns:
            visit_embedding: [batch, visit_embed_dim]
        """
        # Embed codes
        code_embeds = self.code_embeddings(visit_codes)  # [batch, codes, embed_dim]
        
        # LSTM over codes
        lstm_out, (hidden, _) = self.visit_lstm(code_embeds)
        
        # Use last hidden state (or mean pooling)
        visit_repr = hidden[-1]  # [batch, code_embed_dim]
        
        # Project to visit space
        return self.visit_projection(visit_repr)
    
    def forward(self, patient_visits, time_features, visit_mask):
        """
        Encode patient visit sequence.
        
        Args:
            patient_visits: [batch, num_visits, max_codes_per_visit]
            time_features: [batch, num_visits, 2]
            visit_mask: [batch, num_visits, max_codes_per_visit]
        
        Returns:
            sequence_output: [batch, num_visits, hidden_dim]
            final_hidden: [num_layers, batch, hidden_dim]
        """
        batch_size, num_visits, max_codes = patient_visits.shape
        
        # Encode each visit
        visit_embeds = []
        for i in range(num_visits):
            visit_embed = self.encode_visit(
                patient_visits[:, i, :],
                visit_mask[:, i, :]
            )
            visit_embeds.append(visit_embed)
        
        visit_embeds = torch.stack(visit_embeds, dim=1)  # [batch, visits, visit_dim]
        
        # Concatenate time features
        visit_embeds_with_time = torch.cat([visit_embeds, time_features], dim=-1)
        
        # LSTM over visit sequence
        sequence_output, (final_hidden, final_cell) = self.sequence_lstm(
            visit_embeds_with_time
        )
        
        return sequence_output, final_hidden
```

### Disease Progression Model with LSTM

```python
class LSTMProgressionModel(nn.Module):
    """
    LSTM-based disease progression model.
    
    Predicts disease stage and time to progression.
    """
    
    def __init__(
        self,
        vocab_size: int,
        code_embed_dim: int = 128,
        visit_embed_dim: int = 256,
        hidden_dim: int = 512,
        num_stages: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Visit encoder
        self.encoder = LSTMVisitEncoder(
            vocab_size=vocab_size,
            code_embed_dim=code_embed_dim,
            visit_embed_dim=visit_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        )
        
        # Prediction heads
        self.stage_classifier = nn.Linear(hidden_dim, num_stages)
        self.time_to_progression = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, patient_visits, time_features, visit_mask):
        """
        Predict disease progression.
        
        Returns:
            stage_logits: [batch, num_visits, num_stages]
            time_pred: [batch, num_visits, 1]
        """
        # Encode sequence
        sequence_output, _ = self.encoder(patient_visits, time_features, visit_mask)
        sequence_output = self.dropout(sequence_output)
        
        # Predict at each visit
        stage_logits = self.stage_classifier(sequence_output)
        time_pred = torch.relu(self.time_to_progression(sequence_output))
        
        return stage_logits, time_pred
```

---

## Part 3: Resource-Aware Model Configurations

### Configuration System

```python
# src/ehrsequencing/models/configs/model_configs.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class EHRModelConfig:
    """Configuration for EHR sequence models."""
    
    # Model architecture
    code_embed_dim: int
    visit_embed_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float = 0.1
    
    # Model type
    model_type: str = "lstm"  # "lstm" or "transformer"
    
    # Transformer-specific
    num_heads: Optional[int] = None
    use_flash_attention: bool = False
    
    # Vocabulary
    vocab_size: int = 10000
    max_visits: int = 50
    max_codes_per_visit: int = 100
    
    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    use_checkpoint: bool = False
    
    # Task-specific
    num_stages: int = 5  # For disease progression
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def total_params_millions(self) -> float:
        """Estimate total parameters in millions."""
        if self.model_type == "lstm":
            # LSTM parameters
            # Input-to-hidden: 4 * (input_size * hidden_dim + hidden_dim^2)
            # Per layer
            code_lstm_params = 4 * (self.code_embed_dim * self.code_embed_dim + 
                                   self.code_embed_dim ** 2)
            
            visit_lstm_params = 4 * (self.visit_embed_dim * self.hidden_dim + 
                                     self.hidden_dim ** 2) * self.num_layers
            
            # Embeddings
            embed_params = self.vocab_size * self.code_embed_dim
            
            # Projection layers
            proj_params = self.code_embed_dim * self.visit_embed_dim
            
            # Prediction heads
            head_params = self.hidden_dim * (self.num_stages + 1)
            
            total = (code_lstm_params + visit_lstm_params + embed_params + 
                    proj_params + head_params) / 1e6
            
        elif self.model_type == "transformer":
            # Transformer parameters (rough estimate)
            attn_params = 4 * self.visit_embed_dim * self.visit_embed_dim * self.num_layers
            ffn_params = 2 * self.visit_embed_dim * self.hidden_dim * self.num_layers
            embed_params = self.vocab_size * self.code_embed_dim
            
            total = (attn_params + ffn_params + embed_params) / 1e6
        
        return round(total, 2)
    
    def memory_estimate_gb(self, dtype_bytes: int = 4) -> float:
        """Estimate memory usage in GB."""
        # Model parameters
        model_memory = self.total_params_millions * 1e6 * dtype_bytes / 1e9
        
        # Optimizer states (Adam: 2x)
        optimizer_memory = model_memory * 2
        
        # Activations (LSTM is more memory efficient than Transformer)
        if self.model_type == "lstm":
            activation_memory = (
                self.batch_size * self.max_visits * self.hidden_dim * 
                self.num_layers * dtype_bytes / 1e9
            )
        else:  # transformer
            activation_memory = (
                self.batch_size * self.max_visits * self.max_visits * 
                self.num_heads * dtype_bytes / 1e9
            )
        
        # Gradients
        gradient_memory = model_memory
        
        total = model_memory + optimizer_memory + activation_memory + gradient_memory
        total *= 1.2  # 20% overhead
        
        return round(total, 2)
```

### Preset Configurations

```python
# src/ehrsequencing/models/configs/presets.py

from .model_configs import EHRModelConfig

# ============================================================================
# SMALL: M1 MacBook Pro 16GB
# ============================================================================

SMALL_LSTM_CONFIG = EHRModelConfig(
    # Architecture
    code_embed_dim=128,
    visit_embed_dim=256,
    hidden_dim=256,
    num_layers=2,
    dropout=0.1,
    
    # Model type
    model_type="lstm",
    
    # Vocabulary
    vocab_size=10000,
    max_visits=50,
    max_codes_per_visit=50,
    
    # Training (optimized for M1 16GB)
    batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch = 32
    mixed_precision=True,
    use_checkpoint=True,  # Gradient checkpointing
    
    # Task
    num_stages=5,
)

SMALL_TRANSFORMER_CONFIG = EHRModelConfig(
    # Architecture
    code_embed_dim=128,
    visit_embed_dim=256,
    hidden_dim=512,
    num_layers=4,
    num_heads=4,
    dropout=0.1,
    
    # Model type
    model_type="transformer",
    use_flash_attention=False,  # Not available on M1
    
    # Vocabulary
    vocab_size=10000,
    max_visits=30,  # Shorter for memory
    max_codes_per_visit=50,
    
    # Training
    batch_size=2,
    gradient_accumulation_steps=16,  # Effective batch = 32
    mixed_precision=True,
    use_checkpoint=True,
    
    # Task
    num_stages=5,
)

# ============================================================================
# MEDIUM: RunPod 24GB GPU (A10, RTX 4090)
# ============================================================================

MEDIUM_LSTM_CONFIG = EHRModelConfig(
    # Architecture
    code_embed_dim=256,
    visit_embed_dim=512,
    hidden_dim=512,
    num_layers=3,
    dropout=0.1,
    
    # Model type
    model_type="lstm",
    
    # Vocabulary
    vocab_size=20000,
    max_visits=100,
    max_codes_per_visit=100,
    
    # Training (optimized for 24GB GPU)
    batch_size=32,
    gradient_accumulation_steps=1,
    mixed_precision=True,
    use_checkpoint=False,
    
    # Task
    num_stages=5,
)

MEDIUM_TRANSFORMER_CONFIG = EHRModelConfig(
    # Architecture
    code_embed_dim=256,
    visit_embed_dim=512,
    hidden_dim=1024,
    num_layers=6,
    num_heads=8,
    dropout=0.1,
    
    # Model type
    model_type="transformer",
    use_flash_attention=True,
    
    # Vocabulary
    vocab_size=20000,
    max_visits=50,
    max_codes_per_visit=100,
    
    # Training
    batch_size=16,
    gradient_accumulation_steps=2,  # Effective batch = 32
    mixed_precision=True,
    use_checkpoint=False,
    
    # Task
    num_stages=5,
)

# ============================================================================
# LARGE: Cloud 40GB+ GPU (A40, A100)
# ============================================================================

LARGE_LSTM_CONFIG = EHRModelConfig(
    # Architecture
    code_embed_dim=512,
    visit_embed_dim=768,
    hidden_dim=1024,
    num_layers=4,
    dropout=0.1,
    
    # Model type
    model_type="lstm",
    
    # Vocabulary
    vocab_size=50000,
    max_visits=200,
    max_codes_per_visit=150,
    
    # Training (optimized for 40GB+ GPU)
    batch_size=64,
    gradient_accumulation_steps=1,
    mixed_precision=True,
    use_checkpoint=False,
    
    # Task
    num_stages=5,
)

LARGE_TRANSFORMER_CONFIG = EHRModelConfig(
    # Architecture
    code_embed_dim=512,
    visit_embed_dim=768,
    hidden_dim=2048,
    num_layers=12,
    num_heads=12,
    dropout=0.1,
    
    # Model type
    model_type="transformer",
    use_flash_attention=True,
    
    # Vocabulary
    vocab_size=50000,
    max_visits=100,
    max_codes_per_visit=150,
    
    # Training
    batch_size=32,
    gradient_accumulation_steps=2,  # Effective batch = 64
    mixed_precision=True,
    use_checkpoint=False,
    
    # Task
    num_stages=5,
)


def get_model_config(size: str = "small", model_type: str = "lstm") -> EHRModelConfig:
    """
    Get a preset model configuration.
    
    Args:
        size: "small", "medium", or "large"
        model_type: "lstm" or "transformer"
    
    Returns:
        EHRModelConfig instance
    
    Examples:
        >>> # For M1 MacBook development
        >>> config = get_model_config("small", "lstm")
        >>> print(f"Memory: {config.memory_estimate_gb(dtype_bytes=2)}GB")
        
        >>> # For RunPod training
        >>> config = get_model_config("medium", "transformer")
    """
    configs = {
        ("small", "lstm"): SMALL_LSTM_CONFIG,
        ("small", "transformer"): SMALL_TRANSFORMER_CONFIG,
        ("medium", "lstm"): MEDIUM_LSTM_CONFIG,
        ("medium", "transformer"): MEDIUM_TRANSFORMER_CONFIG,
        ("large", "lstm"): LARGE_LSTM_CONFIG,
        ("large", "transformer"): LARGE_TRANSFORMER_CONFIG,
    }
    
    key = (size, model_type)
    if key not in configs:
        raise ValueError(
            f"Unknown config: size='{size}', model_type='{model_type}'. "
            f"Valid sizes: small, medium, large. Valid types: lstm, transformer."
        )
    
    return configs[key]
```

---

## Part 4: Configuration Comparison

### Memory and Performance Estimates

| Config | Model Type | Params (M) | Memory (GB) | Batch | Max Visits | Hardware |
|--------|-----------|-----------|-------------|-------|-----------|----------|
| **Small LSTM** | LSTM | ~5M | ~3GB | 4 (32) | 50 | M1 16GB |
| **Small Transformer** | Transformer | ~8M | ~5GB | 2 (32) | 30 | M1 16GB |
| **Medium LSTM** | LSTM | ~15M | ~8GB | 32 | 100 | RunPod 24GB |
| **Medium Transformer** | Transformer | ~25M | ~12GB | 16 (32) | 50 | RunPod 24GB |
| **Large LSTM** | LSTM | ~40M | ~18GB | 64 | 200 | A40/A100 40GB+ |
| **Large Transformer** | Transformer | ~80M | ~32GB | 32 (64) | 100 | A40/A100 40GB+ |

**Notes:**
- Memory estimates assume fp16 mixed precision
- Batch shows actual batch size (effective batch size in parentheses)
- Small configs use gradient checkpointing
- LSTM is 2-3x more memory efficient than Transformer

---

## Part 5: Usage Examples

### Training Script with Config System

```python
# scripts/train_disease_progression.py

import argparse
import torch
from ehrsequencing.models.configs import get_model_config
from ehrsequencing.models import LSTMProgressionModel, TransformerProgressionModel
from ehrsequencing.data import load_sequences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--model-type", default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data-path", required=True)
    args = parser.parse_args()
    
    # Get configuration
    config = get_model_config(args.size, args.model_type)
    
    print(f"\n{'='*80}")
    print(f"Training {args.model_type.upper()} model ({args.size} config)")
    print(f"{'='*80}")
    print(f"Parameters: ~{config.total_params_millions}M")
    print(f"Memory estimate: ~{config.memory_estimate_gb(dtype_bytes=2)}GB (fp16)")
    print(f"Effective batch size: {config.effective_batch_size}")
    print(f"Max visits: {config.max_visits}")
    print(f"{'='*80}\n")
    
    # Load data
    sequences = load_sequences(args.data_path)
    
    # Create model
    if args.model_type == "lstm":
        model = LSTMProgressionModel(
            vocab_size=config.vocab_size,
            code_embed_dim=config.code_embed_dim,
            visit_embed_dim=config.visit_embed_dim,
            hidden_dim=config.hidden_dim,
            num_stages=config.num_stages,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
    else:
        model = TransformerProgressionModel(
            vocab_size=config.vocab_size,
            embed_dim=config.visit_embed_dim,
            hidden_dim=config.hidden_dim,
            num_stages=config.num_stages,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # ... training code ...

if __name__ == "__main__":
    main()
```

### Local Development Workflow

```bash
# On M1 MacBook: Fast iteration with small LSTM
python scripts/train_disease_progression.py \
    --size small \
    --model-type lstm \
    --epochs 10 \
    --data-path data/ckd_sequences.pt

# Verify logic works, then scale up on RunPod
```

### RunPod Training Workflow

```bash
# On RunPod A10 24GB: Medium transformer
python scripts/train_disease_progression.py \
    --size medium \
    --model-type transformer \
    --epochs 100 \
    --data-path /workspace/ehr-sequencing/data/ckd_sequences.pt
```

---

## Part 6: Pre-trained Model Integration

### Using Pre-trained Embeddings with Resource Configs

```python
from ehrsequencing.embeddings import CEHRBERTWrapper
from ehrsequencing.models.configs import get_model_config

# Load pre-trained embeddings
cehrbert = CEHRBERTWrapper.from_pretrained('cehrbert-base')
pretrained_embeddings = cehrbert.get_code_embeddings()

# Get config
config = get_model_config("small", "lstm")

# Create model with pre-trained embeddings
model = LSTMProgressionModel(
    vocab_size=config.vocab_size,
    code_embed_dim=config.code_embed_dim,
    # ... other params
)

# Replace embeddings
model.encoder.code_embeddings = pretrained_embeddings

# Optionally freeze embeddings
model.encoder.code_embeddings.weight.requires_grad = False
```

---

## Part 7: Recommendations

### Implementation Priority

**Phase 1: LSTM Baseline (Week 1-2)**
1. Implement `LSTMVisitEncoder` (small config)
2. Implement `LSTMProgressionModel`
3. Train on M1 MacBook with small dataset
4. Validate logic and pipeline

**Phase 2: Scale to RunPod (Week 3)**
1. Use medium LSTM config
2. Train on full dataset
3. Establish baseline performance

**Phase 3: Transformer Comparison (Week 4+)**
1. Implement Transformer variant
2. Compare LSTM vs Transformer
3. Decide on production model

### Why Start with LSTM

1. **Fast Development** - Works well on M1 MacBook
2. **Strong Baseline** - Proven for EHR sequences
3. **Memory Efficient** - 2-3x less memory than Transformer
4. **Interpretable** - Easier to debug
5. **Comparison Point** - Validate if Transformer complexity is needed

### When to Use Transformer

- After LSTM baseline is established
- When training on RunPod (24GB+)
- When long-range dependencies are critical
- For production deployment (best performance)

---

## Part 8: File Structure

```
src/ehrsequencing/models/
├── configs/
│   ├── __init__.py
│   ├── model_configs.py      # EHRModelConfig dataclass
│   └── presets.py             # SMALL/MEDIUM/LARGE configs
├── lstm/
│   ├── __init__.py
│   ├── visit_encoder.py       # LSTMVisitEncoder
│   └── progression_model.py   # LSTMProgressionModel
├── transformer/
│   ├── __init__.py
│   ├── visit_encoder.py       # TransformerVisitEncoder
│   └── progression_model.py   # TransformerProgressionModel
└── __init__.py

scripts/
├── train_disease_progression.py  # Main training script
└── compare_models.py              # LSTM vs Transformer comparison
```

---

## Summary

### Key Decisions

1. ✅ **LSTM as baseline** - Excellent fit for visit-grouped sequences
2. ✅ **Resource-aware configs** - Small/Medium/Large for different hardware
3. ✅ **M1 MacBook support** - Small configs for local development
4. ✅ **RunPod scaling** - Medium configs for realistic training
5. ✅ **Logically equivalent** - Same architecture, different scales

### Expected Performance

**LSTM Baseline:**
- M1 MacBook: ~10 min/epoch (small dataset)
- RunPod 24GB: ~2 min/epoch (full dataset)
- A40 40GB: ~1 min/epoch (full dataset)

**Memory Usage:**
- Small LSTM: ~3GB (M1 safe)
- Medium LSTM: ~8GB (RunPod safe)
- Large LSTM: ~18GB (A40 safe)

### Next Steps

1. Implement model config system
2. Implement LSTM baseline
3. Test on M1 MacBook (small config)
4. Scale to RunPod (medium config)
5. Compare with Transformer (optional)

---

**Document Version:** 1.0  
**Last Updated:** January 20, 2026  
**Next Review:** After LSTM baseline implementation
