# Modern Medical Code Embeddings for EHR Sequencing

**Date:** January 19, 2026  
**Status:** Methodology & Implementation Guide

---

## Overview

This document presents modern approaches to learning embeddings for medical codes (ICD, LOINC, SNOMED, RxNorm) that capture:
- **Semantic relationships** - Similar codes have similar embeddings
- **Temporal dependencies** - Account for ordering and time gaps
- **Long-range dependencies** - Chronic conditions spanning years
- **Hierarchical structure** - Medical ontologies and code hierarchies

**Key Evolution from Word2Vec:**
- Word2Vec (2013): Static embeddings, no temporal awareness
- **Modern approaches (2024-2026)**: Temporal, hierarchical, context-aware

---

## Part 1: Problem Formulation

### Input: Patient Medical History

```python
Patient P001 Timeline:
  2020-01-15 09:00:00 | LOINC:4548-4    | Hemoglobin A1c
  2020-01-15 09:00:00 | SNOMED:44054006 | Diabetes mellitus type 2
  2020-01-15 09:00:00 | RXNORM:860975   | Metformin 500 MG
  2020-06-15 10:30:00 | LOINC:4548-4    | Hemoglobin A1c
  2020-12-15 14:00:00 | ICD10:E11.9     | Type 2 diabetes without complications
  2021-06-15 09:15:00 | LOINC:2339-0    | Glucose
```

### Goal: Learn Embedding Function

```
f: Code → ℝ^d

Examples:
  f(ICD10:E11.9)    → [0.23, -0.45, 0.67, ..., 0.12]  # 128-dim vector
  f(LOINC:4548-4)   → [0.19, -0.41, 0.71, ..., 0.08]  # Similar to E11.9 (diabetes)
  f(RXNORM:860975)  → [0.21, -0.43, 0.69, ..., 0.10]  # Metformin (diabetes drug)
```

**Desired Properties:**
1. **Semantic similarity**: Diabetes-related codes cluster together
2. **Temporal awareness**: Codes appearing in sequence have meaningful relationships
3. **Long-range dependencies**: Chronic conditions maintain coherence over time
4. **Hierarchy preservation**: Parent-child relationships in ontologies

---

## Part 2: Modern Embedding Approaches

### Approach 1: Temporal Skip-gram (Med2Vec++)

**Evolution from Word2Vec:**
- Original Med2Vec: Context window ignores time gaps
- **Modern**: Exponential decay based on time distance

#### Architecture

```python
import torch
import torch.nn as nn

class TemporalMed2Vec(nn.Module):
    """
    Med2Vec with temporal awareness via time-distance weighting.
    """
    def __init__(self, vocab_size, embed_dim=128, time_decay=0.1):
        super().__init__()
        self.code_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.time_decay = time_decay
    
    def forward(self, target_code, context_codes, time_deltas):
        """
        Args:
            target_code: [batch_size] - Target code indices
            context_codes: [batch_size, context_size] - Context code indices
            time_deltas: [batch_size, context_size] - Time gaps in days
        
        Returns:
            loss: Negative log-likelihood with temporal weighting
        """
        # Get embeddings
        target_embed = self.code_embeddings(target_code)  # [batch, embed_dim]
        context_embed = self.context_embeddings(context_codes)  # [batch, context, embed_dim]
        
        # Compute similarity scores
        scores = torch.einsum('be,bce->bc', target_embed, context_embed)  # [batch, context]
        
        # Apply temporal decay: weight = exp(-λ * Δt)
        temporal_weights = torch.exp(-self.time_decay * time_deltas)  # [batch, context]
        
        # Weighted loss
        log_probs = torch.log_softmax(scores, dim=-1)
        weighted_loss = -(log_probs * temporal_weights).sum() / temporal_weights.sum()
        
        return weighted_loss
```

#### Training Procedure

```python
def create_temporal_training_pairs(patient_sequences, window_size=5):
    """
    Create (target, context, time_delta) triplets with temporal information.
    
    Args:
        patient_sequences: List of [(timestamp, code), ...] per patient
        window_size: Maximum context window
    
    Returns:
        triplets: [(target_code, context_codes, time_deltas), ...]
    """
    triplets = []
    
    for sequence in patient_sequences:
        for i, (target_time, target_code) in enumerate(sequence):
            context_codes = []
            time_deltas = []
            
            # Look backward and forward in time
            for j in range(max(0, i - window_size), min(len(sequence), i + window_size + 1)):
                if i == j:  # Skip the target code itself
                    continue
                
                context_time, context_code = sequence[j]
                time_delta = abs((target_time - context_time).days)
                
                context_codes.append(context_code)
                time_deltas.append(time_delta)
            
            if context_codes:
                triplets.append((target_code, context_codes, time_deltas))
    
    return triplets

# Training loop
model = TemporalMed2Vec(vocab_size=10000, embed_dim=128, time_decay=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    for target, context, deltas in dataloader:
        loss = model(target, context, deltas)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Pros:**
- Simple extension of familiar Word2Vec
- Explicitly models temporal decay
- Fast training

**Cons:**
- Still limited context window
- No true long-range dependencies
- Treats all code types equally

---

### Approach 2: Hierarchical Temporal Embeddings

**Key Idea:** Different embedding strategies for different sequence representations.

#### For Visit-Grouped Sequences

```python
class VisitEmbedding(nn.Module):
    """
    Embed entire visits, then aggregate to patient level.
    """
    def __init__(self, vocab_size, code_embed_dim=64, visit_embed_dim=128):
        super().__init__()
        self.code_embeddings = nn.Embedding(vocab_size, code_embed_dim)
        self.visit_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=code_embed_dim, nhead=4),
            num_layers=2
        )
        self.visit_projection = nn.Linear(code_embed_dim, visit_embed_dim)
    
    def forward(self, visit_codes):
        """
        Args:
            visit_codes: [batch, max_codes_per_visit] - Codes in a visit
        
        Returns:
            visit_embedding: [batch, visit_embed_dim]
        """
        # Embed individual codes
        code_embeds = self.code_embeddings(visit_codes)  # [batch, codes, code_dim]
        
        # Aggregate codes within visit using Transformer
        visit_repr = self.visit_encoder(code_embeds.transpose(0, 1)).transpose(0, 1)
        
        # Pool to single visit embedding (mean pooling)
        visit_embed = visit_repr.mean(dim=1)  # [batch, code_dim]
        
        # Project to visit embedding space
        return self.visit_projection(visit_embed)  # [batch, visit_dim]
```

#### For Hierarchical Sequences (Code Type Aware)

```python
class HierarchicalCodeEmbedding(nn.Module):
    """
    Separate embeddings for different code types, then combine.
    """
    def __init__(self, vocab_sizes, embed_dim=128):
        super().__init__()
        # Separate embeddings for each code system
        self.icd_embeddings = nn.Embedding(vocab_sizes['ICD'], embed_dim)
        self.loinc_embeddings = nn.Embedding(vocab_sizes['LOINC'], embed_dim)
        self.snomed_embeddings = nn.Embedding(vocab_sizes['SNOMED'], embed_dim)
        self.rxnorm_embeddings = nn.Embedding(vocab_sizes['RXNORM'], embed_dim)
        
        # Cross-code-type attention
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=4)
    
    def forward(self, codes_by_type):
        """
        Args:
            codes_by_type: Dict[str, Tensor] - Codes grouped by type
                {'ICD': [batch, n_icd], 'LOINC': [batch, n_loinc], ...}
        
        Returns:
            unified_embeddings: [batch, total_codes, embed_dim]
        """
        embeds = []
        
        # Embed each code type separately
        if 'ICD' in codes_by_type:
            embeds.append(self.icd_embeddings(codes_by_type['ICD']))
        if 'LOINC' in codes_by_type:
            embeds.append(self.loinc_embeddings(codes_by_type['LOINC']))
        if 'SNOMED' in codes_by_type:
            embeds.append(self.snomed_embeddings(codes_by_type['SNOMED']))
        if 'RXNORM' in codes_by_type:
            embeds.append(self.rxnorm_embeddings(codes_by_type['RXNORM']))
        
        # Concatenate all embeddings
        all_embeds = torch.cat(embeds, dim=1)  # [batch, total_codes, embed_dim]
        
        # Apply cross-attention to capture relationships across code types
        attended, _ = self.cross_attention(
            all_embeds.transpose(0, 1),
            all_embeds.transpose(0, 1),
            all_embeds.transpose(0, 1)
        )
        
        return attended.transpose(0, 1)
```

---

### Approach 3: Transformer-based Contextualized Embeddings (RECOMMENDED)

**Key Insight:** Don't learn static embeddings - learn contextualized representations that depend on the entire patient history.

#### Architecture: Medical Code BERT (MedCodeBERT)

```python
class MedCodeBERT(nn.Module):
    """
    BERT-style model for medical codes with temporal encoding.
    
    Similar to BEHRT but focused on code embeddings.
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        max_seq_len=512,
        max_age=100,
        dropout=0.1
    ):
        super().__init__()
        
        # Code embeddings
        self.code_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Temporal embeddings
        self.age_embeddings = nn.Embedding(max_age * 12, embed_dim)  # Age in months
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.time_delta_projection = nn.Linear(1, embed_dim)  # Continuous time gaps
        
        # Code type embeddings (segment embeddings)
        self.code_type_embeddings = nn.Embedding(5, embed_dim)  # ICD, LOINC, SNOMED, RXNORM, OTHER
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, code_ids, ages, positions, time_deltas, code_types, attention_mask=None):
        """
        Args:
            code_ids: [batch, seq_len] - Code vocabulary indices
            ages: [batch, seq_len] - Patient age in months at each event
            positions: [batch, seq_len] - Position in sequence (0, 1, 2, ...)
            time_deltas: [batch, seq_len] - Days since previous event
            code_types: [batch, seq_len] - Code system type (0=ICD, 1=LOINC, etc.)
            attention_mask: [batch, seq_len] - Mask for padding
        
        Returns:
            contextualized_embeddings: [batch, seq_len, embed_dim]
        """
        # Get base embeddings
        code_embeds = self.code_embeddings(code_ids)
        age_embeds = self.age_embeddings(ages)
        pos_embeds = self.position_embeddings(positions)
        time_embeds = self.time_delta_projection(time_deltas.unsqueeze(-1))
        type_embeds = self.code_type_embeddings(code_types)
        
        # Combine all embeddings
        embeddings = code_embeds + age_embeds + pos_embeds + time_embeds + type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Apply transformer
        if attention_mask is not None:
            # Convert mask to transformer format (True = ignore)
            attention_mask = ~attention_mask.bool()
        
        output = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        return output
```

#### Pre-training Objectives

**1. Masked Language Modeling (MLM)**

```python
def masked_language_modeling_loss(model, batch):
    """
    Randomly mask 15% of codes and predict them.
    """
    code_ids = batch['code_ids'].clone()
    labels = code_ids.clone()
    
    # Create random mask (15% of tokens)
    mask_prob = torch.rand(code_ids.shape)
    mask = (mask_prob < 0.15) & (code_ids != 0)  # Don't mask padding
    
    # Replace masked tokens with [MASK] token (vocab_size - 1)
    code_ids[mask] = model.code_embeddings.num_embeddings - 1
    
    # Get predictions
    embeddings = model(
        code_ids,
        batch['ages'],
        batch['positions'],
        batch['time_deltas'],
        batch['code_types'],
        batch['attention_mask']
    )
    
    # Predict original codes
    logits = torch.matmul(embeddings, model.code_embeddings.weight.T)
    
    # Compute loss only on masked tokens
    loss = F.cross_entropy(
        logits[mask],
        labels[mask],
        ignore_index=0
    )
    
    return loss
```

**2. Next Visit Prediction**

```python
def next_visit_prediction_loss(model, batch):
    """
    Predict codes in next visit given current history.
    """
    # Split sequence into history and target
    history_len = batch['visit_boundaries'][-2]  # Second to last visit
    
    history_embeds = model(
        batch['code_ids'][:, :history_len],
        batch['ages'][:, :history_len],
        batch['positions'][:, :history_len],
        batch['time_deltas'][:, :history_len],
        batch['code_types'][:, :history_len],
        batch['attention_mask'][:, :history_len]
    )
    
    # Pool history to single vector
    history_repr = history_embeds.mean(dim=1)  # [batch, embed_dim]
    
    # Predict next visit codes (multi-label classification)
    next_visit_codes = batch['code_ids'][:, history_len:]
    logits = torch.matmul(history_repr, model.code_embeddings.weight.T)
    
    # Binary cross-entropy for multi-label
    targets = torch.zeros_like(logits)
    for i, codes in enumerate(next_visit_codes):
        targets[i, codes[codes != 0]] = 1
    
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    
    return loss
```

**3. Contrastive Learning for Similar Patients**

```python
def contrastive_patient_similarity_loss(model, batch1, batch2, labels):
    """
    Learn embeddings where similar patients are close, dissimilar are far.
    
    Args:
        batch1, batch2: Two patient sequences
        labels: 1 if similar (same disease), 0 if dissimilar
    """
    # Get patient-level embeddings
    embeds1 = model(batch1['code_ids'], ...).mean(dim=1)  # [batch, embed_dim]
    embeds2 = model(batch2['code_ids'], ...).mean(dim=1)
    
    # Cosine similarity
    similarity = F.cosine_similarity(embeds1, embeds2)
    
    # Contrastive loss
    loss = labels * (1 - similarity) + (1 - labels) * torch.clamp(similarity - 0.2, min=0)
    
    return loss.mean()
```

#### Training Strategy

```python
# Pre-training phase (unsupervised)
model = MedCodeBERT(vocab_size=10000, embed_dim=256, num_layers=6)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(100):
    for batch in pretrain_dataloader:
        # Multi-task pre-training
        mlm_loss = masked_language_modeling_loss(model, batch)
        nvp_loss = next_visit_prediction_loss(model, batch)
        
        loss = mlm_loss + 0.5 * nvp_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save pre-trained embeddings
torch.save(model.code_embeddings.state_dict(), 'pretrained_code_embeddings.pt')
```

---

### Approach 4: Graph-Enhanced Embeddings

**Key Idea:** Leverage medical ontology structure (ICD hierarchy, LOINC parts, SNOMED relationships).

```python
import torch_geometric as pyg

class GraphEnhancedCodeEmbedding(nn.Module):
    """
    Combine sequence-based embeddings with graph structure.
    """
    def __init__(self, vocab_size, embed_dim=128, num_gnn_layers=3):
        super().__init__()
        
        # Base embeddings (from MedCodeBERT)
        self.base_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Graph neural network for ontology structure
        self.gnn_layers = nn.ModuleList([
            pyg.nn.GATConv(embed_dim, embed_dim, heads=4, concat=False)
            for _ in range(num_gnn_layers)
        ])
    
    def forward(self, code_ids, ontology_graph):
        """
        Args:
            code_ids: [batch, seq_len] - Code indices
            ontology_graph: PyG Data object with edges representing relationships
        
        Returns:
            enhanced_embeddings: [batch, seq_len, embed_dim]
        """
        # Get base embeddings
        base_embeds = self.base_embeddings(code_ids)
        
        # Enhance with graph structure
        node_features = self.base_embeddings.weight  # All code embeddings
        
        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, ontology_graph.edge_index)
            node_features = F.relu(node_features)
        
        # Look up enhanced embeddings for input codes
        enhanced_embeds = node_features[code_ids]
        
        return enhanced_embeds
```

---

## Part 3: Recommendations for Your Use Case

### For ICD Code E11.9 (Type 2 Diabetes)

**Recommended Approach: MedCodeBERT (Approach 3)**

**Rationale:**
1. **Temporal awareness**: Diabetes is chronic - need long-range dependencies
2. **Context-dependent**: E11.9 meaning changes based on patient history
3. **Multi-code relationships**: Often co-occurs with LOINC (HbA1c), RxNorm (Metformin)
4. **Scalability**: Transformer handles variable-length sequences well

### For LOINC Code 4548-4 (Hemoglobin A1c)

**Recommended Approach: Hierarchical + MedCodeBERT**

**Rationale:**
1. **LOINC structure**: 6-part hierarchy (Component, Property, Time, System, Scale, Method)
2. **Temporal patterns**: HbA1c measured quarterly for diabetes monitoring
3. **Value-aware**: Embedding should reflect numeric values (high vs normal)

```python
class LOINCEmbedding(nn.Module):
    """
    LOINC-specific embedding with structure awareness.
    """
    def __init__(self, loinc_vocab_size, embed_dim=128):
        super().__init__()
        
        # Base LOINC code embedding
        self.code_embedding = nn.Embedding(loinc_vocab_size, embed_dim)
        
        # LOINC part embeddings
        self.component_embedding = nn.Embedding(1000, embed_dim // 6)
        self.property_embedding = nn.Embedding(100, embed_dim // 6)
        self.time_embedding = nn.Embedding(50, embed_dim // 6)
        self.system_embedding = nn.Embedding(200, embed_dim // 6)
        self.scale_embedding = nn.Embedding(20, embed_dim // 6)
        self.method_embedding = nn.Embedding(500, embed_dim // 6)
        
        # Value encoder (for numeric lab values)
        self.value_encoder = nn.Linear(1, embed_dim // 4)
        
        # Combine all
        self.projection = nn.Linear(embed_dim + embed_dim // 4, embed_dim)
    
    def forward(self, loinc_code, loinc_parts, value):
        """
        Args:
            loinc_code: LOINC code ID
            loinc_parts: Dict with component, property, time, system, scale, method IDs
            value: Numeric lab value (normalized)
        """
        # Base embedding
        code_embed = self.code_embedding(loinc_code)
        
        # Part embeddings
        part_embeds = torch.cat([
            self.component_embedding(loinc_parts['component']),
            self.property_embedding(loinc_parts['property']),
            self.time_embedding(loinc_parts['time']),
            self.system_embedding(loinc_parts['system']),
            self.scale_embedding(loinc_parts['scale']),
            self.method_embedding(loinc_parts['method'])
        ], dim=-1)
        
        # Value embedding
        value_embed = self.value_encoder(value.unsqueeze(-1))
        
        # Combine
        combined = torch.cat([code_embed + part_embeds, value_embed], dim=-1)
        return self.projection(combined)
```

---

## Part 4: Implementation Roadmap

### Phase 1: Data Preparation (Week 1)

**Goal:** Prepare sequences in multiple representations

```python
# scripts/prepare_sequences.py
from ehrsequencing.data.adapters import SyntheaAdapter
from ehrsequencing.data.sequences import SequenceBuilder

# Load data
adapter = SyntheaAdapter('data/synthea/')
events = adapter.load_events()

# Create different representations
builder = SequenceBuilder()

# Flat sequence
flat_sequences = builder.build(events, strategy='flat')
# Output: [code1, code2, code3, ...]

# Visit-grouped
visit_sequences = builder.build(events, strategy='visit-grouped')
# Output: [[visit1_codes], [visit2_codes], ...]

# Hierarchical
hierarchical_sequences = builder.build(events, strategy='hierarchical')
# Output: {
#   'ICD': [code1, code2, ...],
#   'LOINC': [code3, code4, ...],
#   'RXNORM': [code5, code6, ...]
# }

# Save
torch.save({
    'flat': flat_sequences,
    'visit': visit_sequences,
    'hierarchical': hierarchical_sequences
}, 'data/processed/sequences.pt')
```

### Phase 2: Baseline Embeddings (Week 2)

**Goal:** Implement Temporal Skip-gram as baseline

```python
# src/ehrsequencing/embeddings/temporal_skipgram.py
# (Implementation from Approach 1 above)

# examples/train_temporal_skipgram.py
from ehrsequencing.embeddings.temporal_skipgram import TemporalMed2Vec

model = TemporalMed2Vec(vocab_size=10000, embed_dim=128)
# Train on flat sequences
# Evaluate: nearest neighbors, clustering
```

### Phase 3: MedCodeBERT Implementation (Week 3-4)

**Goal:** Implement transformer-based contextualized embeddings

```python
# src/ehrsequencing/embeddings/medcodebert.py
# (Implementation from Approach 3 above)

# examples/pretrain_medcodebert.py
from ehrsequencing.embeddings.medcodebert import MedCodeBERT

model = MedCodeBERT(vocab_size=10000, embed_dim=256, num_layers=6)
# Pre-train with MLM + Next Visit Prediction
# Save pre-trained model
```

### Phase 4: Evaluation (Week 5)

**Goal:** Compare embedding quality

```python
# src/ehrsequencing/evaluation/embedding_eval.py

def evaluate_embeddings(embeddings, test_data):
    """
    Evaluate embedding quality.
    """
    metrics = {}
    
    # 1. Nearest neighbor accuracy
    # For diabetes codes, are nearest neighbors also diabetes-related?
    metrics['nn_accuracy'] = nearest_neighbor_accuracy(embeddings, test_data)
    
    # 2. Clustering quality
    # Do similar diseases cluster together?
    metrics['silhouette'] = clustering_quality(embeddings, test_data)
    
    # 3. Downstream task performance
    # Use embeddings for diagnosis prediction
    metrics['diagnosis_auc'] = diagnosis_prediction_auc(embeddings, test_data)
    
    return metrics
```

### Phase 5: Fine-tuning for Applications (Week 6+)

**Goal:** Use embeddings for downstream tasks

```python
# Disease progression modeling
from ehrsequencing.models import DiseaseProgressionModel

model = DiseaseProgressionModel(
    code_embeddings=pretrained_medcodebert.code_embeddings,
    hidden_dim=256
)
model.train(sequences, targets)

# Patient segmentation
from ehrsequencing.clustering import PatientSegmentation

segmentation = PatientSegmentation(embedding_model=pretrained_medcodebert)
clusters = segmentation.fit_predict(patient_sequences)
```

---

## Part 5: Comparison Table

| Approach | Temporal Aware | Long-Range Deps | Hierarchical | Training Time | Best For |
|----------|---------------|-----------------|--------------|---------------|----------|
| **Temporal Skip-gram** | ✅ (decay) | ❌ (window) | ❌ | Fast (hours) | Baseline, quick experiments |
| **Hierarchical Embeddings** | ⚠️ (limited) | ❌ | ✅ | Medium (days) | Visit-grouped sequences |
| **MedCodeBERT** | ✅✅ (full) | ✅✅ (transformer) | ⚠️ (via types) | Slow (weeks) | **Production, best quality** |
| **Graph-Enhanced** | ⚠️ | ✅ (via graph) | ✅✅ | Slow (weeks) | When ontology is critical |

---

## Part 6: Practical Recommendations

### Start Simple, Iterate

**Week 1-2: Temporal Skip-gram**
- Quick to implement and train
- Establishes baseline
- Validates data pipeline

**Week 3-4: MedCodeBERT**
- State-of-the-art quality
- Handles all sequence types
- Pre-train once, use for all tasks

**Week 5+: Task-Specific Fine-tuning**
- Disease progression
- Patient segmentation
- Phenotype discovery

### Handling Different Sequence Representations

**For Flat Sequences:**
```python
# Use MedCodeBERT directly
embeddings = medcodebert(flat_sequence)
```

**For Visit-Grouped:**
```python
# Embed each visit, then sequence of visits
visit_embeds = [medcodebert(visit).mean(dim=0) for visit in visits]
patient_sequence = torch.stack(visit_embeds)
```

**For Hierarchical:**
```python
# Embed each code type separately, then combine
icd_embeds = medcodebert(icd_codes)
loinc_embeds = loinc_embedding(loinc_codes, loinc_parts, values)
combined = cross_attention(icd_embeds, loinc_embeds)
```

---

## Part 7: Code Examples

### Complete Training Script

```python
# examples/train_code_embeddings.py

import torch
from torch.utils.data import DataLoader
from ehrsequencing.data import load_sequences
from ehrsequencing.embeddings import MedCodeBERT
from ehrsequencing.evaluation import evaluate_embeddings

def main():
    # Load data
    sequences = load_sequences('data/processed/sequences.pt')
    train_loader = DataLoader(sequences['train'], batch_size=32, shuffle=True)
    val_loader = DataLoader(sequences['val'], batch_size=32)
    
    # Initialize model
    model = MedCodeBERT(
        vocab_size=len(sequences['vocab']),
        embed_dim=256,
        num_layers=6,
        num_heads=8
    )
    
    # Pre-training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(100):
        # Train
        model.train()
        for batch in train_loader:
            loss = masked_language_modeling_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate
        if epoch % 10 == 0:
            model.eval()
            metrics = evaluate_embeddings(model, val_loader)
            print(f"Epoch {epoch}: {metrics}")
    
    # Save
    torch.save(model.state_dict(), 'checkpoints/medcodebert_pretrained.pt')
    
    # Extract static embeddings for downstream use
    code_embeddings = model.code_embeddings.weight.detach()
    torch.save(code_embeddings, 'checkpoints/code_embeddings.pt')

if __name__ == '__main__':
    main()
```

---

## References

1. **Med2Vec** - Choi et al., "Multi-layer Representation Learning for Medical Concepts" (2016)
2. **BEHRT** - Li et al., "BEHRT: Transformer for Electronic Health Records" (2020)
3. **Med-BERT** - Rasmy et al., "Med-BERT: Pre-trained Contextualized Embeddings" (2021)
4. **CEHR-BERT** - Pang et al., "CEHR-BERT: Incorporating Temporal Information" (2021)
5. **GraphCare** - Choi et al., "Learning the Graphical Structure of EHR" (2020)

---

**Document Version:** 1.0  
**Last Updated:** January 19, 2026  
**Next Review:** After Phase 3 implementation
