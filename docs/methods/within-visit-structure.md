# Within-Visit Structure and Multi-Level Embeddings

**Date:** January 20, 2026  
**Focus:** How to structure codes within a visit when timestamps are unavailable

---

## The Problem

**Challenge:** A visit contains multiple codes (ICD, LOINC, RxNorm, etc.) that may not have individual timestamps.

```python
Visit 1 (2023-01-15):
  - E11.9 (Type 2 diabetes)
  - 4548-4 (HbA1c)
  - 38341003 (Hypertension)
  - RxNorm:860975 (Metformin)
  
# Question: What order should these codes be in?
# They're all "from the same visit" but conceptually different
```

**Key Questions:**
1. How do we impose structure within each visit?
2. Does order matter if codes lack individual timestamps?
3. What does the LSTM learn at each level?

---

## Part 1: Within-Visit Structure Approaches

### Approach 1: Semantic Grouping (Recommended)

**Idea:** Group codes by their semantic type, impose a canonical ordering.

```python
# Canonical ordering within a visit
VISIT_CODE_ORDER = [
    'diagnosis',      # ICD codes
    'procedure',      # CPT, SNOMED procedures
    'lab',           # LOINC codes
    'medication',    # RxNorm
    'vital',         # Vital signs
]

# Example visit after ordering
Visit 1:
  Diagnosis:  E11.9, 38341003
  Lab:        4548-4
  Medication: RxNorm:860975
```

**Advantages:**
- ✅ Consistent structure across all visits
- ✅ Reflects clinical workflow (diagnose → test → treat)
- ✅ Enables model to learn semantic patterns
- ✅ No arbitrary ordering

**Implementation:**

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class CodeWithType:
    code: str
    code_type: str  # 'diagnosis', 'lab', 'medication', etc.
    value: Optional[float] = None  # For lab values

class VisitStructurer:
    """Structure codes within a visit by semantic type."""
    
    # Canonical ordering
    TYPE_ORDER = {
        'diagnosis': 0,
        'procedure': 1,
        'lab': 2,
        'medication': 3,
        'vital': 4,
        'other': 5
    }
    
    def structure_visit(
        self, 
        codes: List[CodeWithType]
    ) -> List[CodeWithType]:
        """
        Order codes within a visit by semantic type.
        
        Within each type, codes are sorted alphabetically for consistency.
        """
        # Sort by type, then alphabetically within type
        structured = sorted(
            codes,
            key=lambda c: (self.TYPE_ORDER.get(c.code_type, 999), c.code)
        )
        return structured

# Example usage
visit_codes = [
    CodeWithType('RxNorm:860975', 'medication'),
    CodeWithType('E11.9', 'diagnosis'),
    CodeWithType('4548-4', 'lab', value=7.2),
    CodeWithType('38341003', 'diagnosis'),
]

structurer = VisitStructurer()
ordered_codes = structurer.structure_visit(visit_codes)

# Result:
# [E11.9, 38341003, 4548-4, RxNorm:860975]
# diagnosis → diagnosis → lab → medication
```

---

### Approach 2: Set-Based Representation (Order-Invariant)

**Idea:** Treat visit as an **unordered set**, use pooling to aggregate.

```python
class SetBasedVisitEncoder(nn.Module):
    """
    Encode visit as a set of codes (order-invariant).
    
    Uses pooling (mean, max, or attention) to aggregate.
    """
    
    def __init__(self, code_embed_dim: int, visit_embed_dim: int):
        super().__init__()
        self.code_embeddings = nn.Embedding(vocab_size, code_embed_dim)
        self.projection = nn.Linear(code_embed_dim, visit_embed_dim)
    
    def forward(self, visit_codes, visit_mask):
        """
        Args:
            visit_codes: [batch, max_codes_per_visit]
            visit_mask: [batch, max_codes_per_visit]
        
        Returns:
            visit_embedding: [batch, visit_embed_dim]
        """
        # Embed codes
        code_embeds = self.code_embeddings(visit_codes)  # [batch, codes, dim]
        
        # Mask padding
        code_embeds = code_embeds * visit_mask.unsqueeze(-1)
        
        # Aggregate (mean pooling - order invariant)
        visit_repr = code_embeds.sum(dim=1) / visit_mask.sum(dim=1, keepdim=True)
        
        # Project to visit space
        return self.projection(visit_repr)
```

**Advantages:**
- ✅ No ordering assumptions
- ✅ Mathematically clean (permutation invariant)
- ✅ Simple implementation

**Disadvantages:**
- ❌ Loses potential semantic ordering information
- ❌ Treats all codes equally (no clinical workflow)

---

### Approach 3: Attention-Based Aggregation (Best of Both Worlds)

**Idea:** Use **self-attention** within each visit to learn importance weights.

```python
class AttentionVisitEncoder(nn.Module):
    """
    Use self-attention to aggregate codes within a visit.
    
    Learns which codes are most important for the visit representation.
    """
    
    def __init__(
        self, 
        code_embed_dim: int, 
        visit_embed_dim: int,
        num_heads: int = 4
    ):
        super().__init__()
        self.code_embeddings = nn.Embedding(vocab_size, code_embed_dim)
        
        # Self-attention within visit
        self.self_attention = nn.MultiheadAttention(
            embed_dim=code_embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Project to visit embedding
        self.projection = nn.Linear(code_embed_dim, visit_embed_dim)
    
    def forward(self, visit_codes, visit_mask):
        """
        Args:
            visit_codes: [batch, max_codes_per_visit]
            visit_mask: [batch, max_codes_per_visit]
        
        Returns:
            visit_embedding: [batch, visit_embed_dim]
            attention_weights: [batch, num_heads, codes, codes]
        """
        # Embed codes
        code_embeds = self.code_embeddings(visit_codes)  # [batch, codes, dim]
        
        # Self-attention (codes attend to each other)
        attn_output, attn_weights = self.self_attention(
            code_embeds, code_embeds, code_embeds,
            key_padding_mask=~visit_mask.bool()
        )
        
        # Aggregate (mean over codes)
        visit_repr = attn_output.mean(dim=1)  # [batch, code_embed_dim]
        
        # Project to visit space
        visit_embedding = self.projection(visit_repr)
        
        return visit_embedding, attn_weights
```

**Advantages:**
- ✅ Learns importance of each code
- ✅ Order-invariant but captures relationships
- ✅ Interpretable (attention weights show which codes matter)
- ✅ Flexible (adapts to different visit types)

**Disadvantages:**
- ❌ More complex
- ❌ Requires more computation

---

### Approach 4: Hierarchical Type Embeddings

**Idea:** Add **type embeddings** to indicate code category.

```python
class HierarchicalVisitEncoder(nn.Module):
    """
    Add type embeddings to distinguish code categories.
    
    Similar to position embeddings in Transformers.
    """
    
    def __init__(self, code_embed_dim: int, visit_embed_dim: int):
        super().__init__()
        
        # Code embeddings
        self.code_embeddings = nn.Embedding(vocab_size, code_embed_dim)
        
        # Type embeddings (diagnosis, lab, medication, etc.)
        self.type_embeddings = nn.Embedding(
            num_embeddings=6,  # diagnosis, procedure, lab, med, vital, other
            embedding_dim=code_embed_dim
        )
        
        # LSTM over codes
        self.code_lstm = nn.LSTM(
            input_size=code_embed_dim,
            hidden_size=code_embed_dim,
            batch_first=True
        )
        
        self.projection = nn.Linear(code_embed_dim, visit_embed_dim)
    
    def forward(self, visit_codes, code_types, visit_mask):
        """
        Args:
            visit_codes: [batch, max_codes_per_visit]
            code_types: [batch, max_codes_per_visit] (0=diagnosis, 1=procedure, etc.)
            visit_mask: [batch, max_codes_per_visit]
        
        Returns:
            visit_embedding: [batch, visit_embed_dim]
        """
        # Embed codes and types
        code_embeds = self.code_embeddings(visit_codes)
        type_embeds = self.type_embeddings(code_types)
        
        # Combine (additive, like position embeddings)
        combined_embeds = code_embeds + type_embeds
        
        # LSTM over codes (now with type information)
        lstm_out, (hidden, _) = self.code_lstm(combined_embeds)
        
        # Use final hidden state
        visit_repr = hidden[-1]
        
        return self.projection(visit_repr)
```

**Advantages:**
- ✅ Explicitly encodes code type
- ✅ Works with sequential models (LSTM)
- ✅ Learns type-specific patterns

---

## Part 2: What Does LSTM Learn at Each Level?

### Multi-Level Embedding Learning

The LSTM architecture learns **three levels of embeddings**:

```
Level 1: Code Embeddings (learned)
    ↓
Level 2: Visit Embeddings (learned)
    ↓
Level 3: Patient Embeddings (learned)
```

Let me clarify **exactly what is learned** at each level:

---

### Level 1: Code Embeddings

**What:** Individual medical code representations (E11.9, 4548-4, etc.)

**Learned by:** `nn.Embedding` layer (or pre-trained from CEHR-BERT)

```python
self.code_embeddings = nn.Embedding(
    num_embeddings=vocab_size,  # e.g., 10,000 codes
    embedding_dim=code_embed_dim  # e.g., 128
)

# Input: code ID (integer)
code_id = 42  # e.g., E11.9 mapped to ID 42

# Output: code embedding (vector)
code_embedding = self.code_embeddings(code_id)  # [128]
```

**What is learned:**
- Semantic meaning of each code
- Relationships between codes (e.g., diabetes codes cluster together)
- Clinical context (e.g., HbA1c associated with diabetes)

**Example:**
```python
# After training, similar codes have similar embeddings
embedding_E11_9 = code_embeddings[42]    # Type 2 diabetes
embedding_E11_0 = code_embeddings[43]    # Type 1 diabetes
cosine_similarity(embedding_E11_9, embedding_E11_0)  # High similarity

embedding_J45_9 = code_embeddings[100]   # Asthma
cosine_similarity(embedding_E11_9, embedding_J45_9)  # Low similarity
```

**Can use pre-trained:**
```python
# Option 1: Learn from scratch
self.code_embeddings = nn.Embedding(vocab_size, code_embed_dim)

# Option 2: Use pre-trained (CEHR-BERT)
pretrained_embeds = load_cehrbert_embeddings()
self.code_embeddings = nn.Embedding.from_pretrained(pretrained_embeds, freeze=False)
```

---

### Level 2: Visit Embeddings

**What:** Representation of an entire visit (aggregation of codes)

**Learned by:** LSTM or pooling over codes within a visit

```python
# Within-visit LSTM
self.visit_lstm = nn.LSTM(
    input_size=code_embed_dim,  # 128
    hidden_size=code_embed_dim,  # 128
    num_layers=1,
    batch_first=True
)

# Input: sequence of code embeddings for one visit
visit_codes = [code_emb_1, code_emb_2, code_emb_3, ...]  # [num_codes, 128]

# Output: visit embedding
lstm_out, (hidden, _) = self.visit_lstm(visit_codes)
visit_embedding = hidden[-1]  # [128]
```

**What is learned:**
- How to combine multiple codes into a visit representation
- Importance of code order (if using LSTM) or relationships (if using attention)
- Visit-level patterns (e.g., "diabetes visit" vs "routine checkup")

**Example:**
```python
# Visit 1: Diabetes-related
visit_1_codes = [E11.9, 4548-4, RxNorm:860975]  # Diabetes, HbA1c, Metformin
visit_1_embedding = encode_visit(visit_1_codes)  # [128]

# Visit 2: Hypertension-related
visit_2_codes = [I10, 8480-6, RxNorm:197361]  # Hypertension, BP, Lisinopril
visit_2_embedding = encode_visit(visit_2_codes)  # [128]

# Different visit types have different embeddings
cosine_similarity(visit_1_embedding, visit_2_embedding)  # Moderate similarity
```

**Projection to visit space:**
```python
# Often project to a different dimension
self.visit_projection = nn.Linear(code_embed_dim, visit_embed_dim)
visit_embedding = self.visit_projection(visit_repr)  # [128] → [256]
```

---

### Level 3: Patient Embeddings

**What:** Representation of entire patient trajectory (sequence of visits)

**Learned by:** LSTM over visit sequence

```python
# Visit sequence LSTM
self.sequence_lstm = nn.LSTM(
    input_size=visit_embed_dim + 2,  # 256 + 2 time features
    hidden_size=hidden_dim,  # 512
    num_layers=2,
    batch_first=True
)

# Input: sequence of visit embeddings
patient_visits = [visit_emb_1, visit_emb_2, ..., visit_emb_N]  # [N, 256]

# Output: patient embedding
sequence_out, (final_hidden, _) = self.sequence_lstm(patient_visits)
patient_embedding = final_hidden[-1]  # [512]
```

**What is learned:**
- Temporal patterns across visits
- Disease progression trajectories
- Long-term dependencies (e.g., chronic conditions)
- Patient-level risk factors

**Example:**
```python
# Patient A: Stable diabetes
patient_A_visits = [diabetes_visit_1, diabetes_visit_2, diabetes_visit_3]
patient_A_embedding = encode_patient(patient_A_visits)  # [512]

# Patient B: Progressing CKD
patient_B_visits = [ckd_stage2_visit, ckd_stage3_visit, ckd_stage4_visit]
patient_B_embedding = encode_patient(patient_B_visits)  # [512]

# Different trajectories have different embeddings
cosine_similarity(patient_A_embedding, patient_B_embedding)  # Low similarity
```

---

## Part 3: Complete Architecture with All Three Levels

```python
class ThreeLevelLSTMEncoder(nn.Module):
    """
    Three-level LSTM encoder for EHR sequences.
    
    Level 1: Code embeddings (learned or pre-trained)
    Level 2: Visit embeddings (learned via LSTM/attention)
    Level 3: Patient embeddings (learned via LSTM over visits)
    """
    
    def __init__(
        self,
        vocab_size: int,
        code_embed_dim: int = 128,
        visit_embed_dim: int = 256,
        patient_embed_dim: int = 512,
        num_layers: int = 2,
        use_attention: bool = False,
        use_type_embeddings: bool = True
    ):
        super().__init__()
        
        # ====================================================================
        # LEVEL 1: Code Embeddings
        # ====================================================================
        self.code_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=code_embed_dim,
            padding_idx=0
        )
        
        # Optional: Type embeddings (diagnosis, lab, medication, etc.)
        if use_type_embeddings:
            self.type_embeddings = nn.Embedding(
                num_embeddings=6,  # 6 code types
                embedding_dim=code_embed_dim
            )
        else:
            self.type_embeddings = None
        
        # ====================================================================
        # LEVEL 2: Visit Embeddings
        # ====================================================================
        if use_attention:
            # Attention-based visit encoder
            self.visit_encoder = nn.MultiheadAttention(
                embed_dim=code_embed_dim,
                num_heads=4,
                batch_first=True
            )
        else:
            # LSTM-based visit encoder
            self.visit_encoder = nn.LSTM(
                input_size=code_embed_dim,
                hidden_size=code_embed_dim,
                num_layers=1,
                batch_first=True
            )
        
        # Project to visit embedding space
        self.visit_projection = nn.Linear(code_embed_dim, visit_embed_dim)
        
        # ====================================================================
        # LEVEL 3: Patient Embeddings
        # ====================================================================
        self.patient_encoder = nn.LSTM(
            input_size=visit_embed_dim + 2,  # +2 for time features
            hidden_size=patient_embed_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.use_attention = use_attention
    
    def encode_codes(self, codes, code_types=None):
        """
        LEVEL 1: Encode individual codes.
        
        Args:
            codes: [batch, num_codes]
            code_types: [batch, num_codes] (optional)
        
        Returns:
            code_embeddings: [batch, num_codes, code_embed_dim]
        """
        code_embeds = self.code_embeddings(codes)
        
        if self.type_embeddings is not None and code_types is not None:
            type_embeds = self.type_embeddings(code_types)
            code_embeds = code_embeds + type_embeds
        
        return code_embeds
    
    def encode_visit(self, visit_codes, code_types=None, visit_mask=None):
        """
        LEVEL 2: Encode a visit (aggregate codes).
        
        Args:
            visit_codes: [batch, max_codes_per_visit]
            code_types: [batch, max_codes_per_visit]
            visit_mask: [batch, max_codes_per_visit]
        
        Returns:
            visit_embedding: [batch, visit_embed_dim]
        """
        # Get code embeddings
        code_embeds = self.encode_codes(visit_codes, code_types)
        
        if self.use_attention:
            # Attention-based aggregation
            attn_out, _ = self.visit_encoder(
                code_embeds, code_embeds, code_embeds,
                key_padding_mask=~visit_mask.bool() if visit_mask is not None else None
            )
            visit_repr = attn_out.mean(dim=1)
        else:
            # LSTM-based aggregation
            lstm_out, (hidden, _) = self.visit_encoder(code_embeds)
            visit_repr = hidden[-1]
        
        # Project to visit space
        return self.visit_projection(visit_repr)
    
    def encode_patient(self, patient_visits, time_features, visit_mask):
        """
        LEVEL 3: Encode patient trajectory (sequence of visits).
        
        Args:
            patient_visits: [batch, num_visits, max_codes_per_visit]
            time_features: [batch, num_visits, 2]
            visit_mask: [batch, num_visits, max_codes_per_visit]
        
        Returns:
            patient_embedding: [batch, patient_embed_dim]
            visit_embeddings: [batch, num_visits, visit_embed_dim]
        """
        batch_size, num_visits, max_codes = patient_visits.shape
        
        # Encode each visit
        visit_embeds = []
        for i in range(num_visits):
            visit_emb = self.encode_visit(
                patient_visits[:, i, :],
                visit_mask=visit_mask[:, i, :] if visit_mask is not None else None
            )
            visit_embeds.append(visit_emb)
        
        visit_embeds = torch.stack(visit_embeds, dim=1)  # [batch, visits, visit_dim]
        
        # Add time features
        visit_embeds_with_time = torch.cat([visit_embeds, time_features], dim=-1)
        
        # LSTM over visits
        sequence_out, (final_hidden, _) = self.patient_encoder(visit_embeds_with_time)
        
        patient_embedding = final_hidden[-1]  # [batch, patient_embed_dim]
        
        return patient_embedding, visit_embeds
```

---

## Part 4: Summary and Recommendations

### Within-Visit Structure: Recommended Approach

**Recommended primary approach:**

**Semantic Grouping + Type Embeddings (Approach 1 + 4)**

```python
# 1. Order codes by semantic type
visit_codes = structure_by_type(raw_codes)  # diagnosis → lab → medication

# 2. Add type embeddings
code_embeds = code_embeddings(codes) + type_embeddings(types)

# 3. LSTM over ordered codes
visit_embedding = lstm(code_embeds)
```

**Why:**
- ✅ Consistent, interpretable structure
- ✅ Reflects clinical workflow
- ✅ Works well with LSTM
- ✅ Can still use pre-trained code embeddings

**Alternative: Attention-based (Approach 3)**
- Use when order-invariance is desired
- More flexible but more complex
- Better for Transformer models

---

### What LSTM Learns: Summary Table

| Level | What | Learned By | Dimension | Example |
|-------|------|-----------|-----------|---------|
| **Level 1** | Code embeddings | `nn.Embedding` | [vocab_size, 128] | E11.9 → [0.1, -0.3, ...] |
| **Level 2** | Visit embeddings | LSTM/Attention over codes | [256] | [diabetes visit] → [0.5, 0.2, ...] |
| **Level 3** | Patient embeddings | LSTM over visits | [512] | [patient trajectory] → [0.3, -0.1, ...] |

**Key Points:**
1. **All three levels are learned** (or Level 1 can be pre-trained)
2. **Each level captures different granularity**
3. **Gradients flow through all levels** during training
4. **Can freeze Level 1** if using pre-trained embeddings

---

### Implementation Priority

**Phase 1: Simple Baseline**
- Semantic grouping (canonical order)
- LSTM over codes (Level 2)
- LSTM over visits (Level 3)
- Learn all embeddings from scratch

**Phase 2: Enhanced**
- Add type embeddings
- Use pre-trained code embeddings (CEHR-BERT)
- Experiment with attention

**Phase 3: Advanced**
- Attention-based visit encoder
- Hierarchical attention (code → visit → patient)
- Multi-task learning

---

**Next:** Ready to implement the data pipeline with visit structuring?
