# Within-Visit Structure: Pipeline Analysis

**Date:** January 20, 2026  
**Purpose:** Understanding how within-visit code structure is handled across the data pipeline and model

---

## Table of Contents

1. [Overview: The Within-Visit Structure Question](#overview)
2. [Data Pipeline: Structure Preservation](#data-pipeline)
3. [Model: Structure Handling](#model)
4. [Current State Analysis](#current-state-analysis)
5. [Potential Improvements](#potential-improvements)

---

## Overview: The Within-Visit Structure Question

**The Question:** When a visit contains multiple medical codes, does their **order** and **semantic grouping** matter?

**Example Visit:**
```python
Visit on 2023-01-15:
  - E11.9 (Diabetes diagnosis)
  - LOINC:2160-0 (Creatinine lab)
  - LOINC:4548-4 (HbA1c lab)
  - RX:860975 (Metformin medication)
```

**Possible Approaches:**

1. **Unordered Set (Bag-of-Codes)**
   - Treat codes as a set, order doesn't matter
   - Use permutation-invariant aggregation (mean, max, attention)
   
2. **Semantic Ordering**
   - Group by code type: diagnosis → labs → procedures → medications
   - Use sequence model (RNN, Transformer) within visit
   
3. **Temporal Ordering**
   - Order by exact timestamp within visit
   - Preserve fine-grained temporal structure

**Current Implementation:** Hybrid approach - data pipeline supports semantic ordering, but model treats as unordered set.

---

## Data Pipeline: Structure Preservation

The data pipeline (VisitGrouper → PatientSequenceBuilder) **preserves** within-visit structure.

### Step 1: VisitGrouper - Code Organization

**File:** `src/ehrsequencing/data/visit_grouper.py`

#### Visit Data Structure

```python
@dataclass
class Visit:
    visit_id: str
    patient_id: str
    timestamp: datetime
    encounter_id: Optional[str] = None
    codes_by_type: Optional[Dict[str, List[str]]] = None  # ← Structured
    codes_flat: Optional[List[str]] = None                # ← Unstructured
    metadata: Optional[Dict] = None
```

**Key insight:** Visit stores codes **both ways**:
1. `codes_by_type`: Dictionary grouping codes by semantic type
2. `codes_flat`: Simple list of all codes (order of insertion)

---

#### Code Grouping by Type

```python
class VisitGrouper:
    def __init__(
        self,
        preserve_code_types: bool = True,  # ← Controls structure preservation
        code_type_order: Optional[List[str]] = None
    ):
        self.code_type_order = code_type_order or [
            'diagnosis',   # ICD codes first
            'lab',         # Lab results second
            'vital',       # Vital signs third
            'procedure',   # Procedures fourth
            'medication'   # Medications last
        ]
```

**What it does:**

```python
def _create_visit(self, events: List[MedicalEvent]) -> Visit:
    # Group codes by type
    codes_by_type = defaultdict(list)
    codes_flat = []
    
    for event in events:
        code = event.code
        codes_flat.append(code)  # Flat list (insertion order)
        
        if self.preserve_code_types:
            codes_by_type[event.code_type].append(code)  # Grouped by type
    
    return Visit(
        codes_by_type=dict(codes_by_type),
        codes_flat=codes_flat,
        ...
    )
```

---

#### Semantic Ordering via `get_ordered_codes()`

```python
class Visit:
    def get_ordered_codes(self, type_order: Optional[List[str]] = None) -> List[str]:
        """
        Get codes ordered by semantic type.
        
        Returns codes in order: diagnosis → lab → vital → procedure → medication
        """
        if type_order is None:
            type_order = ['diagnosis', 'lab', 'vital', 'procedure', 'medication']
        
        ordered_codes = []
        for code_type in type_order:
            if code_type in self.codes_by_type:
                ordered_codes.extend(self.codes_by_type[code_type])
        
        return ordered_codes
```

**Example:**

```python
# Visit with mixed codes (inserted in arbitrary order)
visit.codes_flat = [
    'RX:860975',      # Medication
    'E11.9',          # Diagnosis
    'LOINC:2160-0',   # Lab
    'LOINC:4548-4'    # Lab
]

# Semantic ordering
visit.get_ordered_codes() = [
    'E11.9',          # Diagnosis (first in semantic order)
    'LOINC:2160-0',   # Lab (second)
    'LOINC:4548-4',   # Lab (second)
    'RX:860975'       # Medication (last)
]
```

---

### Step 2: PatientSequenceBuilder - Semantic Order Option

**File:** `src/ehrsequencing/data/sequence_builder.py`

#### Configuration

```python
class PatientSequenceBuilder:
    def __init__(
        self,
        use_semantic_order: bool = True,  # ← Controls ordering
        ...
    ):
        self.use_semantic_order = use_semantic_order
```

#### Encoding with Semantic Order

```python
def encode_sequence(self, sequence: PatientSequence) -> Dict[str, Any]:
    # Get code sequence (with or without semantic ordering)
    code_sequence = sequence.get_code_sequence(self.use_semantic_order)
    
    # Encode and pad each visit
    for visit_codes in code_sequence:
        encoded_codes = [
            self.vocab.get(code, self.unk_id)
            for code in visit_codes[:self.max_codes_per_visit]
        ]
        # Order is preserved in encoding!
```

**What happens:**

```python
# If use_semantic_order=True
PatientSequence.get_code_sequence(use_semantic_order=True):
    return [visit.get_ordered_codes() for visit in self.visits]
    # Returns: [[diagnosis, labs, meds], [diagnosis, labs], ...]

# If use_semantic_order=False
PatientSequence.get_code_sequence(use_semantic_order=False):
    return [visit.get_all_codes() for visit in self.visits]
    # Returns: [[codes in insertion order], [codes], ...]
```

---

### Data Pipeline Summary

**Structure Preservation:**
```
Raw EHR Events
    ↓
VisitGrouper (preserve_code_types=True)
    ↓
Visit objects with codes_by_type
    ↓
PatientSequenceBuilder (use_semantic_order=True)
    ↓
Encoded sequences with semantic ordering
    ↓ [batch, visits, codes]
    codes within each visit are semantically ordered
```

**Example Flow:**

```python
# Step 1: Raw events
events = [
    MedicalEvent(code='RX:860975', code_type='medication', timestamp=...),
    MedicalEvent(code='E11.9', code_type='diagnosis', timestamp=...),
    MedicalEvent(code='LOINC:2160-0', code_type='lab', timestamp=...),
]

# Step 2: VisitGrouper creates structured visit
visit = Visit(
    codes_by_type={
        'diagnosis': ['E11.9'],
        'lab': ['LOINC:2160-0'],
        'medication': ['RX:860975']
    },
    codes_flat=['RX:860975', 'E11.9', 'LOINC:2160-0']
)

# Step 3: Get ordered codes
visit.get_ordered_codes() = ['E11.9', 'LOINC:2160-0', 'RX:860975']
#                            └─────┘  └────────────┘  └─────────┘
#                            diagnosis    lab           medication

# Step 4: Encode to indices (order preserved)
encoded = [42, 523, 1200]  # Indices for [E11.9, LOINC:2160-0, RX:860975]
```

---

## Model: Structure Handling

The LSTM baseline model **discards** within-visit ordering through set-based aggregation.

### Step 3: LSTMBaseline - Set-Based Aggregation

**File:** `src/ehrsequencing/models/lstm_baseline.py`

#### Current Approach: Permutation-Invariant

```python
class VisitEncoder(nn.Module):
    def __init__(
        self,
        aggregation: str = 'mean',  # ← Permutation-invariant
        ...
    ):
        self.aggregation = aggregation
        # Options: 'mean', 'sum', 'max', 'attention'
        # All are permutation-invariant!
```

**What this means:**

```python
# Visit codes (semantically ordered from data pipeline)
visit_codes = ['E11.9', 'LOINC:2160-0', 'RX:860975']
embeddings = [emb_1, emb_2, emb_3]

# Mean aggregation (permutation-invariant)
visit_vector = mean(embeddings)
                = (emb_1 + emb_2 + emb_3) / 3

# Order doesn't matter!
shuffled_embeddings = [emb_3, emb_1, emb_2]  # Different order
shuffled_visit_vector = mean(shuffled_embeddings)
                      = (emb_3 + emb_1 + emb_2) / 3
                      = same as visit_vector

# Result: visit_vector == shuffled_visit_vector
```

---

#### Why Permutation-Invariant?

**Design Choice Rationale:**

1. **Clinical Reality:** Within a visit, code order is often arbitrary
   ```python
   # These two visits should have similar representations:
   Visit 1: [Diagnosis, Lab, Medication]
   Visit 2: [Lab, Diagnosis, Medication]
   # Same clinical content, different recording order
   ```

2. **Data Inconsistency:** EHR systems may not reliably preserve temporal order within visit
   ```python
   # Lab drawn at 9:00 AM might be recorded after medication at 10:00 AM
   # Due to data entry workflow, not actual timing
   ```

3. **Simplicity:** Set-based aggregation is simple and robust
   ```python
   # No need to worry about:
   # - Positional embeddings
   # - Sequence padding within visits
   # - Training a within-visit sequence model
   ```

---

#### Aggregation Methods Analysis

**All current methods are permutation-invariant:**

| Method | Formula | Permutation-Invariant? | Notes |
|--------|---------|----------------------|-------|
| **Mean** | `(Σ embeddings) / n` | ✅ Yes | Order doesn't affect sum or count |
| **Sum** | `Σ embeddings` | ✅ Yes | Commutative operation |
| **Max** | `max(embeddings)` | ✅ Yes | Max doesn't depend on order |
| **Attention** | `Σ (α_i · emb_i)` | ✅ Yes | Weights α_i computed independently |

**Key Point:** Even attention is permutation-invariant!

```python
# Attention computation
for each code:
    score_i = attention_network(embedding_i)  # Independent scoring
weights = softmax(scores)  # Normalize
visit_vector = Σ (weight_i * embedding_i)  # Weighted sum

# Each code scored independently
# → shuffling codes produces same weights (just in different order)
# → weighted sum is same (commutative)
```

---

### Model Summary

**Current Treatment:**
```
Ordered codes from data pipeline
    [E11.9, LOINC:2160-0, RX:860975]
    ↓
Embed each code
    [emb_1, emb_2, emb_3]
    ↓
VisitEncoder (mean/attention)
    visit_vector = aggregate(embeddings)
    ↓
Order discarded - treated as unordered set
```

---

## Current State Analysis

### What Happens to Semantic Order?

**The Journey of Within-Visit Structure:**

```
Step 1: EHR Data
  Events have code_type: 'diagnosis', 'lab', 'medication'

Step 2: VisitGrouper
  Preserves structure in codes_by_type
  ✅ Structure PRESERVED

Step 3: PatientSequenceBuilder (use_semantic_order=True)
  Encodes codes in semantic order
  ✅ Structure PRESERVED

Step 4: LSTMBaseline.forward()
  Embeds codes (order preserved in tensor)
  visit_codes: [batch, visits, codes]
  ✅ Structure STILL PRESERVED

Step 5: VisitEncoder.forward()
  Aggregates with mean/attention (permutation-invariant)
  ❌ Structure DISCARDED

Final: LSTM
  Receives visit vectors with no within-visit order info
  ❌ Structure NOT USED
```

---

### Visual Data Flow

```python
# Patient Visit 1
Raw data: [RX:860975, E11.9, LOINC:2160-0]  # Arbitrary order

↓ VisitGrouper (structure preserved)

codes_by_type: {
    'diagnosis': ['E11.9'],
    'lab': ['LOINC:2160-0'],
    'medication': ['RX:860975']
}

↓ PatientSequenceBuilder (semantic order applied)

ordered_codes: ['E11.9', 'LOINC:2160-0', 'RX:860975']
                └──────┘  └────────────┘  └─────────┘
                diagnosis      lab          medication

↓ Encoded to indices (order preserved)

visit_codes_tensor: [42, 523, 1200]

↓ LSTMBaseline.embedding (order still preserved)

code_embeddings: [
    [0.5, -0.3, 0.8],   # emb_42 (diagnosis)
    [0.3, 0.6, -0.2],   # emb_523 (lab)
    [0.7, 0.1, 0.4]     # emb_1200 (medication)
]

↓ VisitEncoder (mean aggregation - ORDER DISCARDED)

visit_vector: [0.5, 0.133, 0.333]
              = mean of 3 embeddings
              = same result if codes were shuffled!

↓ LSTM (no within-visit order information)

LSTM only sees: visit_vector (aggregated representation)
```

---

### Is This a Problem?

**Arguments for Current Approach (Set-Based):**

1. **Clinical Validity**
   - Many codes within a visit occur "simultaneously" (e.g., all labs drawn at once)
   - Recording order is often arbitrary (data entry workflow)
   - Semantic grouping (diagnosis first) may be artificial

2. **Data Quality**
   - EHR timestamps within visit may be unreliable
   - Different hospitals have different recording practices
   - Set-based approach is robust to these inconsistencies

3. **Empirical Success**
   - Many successful EHR models use set-based visit encoding
   - Mean pooling is simple and effective
   - Attention can learn to weight important codes

4. **Computational Efficiency**
   - No need for sequence model within visit
   - Faster training and inference
   - Fewer parameters

---

**Arguments for Using Structure (Order-Aware):**

1. **Temporal Causality**
   - Diagnosis → Lab order → Treatment is meaningful
   - E.g., "Diabetes diagnosed, then HbA1c tested, then Metformin prescribed"
   - Order captures clinical workflow

2. **Information Loss**
   - Semantic grouping provides useful inductive bias
   - Code type (diagnosis vs. lab) is meaningful
   - Attention alone may not learn this structure

3. **Interpretability**
   - Structured visits are more interpretable
   - Can explain: "Model focused on diagnosis codes, then labs"
   - Set-based aggregation hides this

4. **Recent Research**
   - Some recent papers show benefits of within-visit structure
   - Particularly for complex multi-morbid patients
   - Hierarchical models (code → type → visit) can improve performance

---

### Empirical Analysis: Does Order Matter?

**Experiment Design:**

```python
# Test 1: Original semantic order
visit_codes_ordered = ['E11.9', 'LOINC:2160-0', 'RX:860975']
visit_vector_ordered = model.visit_encoder(embed(visit_codes_ordered))

# Test 2: Shuffled order
visit_codes_shuffled = ['RX:860975', 'LOINC:2160-0', 'E11.9']
visit_vector_shuffled = model.visit_encoder(embed(visit_codes_shuffled))

# With mean/attention aggregation:
assert torch.allclose(visit_vector_ordered, visit_vector_shuffled)
# ✅ Vectors are identical (permutation-invariant)

# Test 3: Compare model performance
# Train two models:
# - Model A: use_semantic_order=True (current)
# - Model B: use_semantic_order=False (random order)

# With set-based aggregation:
# Performance should be identical (both discard order)
```

**Expected Result with Current Model:**
- Model performance **same** with or without semantic ordering
- Because VisitEncoder discards order anyway

---

## Potential Improvements

### Option 1: Keep Current Approach (Set-Based)

**Rationale:** If empirical results show no benefit from ordering

**Recommendation:**
- Set `use_semantic_order=False` in PatientSequenceBuilder
- Simplify code by removing unused semantic ordering logic
- Document that model treats visits as unordered sets

**Code change:**
```python
# In PatientSequenceBuilder.__init__
self.use_semantic_order = False  # No need to order if model ignores it
```

---

### Option 2: Add Order-Aware Aggregation

**Approach:** Replace mean pooling with sequence model within visit

#### Implementation: GRU Within Visit

```python
class SequentialVisitEncoder(nn.Module):
    """
    Order-aware visit encoder using GRU.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, visit_embeddings, visit_mask):
        """
        Args:
            visit_embeddings: [batch, max_codes, embed_dim]
                              Codes in semantic order!
            visit_mask: [batch, max_codes]
        
        Returns:
            visit_vector: [batch, hidden_dim]
        """
        # Pack sequences (skip padding)
        lengths = visit_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            visit_embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        # GRU processes codes in order
        packed_output, final_hidden = self.gru(packed)
        
        # Use final hidden state as visit representation
        visit_vector = final_hidden.squeeze(0)  # [batch, hidden_dim]
        
        return self.dropout(visit_vector)
```

**Usage:**
```python
# In LSTMBaseline.__init__
self.visit_encoder = SequentialVisitEncoder(
    embedding_dim=embedding_dim,
    hidden_dim=embedding_dim  # Match dimensions
)

# Now semantic order matters!
# Model learns: diagnosis → lab → medication pattern
```

---

#### Implementation: Transformer Within Visit

```python
class TransformerVisitEncoder(nn.Module):
    """
    Order-aware visit encoder using Transformer with positional encoding.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Positional encoding for order
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, visit_embeddings, visit_mask):
        """
        Args:
            visit_embeddings: [batch, max_codes, embed_dim]
                              Codes in semantic order!
            visit_mask: [batch, max_codes]
        
        Returns:
            visit_vector: [batch, embed_dim]
        """
        # Add positional encoding (encodes order)
        visit_embeddings = self.pos_encoder(visit_embeddings)
        
        # Mask for attention
        attn_mask = ~visit_mask.bool()
        
        # Transformer processes with order awareness
        output = self.transformer(
            visit_embeddings,
            src_key_padding_mask=attn_mask
        )
        
        # Mean pool over codes (after attending with order info)
        visit_mask_expanded = visit_mask.unsqueeze(-1)
        masked_output = output * visit_mask_expanded
        visit_vector = masked_output.sum(dim=1) / visit_mask.sum(dim=1, keepdim=True)
        
        return self.dropout(visit_vector)


class PositionalEncoding(nn.Module):
    """Positional encoding for order information."""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:x.size(1)]
```

---

### Option 3: Hierarchical Type Embeddings

**Approach:** Embed code type separately, add to code embedding

```python
class TypeAwareVisitEncoder(nn.Module):
    """
    Visit encoder that uses code type information.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_code_types: int = 5,  # diagnosis, lab, vital, procedure, medication
        aggregation: str = 'mean'
    ):
        super().__init__()
        
        # Type embeddings
        self.type_embeddings = nn.Embedding(num_code_types, embedding_dim)
        
        # Same aggregation as before
        self.aggregation = aggregation
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.Tanh(),
                nn.Linear(embedding_dim // 2, 1)
            )
    
    def forward(self, visit_embeddings, code_types, visit_mask):
        """
        Args:
            visit_embeddings: [batch, max_codes, embed_dim]
            code_types: [batch, max_codes] - Type ID for each code
            visit_mask: [batch, max_codes]
        
        Returns:
            visit_vector: [batch, embed_dim]
        """
        # Add type information to code embeddings
        type_embeds = self.type_embeddings(code_types)
        enhanced_embeddings = visit_embeddings + type_embeds
        
        # Aggregate (now with type information)
        visit_mask = visit_mask.unsqueeze(-1)
        
        if self.aggregation == 'mean':
            masked_embeddings = enhanced_embeddings * visit_mask
            visit_vector = masked_embeddings.sum(dim=1) / visit_mask.sum(dim=1).clamp(min=1)
        elif self.aggregation == 'attention':
            attention_scores = self.attention(enhanced_embeddings)
            attention_scores = attention_scores.masked_fill(visit_mask == 0, float('-inf'))
            attention_weights = torch.softmax(attention_scores, dim=1)
            visit_vector = (enhanced_embeddings * attention_weights).sum(dim=1)
        
        return visit_vector
```

**Usage:**
```python
# Need to pass code types from data
# In sequence_builder.py, add:
result['code_types'] = [[type_vocab[code_type] for code in visit] for visit in visits]

# In model forward:
visit_vector = self.visit_encoder(
    visit_embeddings=code_embeddings,
    code_types=code_types,      # New parameter
    visit_mask=visit_mask
)
```

**Benefit:** Uses type information without requiring order

---

### Option 4: Separate Encoders per Code Type

**Approach:** Encode each type separately, then combine

```python
class MultiTypeVisitEncoder(nn.Module):
    """
    Encode each code type separately, then combine.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        
        # Separate aggregation for each type
        self.type_encoders = nn.ModuleDict({
            'diagnosis': nn.Linear(embedding_dim, embedding_dim),
            'lab': nn.Linear(embedding_dim, embedding_dim),
            'vital': nn.Linear(embedding_dim, embedding_dim),
            'procedure': nn.Linear(embedding_dim, embedding_dim),
            'medication': nn.Linear(embedding_dim, embedding_dim),
        })
        
        # Combine types
        self.combiner = nn.Linear(5 * embedding_dim, embedding_dim)
    
    def forward(self, visit_embeddings_by_type):
        """
        Args:
            visit_embeddings_by_type: Dict mapping type to embeddings
                {'diagnosis': [batch, n_diag, embed_dim], ...}
        
        Returns:
            visit_vector: [batch, embed_dim]
        """
        type_vectors = []
        
        for code_type, embeddings in visit_embeddings_by_type.items():
            # Aggregate codes of this type
            type_mean = embeddings.mean(dim=1)  # [batch, embed_dim]
            
            # Type-specific transformation
            type_vector = self.type_encoders[code_type](type_mean)
            type_vectors.append(type_vector)
        
        # Concatenate all type vectors
        combined = torch.cat(type_vectors, dim=-1)  # [batch, 5*embed_dim]
        
        # Final aggregation
        visit_vector = self.combiner(combined)  # [batch, embed_dim]
        
        return visit_vector
```

---

### Recommendation

**Based on current state:**

1. **Short-term:** Keep current set-based approach
   - Simple, robust, widely used
   - Works well in practice
   - Disable `use_semantic_order` to simplify

2. **Experiment:** Test Option 3 (Type Embeddings)
   - Minimal code change
   - Uses semantic information without requiring order
   - Can still use mean/attention aggregation

3. **If needed:** Try Option 2 (Sequential Encoding)
   - Use if empirical results show benefit
   - GRU simpler than Transformer
   - But adds complexity and parameters

**Experimental Protocol:**
```python
# Baseline: Current model (set-based)
model_baseline = LSTMBaseline(aggregation='mean')

# Variant 1: Add type embeddings
model_type_aware = LSTMBaseline(visit_encoder=TypeAwareVisitEncoder())

# Variant 2: Sequential encoding
model_sequential = LSTMBaseline(visit_encoder=SequentialVisitEncoder())

# Compare on held-out test set
# If Variant 1 or 2 significantly outperforms baseline, adopt it
```

---

## Summary

### Current Pipeline

| Component | Structure Handling | Order Preserved? |
|-----------|-------------------|------------------|
| **VisitGrouper** | Groups codes by type | ✅ Yes (in codes_by_type) |
| **Visit.get_ordered_codes()** | Returns semantic order | ✅ Yes |
| **PatientSequenceBuilder** | Encodes in order | ✅ Yes |
| **LSTMBaseline.embedding** | Embeds in order | ✅ Yes |
| **VisitEncoder (mean/attention)** | Aggregates (permutation-invariant) | ❌ No (discarded) |
| **LSTM** | Processes visit vectors | ❌ No (never sees order) |

---

### Key Insights

1. **Data pipeline preserves structure**, but **model discards it**
   - Semantic ordering is computed but not used
   - This is intentional (set-based design)

2. **Set-based aggregation is defensible**
   - Robust to data quality issues
   - Clinically valid (many codes occur simultaneously)
   - Computationally efficient

3. **Opportunities for improvement**
   - Can add type embeddings (minimal change)
   - Can use sequential encoding (bigger change)
   - Should experiment to see if structure helps

4. **Current code can be simplified**
   - If order not used, set `use_semantic_order=False`
   - Remove semantic ordering logic from data pipeline
   - Document that visits are treated as sets

---

### Recommended Actions

**Immediate:**
1. Document current behavior (visits as sets)
2. Add flag to disable semantic ordering (simplify pipeline)
3. Add unit tests to verify permutation invariance

**Experimental:**
4. Implement type-aware encoder (Option 3)
5. Run ablation study: baseline vs. type-aware
6. If beneficial, adopt type-aware approach

**Future:**
7. Consider sequential encoding for complex datasets
8. Explore hierarchical models (type → visit → patient)
9. Add interpretability tools to understand what model learns

The current design is reasonable, but there's potential to better leverage within-visit structure if empirical results support it.
