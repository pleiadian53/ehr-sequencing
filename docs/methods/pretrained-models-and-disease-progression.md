# Pre-trained Models & Disease Progression Modeling

**Date:** January 19, 2026  
**Focus:** Using pre-trained foundation models + Visit-grouped sequences for disease staging

---

## Part 1: Pre-trained Foundation Models for EHR

### Available Pre-trained Models (2024-2026)

You're absolutely right - **don't train from scratch**. Use these pre-trained models:

#### 1. BEHRT (BERT for EHR)

**Paper:** Li et al., "BEHRT: Transformer for Electronic Health Records" (2020)  
**Pre-trained on:** MIMIC-III (40K+ patients)  
**Available:** [GitHub](https://github.com/deepmedicine/BEHRT)

```python
# Load pre-trained BEHRT
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bvanaken/clinical-bert")
tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-bert")

# Fine-tune for your task
# (Much cheaper than training from scratch)
```

**Architecture:**
- Vocabulary: ~30K medical codes (ICD, procedures)
- Embedding: 768-dim
- Layers: 12 transformer layers
- Pre-training: MLM + Next Visit Prediction

**Pros:**
- ✅ Pre-trained on real EHR data
- ✅ Handles temporal sequences
- ✅ Age + visit + position embeddings built-in

**Cons:**
- ⚠️ Limited to codes in MIMIC-III
- ⚠️ No LOINC codes (mostly ICD + procedures)

---

#### 2. Med-BERT

**Paper:** Rasmy et al., "Med-BERT: Pre-trained Contextualized Embeddings" (2021)  
**Pre-trained on:** 28M patients from Cerner Health Facts  
**Available:** [GitHub](https://github.com/ZhiGroup/Med-BERT)

```python
from medbert import MedBERT

# Load pre-trained
model = MedBERT.from_pretrained('medbert-base')

# Fine-tune on your CKD cohort
model.fine_tune(ckd_sequences, task='disease_progression')
```

**Architecture:**
- Vocabulary: 50K+ codes (ICD-9/10, NDC, CPT)
- Embedding: 256-dim
- Layers: 6 transformer layers
- Pre-training: Prolonged Length of Stay prediction

**Pros:**
- ✅ Largest pre-training dataset
- ✅ Includes medication codes (NDC)
- ✅ Proven on disease progression tasks

**Cons:**
- ⚠️ Still limited LOINC coverage

---

#### 3. CEHR-BERT

**Paper:** Pang et al., "CEHR-BERT: Incorporating Temporal Information" (2021)  
**Pre-trained on:** Columbia University Medical Center (4M+ patients)  
**Available:** [GitHub](https://github.com/cumc-dbmi/cehr-bert)

```python
from cehrbert import CEHRBERT

# Load pre-trained
model = CEHRBERT.from_pretrained('cehrbert-base')

# Supports continuous time encoding
embeddings = model.encode(
    codes=patient_codes,
    timestamps=patient_timestamps  # ← Key feature
)
```

**Architecture:**
- Vocabulary: 40K+ codes
- Embedding: 128-dim
- Layers: 4 transformer layers
- **Key feature:** Continuous time encoding (not just position)

**Pros:**
- ✅ Best temporal modeling
- ✅ Handles irregular time intervals
- ✅ Designed for longitudinal prediction

**Cons:**
- ⚠️ Smaller model (fewer layers)

---

#### 4. ClinicalBERT (Text + Codes)

**Paper:** Alsentzer et al., "Publicly Available Clinical BERT Embeddings" (2019)  
**Pre-trained on:** MIMIC-III clinical notes + codes  
**Available:** HuggingFace `emilyalsentzer/Bio_ClinicalBERT`

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Use for clinical notes + codes
text_embedding = model(**tokenizer("Patient has CKD stage 3", return_tensors="pt"))
```

**Pros:**
- ✅ Handles both text and codes
- ✅ Easy to use (HuggingFace)
- ✅ Well-documented

**Cons:**
- ⚠️ Primarily for text, not optimized for code sequences

---

### Recommendation for Your Use Case

**For CKD Disease Staging:**

**Primary: CEHR-BERT** (best temporal modeling)  
**Backup: Med-BERT** (largest pre-training, proven on progression)

**Hybrid Approach (Recommended):**
```python
# Use pre-trained for code embeddings
pretrained_model = CEHRBERT.from_pretrained('cehrbert-base')

# Extract code embeddings
code_embeddings = pretrained_model.get_code_embeddings()

# Use these as initialization for your visit-grouped model
visit_model = VisitGroupedProgressionModel(
    code_embeddings=code_embeddings,  # ← Pre-trained
    visit_encoder='lstm',
    progression_head='survival'
)
```

---

## Part 2: Visit-Grouped Sequences for Disease Progression

### Why Visit-Grouped is Ideal for Disease Staging

You're absolutely right! Visit-grouped sequences are **perfect** for disease progression because:

1. **Clinical Reality:** Disease staging happens at visits (e.g., CKD diagnosed at clinic visit)
2. **Natural Granularity:** Each visit = snapshot of patient state
3. **Temporal Structure:** Visit intervals encode disease velocity
4. **Interpretability:** Can explain "at Visit 5, patient progressed due to..."

### Architecture: Hierarchical Visit-Grouped Model

```python
import torch
import torch.nn as nn

class VisitGroupedProgressionModel(nn.Module):
    """
    Two-level hierarchy:
    1. Code-level: Embed codes within each visit
    2. Visit-level: Model sequence of visits for progression
    """
    def __init__(
        self,
        pretrained_code_embeddings,  # From CEHR-BERT
        code_embed_dim=128,
        visit_embed_dim=256,
        hidden_dim=512,
        num_stages=5,  # CKD stages 1-5
        dropout=0.1
    ):
        super().__init__()
        
        # Level 1: Code embeddings (pre-trained)
        self.code_embeddings = nn.Embedding.from_pretrained(
            pretrained_code_embeddings,
            freeze=False  # Allow fine-tuning
        )
        
        # Level 1: Within-visit aggregation
        self.visit_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=code_embed_dim,
                nhead=4,
                dim_feedforward=code_embed_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Project to visit embedding
        self.visit_projection = nn.Linear(code_embed_dim, visit_embed_dim)
        
        # Level 2: Visit sequence modeling
        self.visit_lstm = nn.LSTM(
            input_size=visit_embed_dim + 2,  # +2 for time features
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=False  # Causal for prediction
        )
        
        # Progression prediction head
        self.stage_classifier = nn.Linear(hidden_dim, num_stages)
        self.time_to_progression = nn.Linear(hidden_dim, 1)  # Days until next stage
        
        self.dropout = nn.Dropout(dropout)
    
    def encode_visit(self, visit_codes, visit_mask):
        """
        Encode a single visit into a fixed-size embedding.
        
        Args:
            visit_codes: [batch, max_codes_per_visit]
            visit_mask: [batch, max_codes_per_visit] - 1 for real codes, 0 for padding
        
        Returns:
            visit_embedding: [batch, visit_embed_dim]
        """
        # Embed codes
        code_embeds = self.code_embeddings(visit_codes)  # [batch, codes, code_dim]
        
        # Aggregate codes within visit using Transformer
        # Mask padding tokens
        attn_mask = ~visit_mask.bool()  # True = ignore
        visit_repr = self.visit_encoder(
            code_embeds,
            src_key_padding_mask=attn_mask
        )  # [batch, codes, code_dim]
        
        # Pool to single visit embedding (mean over non-padding)
        visit_mask_expanded = visit_mask.unsqueeze(-1)  # [batch, codes, 1]
        masked_repr = visit_repr * visit_mask_expanded
        visit_embed = masked_repr.sum(dim=1) / visit_mask.sum(dim=1, keepdim=True)
        
        # Project to visit space
        return self.visit_projection(visit_embed)  # [batch, visit_dim]
    
    def forward(self, patient_visits, time_features, visit_mask):
        """
        Predict disease progression from visit sequence.
        
        Args:
            patient_visits: [batch, num_visits, max_codes_per_visit]
            time_features: [batch, num_visits, 2] - (days_since_first, days_since_prev)
            visit_mask: [batch, num_visits, max_codes_per_visit]
        
        Returns:
            stage_logits: [batch, num_visits, num_stages]
            time_to_progression: [batch, num_visits, 1]
        """
        batch_size, num_visits, max_codes = patient_visits.shape
        
        # Encode each visit
        visit_embeds = []
        for i in range(num_visits):
            visit_embed = self.encode_visit(
                patient_visits[:, i, :],
                visit_mask[:, i, :]
            )  # [batch, visit_dim]
            visit_embeds.append(visit_embed)
        
        visit_embeds = torch.stack(visit_embeds, dim=1)  # [batch, visits, visit_dim]
        
        # Concatenate time features
        visit_embeds_with_time = torch.cat([
            visit_embeds,
            time_features
        ], dim=-1)  # [batch, visits, visit_dim + 2]
        
        # Model visit sequence
        lstm_out, _ = self.visit_lstm(visit_embeds_with_time)  # [batch, visits, hidden]
        lstm_out = self.dropout(lstm_out)
        
        # Predict stage at each visit
        stage_logits = self.stage_classifier(lstm_out)  # [batch, visits, num_stages]
        
        # Predict time to next stage
        time_pred = self.time_to_progression(lstm_out)  # [batch, visits, 1]
        time_pred = torch.relu(time_pred)  # Ensure positive
        
        return stage_logits, time_pred
```

---

## Part 3: CKD Disease Staging - Complete Example

### Data Preparation

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class CKDSequenceBuilder:
    """
    Build visit-grouped sequences for CKD patients.
    """
    def __init__(self, vocab):
        self.vocab = vocab
        self.ckd_codes = {
            'CKD1': ['N18.1'],  # Stage 1
            'CKD2': ['N18.2'],  # Stage 2
            'CKD3': ['N18.3', 'N18.30', 'N18.31', 'N18.32'],  # Stage 3
            'CKD4': ['N18.4'],  # Stage 4
            'CKD5': ['N18.5', 'N18.6'],  # Stage 5 / ESRD
        }
    
    def extract_ckd_stage(self, codes):
        """Extract CKD stage from ICD codes in a visit."""
        for stage, stage_codes in self.ckd_codes.items():
            if any(code in codes for code in stage_codes):
                return int(stage[-1])  # Return stage number
        return None
    
    def build_patient_sequence(self, patient_df):
        """
        Build visit-grouped sequence for one patient.
        
        Args:
            patient_df: DataFrame with columns [timestamp, code, code_type, value]
        
        Returns:
            sequence: {
                'visits': List of visit dicts,
                'stages': List of CKD stages at each visit,
                'progression_events': List of (visit_idx, old_stage, new_stage)
            }
        """
        # Group by visit (same day = same visit)
        patient_df['visit_date'] = pd.to_datetime(patient_df['timestamp']).dt.date
        visits = []
        stages = []
        
        for visit_date, visit_df in patient_df.groupby('visit_date'):
            # Extract codes
            codes = visit_df['code'].tolist()
            code_ids = [self.vocab.get(code, self.vocab['[UNK]']) for code in codes]
            
            # Extract CKD stage
            stage = self.extract_ckd_stage(codes)
            
            # Time features
            if len(visits) == 0:
                days_since_first = 0
                days_since_prev = 0
            else:
                first_date = visits[0]['date']
                prev_date = visits[-1]['date']
                days_since_first = (visit_date - first_date).days
                days_since_prev = (visit_date - prev_date).days
            
            visits.append({
                'date': visit_date,
                'codes': code_ids,
                'days_since_first': days_since_first,
                'days_since_prev': days_since_prev
            })
            stages.append(stage)
        
        # Identify progression events
        progression_events = []
        for i in range(1, len(stages)):
            if stages[i] is not None and stages[i-1] is not None:
                if stages[i] > stages[i-1]:
                    progression_events.append((i, stages[i-1], stages[i]))
        
        return {
            'visits': visits,
            'stages': stages,
            'progression_events': progression_events
        }
    
    def build_dataset(self, all_patients_df):
        """Build dataset for all patients."""
        sequences = []
        for patient_id, patient_df in all_patients_df.groupby('patient_id'):
            seq = self.build_patient_sequence(patient_df)
            if len(seq['visits']) >= 3:  # Minimum 3 visits
                sequences.append({
                    'patient_id': patient_id,
                    **seq
                })
        return sequences
```

### Training

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CKDProgressionDataset(Dataset):
    """PyTorch dataset for CKD progression."""
    def __init__(self, sequences, max_visits=20, max_codes_per_visit=50):
        self.sequences = sequences
        self.max_visits = max_visits
        self.max_codes = max_codes_per_visit
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Pad/truncate visits
        num_visits = min(len(seq['visits']), self.max_visits)
        
        # Initialize tensors
        visit_codes = torch.zeros(self.max_visits, self.max_codes, dtype=torch.long)
        visit_mask = torch.zeros(self.max_visits, self.max_codes, dtype=torch.float)
        time_features = torch.zeros(self.max_visits, 2, dtype=torch.float)
        stage_labels = torch.full((self.max_visits,), -1, dtype=torch.long)  # -1 = no label
        
        for i in range(num_visits):
            visit = seq['visits'][i]
            codes = visit['codes'][:self.max_codes]
            
            visit_codes[i, :len(codes)] = torch.tensor(codes)
            visit_mask[i, :len(codes)] = 1.0
            time_features[i, 0] = visit['days_since_first'] / 365.0  # Normalize to years
            time_features[i, 1] = visit['days_since_prev'] / 30.0  # Normalize to months
            
            if seq['stages'][i] is not None:
                stage_labels[i] = seq['stages'][i] - 1  # 0-indexed
        
        return {
            'visit_codes': visit_codes,
            'visit_mask': visit_mask,
            'time_features': time_features,
            'stage_labels': stage_labels,
            'num_visits': num_visits
        }

# Training loop
def train_ckd_model(model, train_loader, val_loader, num_epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    stage_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    time_criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Forward pass
            stage_logits, time_pred = model(
                batch['visit_codes'],
                batch['time_features'],
                batch['visit_mask']
            )
            
            # Stage classification loss
            stage_loss = stage_criterion(
                stage_logits.view(-1, stage_logits.size(-1)),
                batch['stage_labels'].view(-1)
            )
            
            # Time to progression loss (only for progression events)
            # TODO: Compute actual time to next stage from data
            
            loss = stage_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        if epoch % 5 == 0:
            val_metrics = evaluate_ckd_model(model, val_loader)
            print(f"Epoch {epoch}: Train Loss={total_loss/len(train_loader):.4f}, "
                  f"Val AUC={val_metrics['auc']:.4f}")
```

### Evaluation

```python
from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate_ckd_model(model, val_loader):
    """Evaluate CKD progression model."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            stage_logits, time_pred = model(
                batch['visit_codes'],
                batch['time_features'],
                batch['visit_mask']
            )
            
            # Get predictions
            probs = torch.softmax(stage_logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            # Filter valid labels
            valid_mask = batch['stage_labels'] != -1
            
            all_preds.extend(preds[valid_mask].cpu().numpy())
            all_labels.extend(batch['stage_labels'][valid_mask].cpu().numpy())
            all_probs.extend(probs[valid_mask].cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Multi-class AUC (one-vs-rest)
    all_probs = np.array(all_probs)
    all_labels_onehot = np.eye(5)[all_labels]
    auc = roc_auc_score(all_labels_onehot, all_probs, average='macro', multi_class='ovr')
    
    return {
        'accuracy': accuracy,
        'auc': auc
    }
```

---

## Part 4: Why Visit-Grouped is Superior for Disease Progression

### Comparison with Other Approaches

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Flat Sequence** | Simple, captures all codes | No visit structure, hard to interpret | General embeddings |
| **Visit-Grouped** | ✅ Clinical reality, ✅ Interpretable, ✅ Natural granularity | More complex | **Disease progression** |
| **Hierarchical** | Respects code types | Loses temporal ordering within visit | Multi-modal analysis |
| **Time-binned** | Fixed intervals | Artificial boundaries | Population studies |

### Key Advantages for CKD Staging

1. **Clinical Alignment:** CKD stages are assessed at clinic visits
2. **Interpretability:** "Patient progressed at Visit 5 due to elevated creatinine + proteinuria"
3. **Temporal Modeling:** Visit intervals encode disease velocity (rapid vs slow progression)
4. **Prediction Target:** "Will patient progress to next stage by next visit?"

### Visit Embeddings Capture Disease State

```python
# Visit embedding captures:
visit_embedding = f(
    diagnosis_codes,      # CKD stage, comorbidities
    lab_values,           # Creatinine, GFR, proteinuria
    medications,          # ACE inhibitors, diuretics
    procedures,           # Dialysis, transplant
    time_since_last_visit # Disease velocity
)

# Sequence of visits = disease trajectory
trajectory = [visit1_embed, visit2_embed, ..., visitN_embed]

# Predict next state
next_stage = progression_model(trajectory)
```

---

## Part 5: Practical Implementation Plan

### Week 1: Setup with Pre-trained Model

```bash
# Install pre-trained model
pip install cehr-bert  # or med-bert

# Load pre-trained embeddings
python scripts/load_pretrained_embeddings.py
```

### Week 2: Build Visit-Grouped Sequences

```python
# scripts/build_ckd_sequences.py
from ehrsequencing.data import CKDSequenceBuilder

builder = CKDSequenceBuilder(vocab)
sequences = builder.build_dataset(ckd_patients_df)

# Save
torch.save(sequences, 'data/processed/ckd_sequences.pt')
```

### Week 3: Train Progression Model

```python
# examples/train_ckd_progression.py
from ehrsequencing.models import VisitGroupedProgressionModel

# Load pre-trained embeddings
pretrained = load_cehrbert_embeddings()

# Initialize model
model = VisitGroupedProgressionModel(
    pretrained_code_embeddings=pretrained,
    num_stages=5
)

# Train
train_ckd_model(model, train_loader, val_loader)
```

### Week 4: Evaluate & Interpret

```python
# Evaluate
metrics = evaluate_ckd_model(model, test_loader)

# Interpret: Which codes drive progression?
attention_weights = model.get_visit_attention(patient_sequence)
important_codes = get_top_codes_by_attention(attention_weights)
```

---

## Part 6: Advanced: Survival Analysis for Time-to-Progression

```python
from lifelines import CoxPHFitter

class SurvivalProgressionModel(VisitGroupedProgressionModel):
    """
    Combine visit embeddings with survival analysis.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Cox proportional hazards head
        self.cox_head = nn.Linear(kwargs['hidden_dim'], 1)
    
    def predict_time_to_progression(self, patient_visits, time_features, visit_mask):
        """
        Predict time until progression to next stage.
        
        Returns:
            hazard: Risk score (higher = faster progression)
        """
        # Get visit sequence representation
        _, lstm_out = self.forward(patient_visits, time_features, visit_mask)
        
        # Use last visit representation
        last_visit_repr = lstm_out[:, -1, :]  # [batch, hidden]
        
        # Predict hazard
        hazard = self.cox_head(last_visit_repr)  # [batch, 1]
        
        return hazard
```

---

## Summary & Recommendations

### For Your CKD Disease Staging Task

**Recommended Architecture:**

1. **Code Embeddings:** CEHR-BERT pre-trained (don't train from scratch)
2. **Sequence Representation:** Visit-grouped (your intuition is correct!)
3. **Visit Encoder:** Transformer (aggregate codes within visit)
4. **Sequence Model:** LSTM or Transformer (model visit sequence)
5. **Prediction Head:** Multi-task (stage classification + time-to-progression)

**Why This Works:**

- ✅ **Pre-trained embeddings** → Captures medical knowledge without expensive training
- ✅ **Visit-grouped** → Aligns with clinical reality and disease assessment
- ✅ **Hierarchical** → Captures both within-visit patterns and across-visit progression
- ✅ **Interpretable** → Can explain which visits/codes drive progression

**Expected Performance:**

- Stage classification AUC: 0.85-0.90
- Time-to-progression C-index: 0.75-0.80
- Training time: Days (not weeks/months)
- Inference: <100ms per patient

---

## Next Steps

1. **Choose pre-trained model:** CEHR-BERT or Med-BERT
2. **Build visit-grouped sequences** for the target cohort
3. **Fine-tune** on the data (much cheaper than training from scratch)
4. **Evaluate** on disease progression metrics
5. **Interpret** attention weights for clinical insights

The visit-grouped approach is **ideal** for disease progression modeling.

---

**Document Version:** 1.0  
**Last Updated:** January 19, 2026
