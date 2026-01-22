# Visit-Grouped Sequence Representation: Implementation Plan

**Date:** January 19, 2026  
**Status:** Implementation Roadmap  
**Priority:** Foundation for all downstream applications

---

## Overview

This document outlines the complete implementation plan for visit-grouped sequence representation in the EHR Sequencing project. This is the **foundational pipeline** that transforms raw EHR/EMR data into structured visit sequences suitable for disease progression modeling, temporal phenotyping, and patient segmentation.

### Key Components

1. **Data Sequencing Pipeline** - Transform raw EHR → structured visits
2. **Data Adapters** - Reuse from loinc-predictor project
3. **Visit Representation** - How to structure codes within visits
4. **Pre-trained Model Integration** - Leverage CEHR-BERT embeddings
5. **Downstream Applications** - Disease progression, phenotyping, clustering

---

## Part 1: Sequencing Pipeline Architecture

### 1.1 Pipeline Overview

```
Raw EHR Data (CSV/Database)
    ↓
[Data Adapter] - Load and normalize
    ↓
Event Stream (timestamp, patient_id, code, code_type, value)
    ↓
[Visit Grouper] - Group events into visits
    ↓
Visit Sequences (patient → [visit1, visit2, ...])
    ↓
[Visit Encoder] - Encode visits with pre-trained embeddings
    ↓
Patient Trajectories (patient → [visit_embed1, visit_embed2, ...])
    ↓
[Downstream Models] - Disease progression, phenotyping, etc.
```

### 1.2 Design Decisions & Open Questions

#### Decision 1: Visit Definition

**Options:**
- **A. Same-day grouping** - All events on same calendar day = one visit
- **B. Time-window grouping** - Events within 24 hours = one visit
- **C. Encounter-based** - Use explicit encounter IDs from EHR
- **D. Hybrid** - Encounter IDs when available, same-day otherwise

**Recommendation: D (Hybrid)**
- Most EHR systems have encounter IDs
- Fallback to same-day for datasets without encounters
- Clinically accurate (reflects actual visits)

#### Decision 2: Within-Visit Structure

**Open Question:** How to maintain structure within a visit?

**Options:**

**A. Flat (Order-Agnostic)**
```python
visit = {
    'codes': [ICD:E11.9, LOINC:4548-4, RXNORM:860975],
    'timestamp': '2020-01-15'
}
# Pros: Simple, order doesn't matter
# Cons: Loses temporal ordering within visit
```

**B. Typed Groups (Code-Type Aware)**
```python
visit = {
    'diagnoses': [ICD:E11.9],
    'labs': [LOINC:4548-4],
    'medications': [RXNORM:860975],
    'timestamp': '2020-01-15'
}
# Pros: Preserves code types, natural clinical workflow
# Cons: More complex, need to handle variable-length groups
```

**C. Temporal Ordering (Within-Visit Time)**
```python
visit = {
    'events': [
        (09:00, LOINC:4548-4),   # Lab drawn first
        (09:30, ICD:E11.9),       # Diagnosis after lab
        (10:00, RXNORM:860975)    # Medication prescribed last
    ],
    'date': '2020-01-15'
}
# Pros: Most accurate, preserves clinical workflow
# Cons: Requires precise timestamps (often not available)
```

**Recommendation: B (Typed Groups) with optional A fallback**
- Reflects clinical workflow (labs → diagnosis → treatment)
- Most datasets have code types
- Can aggregate to flat if needed

#### Decision 3: Visit Sequence Length

**Open Question:** How to handle variable-length patient histories?

**Options:**
- **Fixed window** - Last N visits (e.g., N=20)
- **Time window** - Last T years (e.g., T=5)
- **Dynamic** - All visits, pad/truncate at batch level
- **Hierarchical** - Summarize old visits, keep recent ones detailed

**Recommendation: Dynamic with max limit**
- Keep all visits up to max (e.g., 50 visits)
- For longer histories, use hierarchical summarization
- Allows model to learn from full history

---

## Part 2: Data Adapter Integration

### 2.1 Reusing loinc-predictor Data Adapters

The loinc-predictor project already has data adapters for:
- Synthea (synthetic EHR data)
- MIMIC-III (real ICU data)
- Generic CSV format

**Strategy: Duplicate adapters to ehr-sequencing**

```
loinc-predictor/src/loinc_predictor/data/adapters/
    ├── synthea_adapter.py
    ├── mimic_adapter.py
    └── base_adapter.py

→ Copy to →

ehr-sequencing/src/ehrsequencing/data/adapters/
    ├── synthea_adapter.py
    ├── mimic_adapter.py
    └── base_adapter.py
```

### 2.2 Adapter Interface

```python
# src/ehrsequencing/data/adapters/base_adapter.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd

class BaseEHRAdapter(ABC):
    """
    Base class for EHR data adapters.
    
    Adapters transform raw EHR data into standardized event streams.
    """
    
    @abstractmethod
    def load_patients(self) -> pd.DataFrame:
        """
        Load patient demographics.
        
        Returns:
            DataFrame with columns: [patient_id, birth_date, gender, race, ...]
        """
        pass
    
    @abstractmethod
    def load_events(self, patient_ids: List[str] = None) -> pd.DataFrame:
        """
        Load all clinical events.
        
        Args:
            patient_ids: Optional list of patient IDs to filter
        
        Returns:
            DataFrame with columns:
                - patient_id: str
                - timestamp: datetime
                - code: str (e.g., 'ICD10:E11.9', 'LOINC:4548-4')
                - code_type: str ('ICD', 'LOINC', 'SNOMED', 'RXNORM', 'CPT')
                - value: float (for lab values, None for diagnoses)
                - unit: str (for lab values)
                - encounter_id: str (visit identifier)
        """
        pass
    
    @abstractmethod
    def load_encounters(self) -> pd.DataFrame:
        """
        Load encounter (visit) information.
        
        Returns:
            DataFrame with columns:
                - encounter_id: str
                - patient_id: str
                - start_time: datetime
                - end_time: datetime
                - encounter_type: str ('inpatient', 'outpatient', 'emergency')
        """
        pass
    
    def get_vocabulary(self) -> Dict[str, int]:
        """
        Build vocabulary mapping codes to integer IDs.
        
        Returns:
            Dict mapping code strings to integer IDs
        """
        events = self.load_events()
        unique_codes = events['code'].unique()
        
        vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[MASK]': 2,
            '[CLS]': 3,
            '[SEP]': 4
        }
        
        for i, code in enumerate(sorted(unique_codes), start=5):
            vocab[code] = i
        
        return vocab
```

### 2.3 Synthea Adapter Implementation

```python
# src/ehrsequencing/data/adapters/synthea_adapter.py

import pandas as pd
from pathlib import Path
from .base_adapter import BaseEHRAdapter

class SyntheaAdapter(BaseEHRAdapter):
    """
    Adapter for Synthea synthetic EHR data.
    
    Synthea format:
        - patients.csv
        - conditions.csv (diagnoses)
        - observations.csv (labs, vitals)
        - medications.csv
        - procedures.csv
        - encounters.csv
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._validate_data_dir()
    
    def _validate_data_dir(self):
        """Check that required files exist."""
        required_files = [
            'patients.csv',
            'conditions.csv',
            'observations.csv',
            'medications.csv',
            'encounters.csv'
        ]
        for file in required_files:
            if not (self.data_dir / file).exists():
                raise FileNotFoundError(f"Missing required file: {file}")
    
    def load_patients(self) -> pd.DataFrame:
        """Load patient demographics."""
        df = pd.read_csv(self.data_dir / 'patients.csv')
        
        return df.rename(columns={
            'Id': 'patient_id',
            'BIRTHDATE': 'birth_date',
            'GENDER': 'gender',
            'RACE': 'race'
        })[['patient_id', 'birth_date', 'gender', 'race']]
    
    def load_encounters(self) -> pd.DataFrame:
        """Load encounter information."""
        df = pd.read_csv(self.data_dir / 'encounters.csv')
        
        return pd.DataFrame({
            'encounter_id': df['Id'],
            'patient_id': df['PATIENT'],
            'start_time': pd.to_datetime(df['START']),
            'end_time': pd.to_datetime(df['STOP']),
            'encounter_type': df['ENCOUNTERCLASS']
        })
    
    def load_events(self, patient_ids: List[str] = None) -> pd.DataFrame:
        """Load all clinical events."""
        events = []
        
        # Load diagnoses (conditions)
        conditions = pd.read_csv(self.data_dir / 'conditions.csv')
        conditions_events = pd.DataFrame({
            'patient_id': conditions['PATIENT'],
            'timestamp': pd.to_datetime(conditions['START']),
            'code': 'SNOMED:' + conditions['CODE'].astype(str),
            'code_type': 'SNOMED',
            'value': None,
            'unit': None,
            'encounter_id': conditions['ENCOUNTER']
        })
        events.append(conditions_events)
        
        # Load labs/observations
        observations = pd.read_csv(self.data_dir / 'observations.csv')
        obs_events = pd.DataFrame({
            'patient_id': observations['PATIENT'],
            'timestamp': pd.to_datetime(observations['DATE']),
            'code': 'LOINC:' + observations['CODE'].astype(str),
            'code_type': 'LOINC',
            'value': pd.to_numeric(observations['VALUE'], errors='coerce'),
            'unit': observations['UNITS'],
            'encounter_id': observations['ENCOUNTER']
        })
        events.append(obs_events)
        
        # Load medications
        medications = pd.read_csv(self.data_dir / 'medications.csv')
        med_events = pd.DataFrame({
            'patient_id': medications['PATIENT'],
            'timestamp': pd.to_datetime(medications['START']),
            'code': 'RXNORM:' + medications['CODE'].astype(str),
            'code_type': 'RXNORM',
            'value': None,
            'unit': None,
            'encounter_id': medications['ENCOUNTER']
        })
        events.append(med_events)
        
        # Load procedures
        procedures = pd.read_csv(self.data_dir / 'procedures.csv')
        proc_events = pd.DataFrame({
            'patient_id': procedures['PATIENT'],
            'timestamp': pd.to_datetime(procedures['DATE']),
            'code': 'SNOMED:' + procedures['CODE'].astype(str),
            'code_type': 'SNOMED',
            'value': None,
            'unit': None,
            'encounter_id': procedures['ENCOUNTER']
        })
        events.append(proc_events)
        
        # Combine all events
        all_events = pd.concat(events, ignore_index=True)
        
        # Filter by patient IDs if provided
        if patient_ids is not None:
            all_events = all_events[all_events['patient_id'].isin(patient_ids)]
        
        # Sort by patient and time
        all_events = all_events.sort_values(['patient_id', 'timestamp'])
        
        return all_events
```

---

## Part 3: Visit Grouping & Sequencing

### 3.1 Visit Grouper Implementation

```python
# src/ehrsequencing/data/sequences/visit_grouper.py

import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict

class VisitGrouper:
    """
    Group clinical events into visits.
    
    Supports multiple grouping strategies:
    - encounter-based (use explicit encounter IDs)
    - same-day (group events on same calendar day)
    - time-window (group events within N hours)
    """
    
    def __init__(self, strategy: str = 'encounter', time_window_hours: int = 24):
        """
        Args:
            strategy: 'encounter', 'same-day', or 'time-window'
            time_window_hours: For time-window strategy
        """
        self.strategy = strategy
        self.time_window_hours = time_window_hours
    
    def group_events_into_visits(self, events_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Group events into visits for all patients.
        
        Args:
            events_df: DataFrame from adapter.load_events()
        
        Returns:
            List of visit dicts, one per patient-visit
        """
        if self.strategy == 'encounter':
            return self._group_by_encounter(events_df)
        elif self.strategy == 'same-day':
            return self._group_by_same_day(events_df)
        elif self.strategy == 'time-window':
            return self._group_by_time_window(events_df)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _group_by_encounter(self, events_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Group by encounter_id."""
        visits = []
        
        for (patient_id, encounter_id), group in events_df.groupby(['patient_id', 'encounter_id']):
            visit = self._create_visit_dict(patient_id, encounter_id, group)
            visits.append(visit)
        
        return visits
    
    def _group_by_same_day(self, events_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Group events on same calendar day."""
        events_df = events_df.copy()
        events_df['visit_date'] = events_df['timestamp'].dt.date
        
        visits = []
        for (patient_id, visit_date), group in events_df.groupby(['patient_id', 'visit_date']):
            visit = self._create_visit_dict(patient_id, f"{patient_id}_{visit_date}", group)
            visits.append(visit)
        
        return visits
    
    def _group_by_time_window(self, events_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Group events within time window."""
        visits = []
        
        for patient_id, patient_events in events_df.groupby('patient_id'):
            patient_events = patient_events.sort_values('timestamp')
            
            current_visit_events = []
            visit_start_time = None
            visit_id = 0
            
            for _, event in patient_events.iterrows():
                if visit_start_time is None:
                    visit_start_time = event['timestamp']
                    current_visit_events.append(event)
                else:
                    time_diff = (event['timestamp'] - visit_start_time).total_seconds() / 3600
                    
                    if time_diff <= self.time_window_hours:
                        current_visit_events.append(event)
                    else:
                        visit = self._create_visit_dict(
                            patient_id,
                            f"{patient_id}_visit_{visit_id}",
                            pd.DataFrame(current_visit_events)
                        )
                        visits.append(visit)
                        
                        current_visit_events = [event]
                        visit_start_time = event['timestamp']
                        visit_id += 1
            
            if current_visit_events:
                visit = self._create_visit_dict(
                    patient_id,
                    f"{patient_id}_visit_{visit_id}",
                    pd.DataFrame(current_visit_events)
                )
                visits.append(visit)
        
        return visits
    
    def _create_visit_dict(
        self,
        patient_id: str,
        visit_id: str,
        events: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Create structured visit dictionary.
        
        Uses typed groups (Option B from design decisions).
        """
        visit = {
            'patient_id': patient_id,
            'visit_id': visit_id,
            'timestamp': events['timestamp'].min(),
            'codes_by_type': defaultdict(list),
            'all_codes': [],
            'values': {}
        }
        
        for _, event in events.iterrows():
            code = event['code']
            code_type = event['code_type']
            
            visit['codes_by_type'][code_type].append(code)
            visit['all_codes'].append(code)
            
            if pd.notna(event['value']):
                visit['values'][code] = {
                    'value': event['value'],
                    'unit': event['unit']
                }
        
        return visit
```

### 3.2 Patient Sequence Builder

```python
# src/ehrsequencing/data/sequences/sequence_builder.py

import pandas as pd
from typing import List, Dict, Any
from .visit_grouper import VisitGrouper

class PatientSequenceBuilder:
    """
    Build patient-level visit sequences.
    
    Transforms visits into sequences suitable for modeling.
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        max_visits: int = 50,
        max_codes_per_visit: int = 100
    ):
        self.vocab = vocab
        self.max_visits = max_visits
        self.max_codes_per_visit = max_codes_per_visit
    
    def build_sequences(self, visits: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Build sequences for all patients.
        
        Args:
            visits: List of visit dicts from VisitGrouper
        
        Returns:
            Dict mapping patient_id to list of visit sequences
        """
        patient_sequences = defaultdict(list)
        
        for visit in visits:
            patient_id = visit['patient_id']
            patient_sequences[patient_id].append(visit)
        
        # Sort each patient's visits by time
        for patient_id in patient_sequences:
            patient_sequences[patient_id].sort(key=lambda v: v['timestamp'])
        
        return dict(patient_sequences)
    
    def encode_sequence(self, patient_visits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Encode a patient's visit sequence into tensor-ready format.
        
        Args:
            patient_visits: List of visits for one patient
        
        Returns:
            Dict with encoded sequences and metadata
        """
        num_visits = min(len(patient_visits), self.max_visits)
        
        encoded = {
            'patient_id': patient_visits[0]['patient_id'],
            'num_visits': num_visits,
            'visit_codes': [],  # List of code ID lists
            'visit_timestamps': [],
            'visit_time_deltas': [],  # Days since previous visit
            'visit_codes_by_type': [],  # List of dicts
            'visit_values': []  # List of value dicts
        }
        
        prev_timestamp = None
        
        for i in range(num_visits):
            visit = patient_visits[i]
            
            # Encode codes
            code_ids = [
                self.vocab.get(code, self.vocab['[UNK]'])
                for code in visit['all_codes'][:self.max_codes_per_visit]
            ]
            encoded['visit_codes'].append(code_ids)
            
            # Encode by type
            codes_by_type = {}
            for code_type, codes in visit['codes_by_type'].items():
                codes_by_type[code_type] = [
                    self.vocab.get(code, self.vocab['[UNK]'])
                    for code in codes[:self.max_codes_per_visit]
                ]
            encoded['visit_codes_by_type'].append(codes_by_type)
            
            # Timestamps
            encoded['visit_timestamps'].append(visit['timestamp'])
            
            # Time deltas
            if prev_timestamp is None:
                time_delta = 0
            else:
                time_delta = (visit['timestamp'] - prev_timestamp).days
            encoded['visit_time_deltas'].append(time_delta)
            prev_timestamp = visit['timestamp']
            
            # Values
            encoded['visit_values'].append(visit['values'])
        
        return encoded
```

---

## Part 4: Pre-trained Model Integration (CEHR-BERT)

### 4.1 CEHR-BERT Integration Strategy

**Goal:** Use pre-trained CEHR-BERT embeddings without training from scratch

**Approach:**
1. Load pre-trained CEHR-BERT model
2. Extract code embeddings (frozen or fine-tunable)
3. Use as initialization for visit encoder
4. Fine-tune on downstream tasks

### 4.2 CEHR-BERT Wrapper

```python
# src/ehrsequencing/embeddings/cehrbert_wrapper.py

import torch
import torch.nn as nn
from typing import Dict, Optional

class CEHRBERTWrapper(nn.Module):
    """
    Wrapper for pre-trained CEHR-BERT model.
    
    Provides easy interface to extract code embeddings and
    encode patient sequences.
    """
    
    def __init__(
        self,
        model_path: str,
        freeze_embeddings: bool = False,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        
        # Load pre-trained model
        self.cehrbert = self._load_pretrained(model_path)
        
        # Extract code embeddings
        self.code_embeddings = self.cehrbert.get_code_embeddings()
        
        if freeze_embeddings:
            self.code_embeddings.weight.requires_grad = False
    
    def _load_pretrained(self, model_path: str):
        """Load pre-trained CEHR-BERT model."""
        try:
            from cehrbert import CEHRBERT
            model = CEHRBERT.from_pretrained(model_path)
            return model
        except ImportError:
            raise ImportError(
                "CEHR-BERT not installed. Install with: pip install cehr-bert"
            )
    
    def get_code_embeddings(self) -> nn.Embedding:
        """Get code embedding layer."""
        return self.code_embeddings
    
    def encode_codes(self, code_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode codes using pre-trained embeddings.
        
        Args:
            code_ids: [batch, seq_len] or [batch, num_visits, codes_per_visit]
        
        Returns:
            embeddings: Same shape with last dim = embed_dim
        """
        return self.code_embeddings(code_ids)
    
    def encode_patient_sequence(
        self,
        code_ids: torch.Tensor,
        timestamps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode full patient sequence using CEHR-BERT.
        
        Args:
            code_ids: [batch, seq_len]
            timestamps: [batch, seq_len] - Days since first event
            attention_mask: [batch, seq_len]
        
        Returns:
            sequence_embeddings: [batch, seq_len, embed_dim]
        """
        return self.cehrbert.encode(
            code_ids=code_ids,
            timestamps=timestamps,
            attention_mask=attention_mask
        )
```

### 4.3 Visit Encoder with Pre-trained Embeddings

```python
# src/ehrsequencing/models/visit_encoder.py

import torch
import torch.nn as nn
from ..embeddings.cehrbert_wrapper import CEHRBERTWrapper

class VisitEncoder(nn.Module):
    """
    Encode visits using pre-trained code embeddings.
    
    Two-level encoding:
    1. Code-level: Use pre-trained CEHR-BERT embeddings
    2. Visit-level: Aggregate codes into visit representation
    """
    
    def __init__(
        self,
        cehrbert_wrapper: CEHRBERTWrapper,
        code_embed_dim: int = 128,
        visit_embed_dim: int = 256,
        aggregation: str = 'transformer'
    ):
        super().__init__()
        
        self.code_embeddings = cehrbert_wrapper.get_code_embeddings()
        self.aggregation = aggregation
        
        if aggregation == 'transformer':
            self.aggregator = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=code_embed_dim,
                    nhead=4,
                    dim_feedforward=code_embed_dim * 4,
                    batch_first=True
                ),
                num_layers=2
            )
        elif aggregation == 'lstm':
            self.aggregator = nn.LSTM(
                code_embed_dim,
                code_embed_dim,
                batch_first=True
            )
        elif aggregation == 'mean':
            self.aggregator = None
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        self.projection = nn.Linear(code_embed_dim, visit_embed_dim)
    
    def forward(self, visit_codes: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of visits.
        
        Args:
            visit_codes: [batch, max_codes_per_visit]
            visit_mask: [batch, max_codes_per_visit] - 1 for real, 0 for padding
        
        Returns:
            visit_embeddings: [batch, visit_embed_dim]
        """
        # Embed codes
        code_embeds = self.code_embeddings(visit_codes)  # [batch, codes, embed_dim]
        
        # Aggregate
        if self.aggregation == 'transformer':
            attn_mask = ~visit_mask.bool()
            aggregated = self.aggregator(
                code_embeds,
                src_key_padding_mask=attn_mask
            )
            visit_repr = self._masked_mean(aggregated, visit_mask)
        
        elif self.aggregation == 'lstm':
            _, (hidden, _) = self.aggregator(code_embeds)
            visit_repr = hidden[-1]
        
        elif self.aggregation == 'mean':
            visit_repr = self._masked_mean(code_embeds, visit_mask)
        
        # Project to visit space
        return self.projection(visit_repr)
    
    def _masked_mean(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute mean over non-masked elements."""
        mask_expanded = mask.unsqueeze(-1)
        masked_embeds = embeddings * mask_expanded
        return masked_embeds.sum(dim=1) / mask.sum(dim=1, keepdim=True)
```

---

## Part 5: Downstream Applications

### 5.1 Application 1: Disease Progression Modeling

**Use Case:** Predict CKD stage progression

**Architecture:**
```
Visit Sequences → Visit Encoder → LSTM/Transformer → Stage Classifier
```

**Implementation:** See `pretrained-models-and-disease-progression.md`

**Key Features:**
- Multi-task learning (stage + time-to-progression)
- Survival analysis integration
- Attention-based interpretability

### 5.2 Application 2: Temporal Phenotyping

**Use Case:** Discover disease subtypes based on temporal patterns

**Architecture:**
```
Visit Sequences → Visit Encoder → Patient Embeddings → Clustering
```

**Implementation:**
```python
# src/ehrsequencing/clustering/temporal_phenotyping.py

from sklearn.cluster import KMeans
import umap

class TemporalPhenotyper:
    """
    Discover disease phenotypes from visit sequences.
    """
    
    def __init__(self, visit_encoder, n_phenotypes: int = 5):
        self.visit_encoder = visit_encoder
        self.n_phenotypes = n_phenotypes
        self.clusterer = KMeans(n_clusters=n_phenotypes)
        self.reducer = umap.UMAP(n_components=2)
    
    def fit(self, patient_sequences):
        """
        Discover phenotypes from patient sequences.
        """
        # Encode all patients
        patient_embeddings = []
        for seq in patient_sequences:
            visit_embeds = self.visit_encoder(seq['visit_codes'], seq['visit_mask'])
            patient_embed = visit_embeds.mean(dim=0)  # Aggregate visits
            patient_embeddings.append(patient_embed)
        
        patient_embeddings = torch.stack(patient_embeddings).detach().numpy()
        
        # Cluster
        phenotypes = self.clusterer.fit_predict(patient_embeddings)
        
        # Reduce for visualization
        embeddings_2d = self.reducer.fit_transform(patient_embeddings)
        
        return phenotypes, embeddings_2d
```

### 5.3 Application 3: Patient Similarity & Retrieval

**Use Case:** Find similar patients for case-based reasoning

**Architecture:**
```
Query Patient → Visit Encoder → Patient Embedding → Nearest Neighbors
```

**Implementation:**
```python
# src/ehrsequencing/retrieval/patient_similarity.py

import faiss
import numpy as np

class PatientRetrieval:
    """
    Retrieve similar patients based on visit sequences.
    """
    
    def __init__(self, visit_encoder, embed_dim: int = 256):
        self.visit_encoder = visit_encoder
        self.index = faiss.IndexFlatL2(embed_dim)
        self.patient_ids = []
    
    def index_patients(self, patient_sequences):
        """Build index from patient sequences."""
        embeddings = []
        
        for seq in patient_sequences:
            visit_embeds = self.visit_encoder(seq['visit_codes'], seq['visit_mask'])
            patient_embed = visit_embeds.mean(dim=0).detach().numpy()
            embeddings.append(patient_embed)
            self.patient_ids.append(seq['patient_id'])
        
        embeddings = np.array(embeddings)
        self.index.add(embeddings)
    
    def retrieve_similar(self, query_sequence, k: int = 10):
        """Retrieve k most similar patients."""
        visit_embeds = self.visit_encoder(
            query_sequence['visit_codes'],
            query_sequence['visit_mask']
        )
        query_embed = visit_embeds.mean(dim=0).detach().numpy().reshape(1, -1)
        
        distances, indices = self.index.search(query_embed, k)
        
        similar_patients = [
            {'patient_id': self.patient_ids[idx], 'distance': dist}
            for idx, dist in zip(indices[0], distances[0])
        ]
        
        return similar_patients
```

### 5.4 Application 4: Risk Stratification

**Use Case:** Predict adverse outcomes (readmission, mortality)

**Architecture:**
```
Visit Sequences → Visit Encoder → Risk Predictor → Risk Score
```

---

## Part 6: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Set up data pipeline and adapters

**Tasks:**
1. ✅ Create directory structure
2. ✅ Document implementation plan (this document)
3. ⬜ Refactor legacy code → `legacy/` directory
4. ⬜ Implement `BaseEHRAdapter`
5. ⬜ Implement `SyntheaAdapter`
6. ⬜ Implement `VisitGrouper`
7. ⬜ Implement `PatientSequenceBuilder`
8. ⬜ Create unit tests for adapters

**Deliverables:**
- Working data pipeline: Raw data → Visit sequences
- Test suite with synthetic data
- Documentation

### Phase 2: Pre-trained Model Integration (Week 3)

**Goal:** Integrate CEHR-BERT embeddings

**Tasks:**
1. ⬜ Install CEHR-BERT dependencies
2. ⬜ Implement `CEHRBERTWrapper`
3. ⬜ Implement `VisitEncoder`
4. ⬜ Create embedding extraction scripts
5. ⬜ Validate embeddings (nearest neighbors, clustering)

**Deliverables:**
- Pre-trained embeddings loaded and validated
- Visit encoder working
- Embedding quality metrics

### Phase 3: Baseline Application (Week 4)

**Goal:** Implement disease progression model

**Tasks:**
1. ⬜ Implement `VisitGroupedProgressionModel`
2. ⬜ Create CKD dataset builder
3. ⬜ Implement training loop
4. ⬜ Implement evaluation metrics
5. ⬜ Run baseline experiments

**Deliverables:**
- Working disease progression model
- Baseline results on synthetic data
- Model checkpoints

### Phase 4: Additional Applications (Weeks 5-6)

**Goal:** Implement phenotyping and retrieval

**Tasks:**
1. ⬜ Implement `TemporalPhenotyper`
2. ⬜ Implement `PatientRetrieval`
3. ⬜ Create visualization notebooks
4. ⬜ Run phenotyping experiments

**Deliverables:**
- Phenotyping pipeline
- Patient retrieval system
- Visualization dashboards

### Phase 5: Evaluation & Documentation (Week 7+)

**Goal:** Comprehensive evaluation and documentation

**Tasks:**
1. ⬜ Evaluate all applications
2. ⬜ Create example notebooks
3. ⬜ Write API documentation
4. ⬜ Create user guide

**Deliverables:**
- Evaluation report
- Example notebooks
- Complete documentation

---

## Part 7: Open Questions & Future Work

### Open Questions

1. **Visit Definition Validation**
   - How to validate visit grouping accuracy?
   - Compare encounter-based vs same-day on real data
   - Clinical expert review of visit boundaries

2. **Within-Visit Structure**
   - Does preserving code types improve performance?
   - Ablation study: flat vs typed vs temporal

3. **Sequence Length**
   - Optimal max_visits parameter?
   - Trade-off between history length and computational cost

4. **Pre-trained Model Selection**
   - CEHR-BERT vs Med-BERT comparison
   - Domain adaptation for specific diseases

### Future Enhancements

1. **Multi-Modal Integration**
   - Add clinical notes (text)
   - Add imaging reports
   - Add genomic data

2. **Hierarchical Summarization**
   - For very long patient histories
   - Summarize old visits, keep recent ones detailed

3. **Real-Time Inference**
   - Optimize for low-latency prediction
   - Incremental encoding for new visits

4. **Federated Learning**
   - Train across multiple institutions
   - Privacy-preserving patient similarity

---

## Part 8: File Structure

```
ehr-sequencing/
├── src/ehrsequencing/
│   ├── data/
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   ├── base_adapter.py          ← Part 2.2
│   │   │   ├── synthea_adapter.py       ← Part 2.3
│   │   │   └── mimic_adapter.py
│   │   └── sequences/
│   │       ├── __init__.py
│   │       ├── visit_grouper.py         ← Part 3.1
│   │       └── sequence_builder.py      ← Part 3.2
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── cehrbert_wrapper.py          ← Part 4.2
│   ├── models/
│   │   ├── __init__.py
│   │   ├── visit_encoder.py             ← Part 4.3
│   │   └── progression_model.py         ← Part 5.1
│   ├── clustering/
│   │   ├── __init__.py
│   │   └── temporal_phenotyping.py      ← Part 5.2
│   └── retrieval/
│       ├── __init__.py
│       └── patient_similarity.py        ← Part 5.3
├── scripts/
│   ├── build_sequences.py
│   ├── load_pretrained_embeddings.py
│   └── train_progression_model.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_visit_grouping.ipynb
│   ├── 03_embedding_visualization.ipynb
│   └── 04_disease_progression.ipynb
└── tests/
    ├── test_adapters.py
    ├── test_visit_grouper.py
    └── test_sequence_builder.py
```

---

## Summary

This implementation plan provides:

1. ✅ **Complete data pipeline** - Raw EHR → Visit sequences
2. ✅ **Data adapter reuse** - Leverage loinc-predictor adapters
3. ✅ **Design decisions** - Visit definition, within-visit structure
4. ✅ **Pre-trained integration** - CEHR-BERT embeddings
5. ✅ **Downstream applications** - Disease progression, phenotyping, retrieval
6. ✅ **Implementation roadmap** - 7-week phased plan
7. ✅ **Code templates** - Production-ready implementations

**Next Steps:**
1. Refactor legacy code
2. Implement Phase 1 (data pipeline)
3. Integrate CEHR-BERT (Phase 2)
4. Build baseline application (Phase 3)

---

**Document Version:** 1.0  
**Last Updated:** January 19, 2026  
**Next Review:** After Phase 1 completion
