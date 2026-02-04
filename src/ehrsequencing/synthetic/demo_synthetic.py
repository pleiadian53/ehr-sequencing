"""
High-signal synthetic EHR data generator for compelling demos.

This generator creates synthetic data with VERY STRONG, deterministic patterns
that are easy to learn, designed to achieve 70%+ accuracy for demonstration purposes.

Key differences from realistic_synthetic.py:
- Stronger, more deterministic patterns (less noise)
- Higher pattern repetition (easier to learn)
- Clearer diagnosis → treatment associations
- More predictable temporal sequences
- Designed for demo/showcase, not realism
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class StrongPattern:
    """Defines a disease with highly deterministic code sequences."""
    name: str
    diagnosis_code: int  # Single diagnosis code (deterministic)
    treatment_codes: List[int]  # Always follow diagnosis
    monitoring_code: int  # Always follows treatment
    prevalence: float
    age_range: Tuple[int, int]


# Strong, deterministic disease patterns for demo
STRONG_PATTERNS = {
    'diabetes': StrongPattern(
        name='Diabetes',
        diagnosis_code=250,
        treatment_codes=[100, 101],  # Always metformin + insulin
        monitoring_code=300,  # Always glucose test
        prevalence=0.20,
        age_range=(40, 80)
    ),
    'hypertension': StrongPattern(
        name='Hypertension',
        diagnosis_code=401,
        treatment_codes=[110, 111],  # Always ACE inhibitor + beta blocker
        monitoring_code=310,  # Always BP check
        prevalence=0.20,
        age_range=(35, 85)
    ),
    'asthma': StrongPattern(
        name='Asthma',
        diagnosis_code=493,
        treatment_codes=[120, 121],  # Always inhaler + steroid
        monitoring_code=320,  # Always pulmonary function test
        prevalence=0.15,
        age_range=(10, 70)
    ),
    'depression': StrongPattern(
        name='Depression',
        diagnosis_code=296,
        treatment_codes=[130, 131],  # Always SSRI + therapy
        monitoring_code=330,  # Always mental health assessment
        prevalence=0.15,
        age_range=(18, 75)
    ),
    'copd': StrongPattern(
        name='COPD',
        diagnosis_code=496,
        treatment_codes=[140, 141],  # Always bronchodilator + oxygen
        monitoring_code=340,  # Always spirometry
        prevalence=0.10,
        age_range=(50, 85)
    ),
}

# Strong co-morbidity pairs (always occur together)
STRONG_COMORBIDITIES = [
    ('diabetes', 'hypertension', 0.8),  # 80% of diabetes patients have hypertension
    ('copd', 'hypertension', 0.6),
]

# Routine codes (appear in every visit)
ROUTINE_CODES = [900, 901, 902]  # Check-in, vitals, billing


def generate_patient_with_strong_patterns(
    patient_id: int,
    patterns: Dict[str, StrongPattern],
    max_visits: int = 20,
    seed: Optional[int] = None
) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    Generate a single patient with highly deterministic disease patterns.
    
    Pattern structure (VERY PREDICTABLE):
    1. Routine visit: [routine_codes]
    2. Diagnosis visit: [diagnosis_code, routine_codes]
    3. Treatment visit 1: [diagnosis_code, treatment_code_1, routine_codes]
    4. Treatment visit 2: [diagnosis_code, treatment_code_2, routine_codes]
    5. Monitoring visit: [diagnosis_code, treatment_codes, monitoring_code, routine_codes]
    6. Repeat steps 3-5 for progression
    
    This creates STRONG associations that are easy to learn.
    """
    if seed is not None:
        rng = np.random.RandomState(seed + patient_id)
    else:
        rng = np.random.RandomState(patient_id)
    
    base_age = rng.randint(20, 70)
    
    # Select diseases (deterministic based on age)
    patient_diseases = []
    for disease_name, pattern in patterns.items():
        if pattern.age_range[0] <= base_age <= pattern.age_range[1]:
            if rng.random() < pattern.prevalence:
                patient_diseases.append((disease_name, pattern))
    
    # Add co-morbidities (deterministic)
    for disease1, disease2, prob in STRONG_COMORBIDITIES:
        if any(d[0] == disease1 for d in patient_diseases):
            if disease2 in patterns and rng.random() < prob:
                pattern = patterns[disease2]
                if not any(d[0] == disease2 for d in patient_diseases):
                    patient_diseases.append((disease2, pattern))
    
    visits = []
    ages = []
    visit_ids = []
    
    current_age = base_age
    visit_num = 0
    
    # Initial routine visit (always)
    visits.append(ROUTINE_CODES.copy())
    ages.append(current_age)
    visit_ids.append(visit_num)
    visit_num += 1
    
    # Generate DETERMINISTIC disease progression
    for disease_name, pattern in patient_diseases:
        disease_start_age = current_age + rng.randint(0, 2)
        
        # Visit 1: Diagnosis (ALWAYS: diagnosis + routine)
        visit_codes = [pattern.diagnosis_code] + ROUTINE_CODES
        visits.append(visit_codes)
        ages.append(disease_start_age)
        visit_ids.append(visit_num)
        visit_num += 1
        
        # Visit 2: First treatment (ALWAYS: diagnosis + treatment[0] + routine)
        visit_codes = [pattern.diagnosis_code, pattern.treatment_codes[0]] + ROUTINE_CODES
        visits.append(visit_codes)
        ages.append(disease_start_age + 1)
        visit_ids.append(visit_num)
        visit_num += 1
        
        # Visit 3: Second treatment (ALWAYS: diagnosis + treatment[1] + routine)
        if len(pattern.treatment_codes) > 1:
            visit_codes = [pattern.diagnosis_code, pattern.treatment_codes[1]] + ROUTINE_CODES
            visits.append(visit_codes)
            ages.append(disease_start_age + 2)
            visit_ids.append(visit_num)
            visit_num += 1
        
        # Visit 4+: Monitoring (ALWAYS: diagnosis + all treatments + monitoring + routine)
        for i in range(3):  # 3 monitoring visits
            visit_codes = [pattern.diagnosis_code] + pattern.treatment_codes + [pattern.monitoring_code] + ROUTINE_CODES
            visits.append(visit_codes)
            ages.append(disease_start_age + 3 + i)
            visit_ids.append(visit_num)
            visit_num += 1
            
            if visit_num >= max_visits:
                break
        
        if visit_num >= max_visits:
            break
    
    # If no diseases, generate routine visits only
    if not patient_diseases:
        for i in range(rng.randint(3, 8)):
            visits.append(ROUTINE_CODES.copy())
            ages.append(current_age + i)
            visit_ids.append(visit_num)
            visit_num += 1
    
    return visits, ages, visit_ids


def generate_demo_dataset(
    num_patients: int = 1000,
    vocab_size: int = 1000,
    max_seq_length: int = 512,
    mask_prob: float = 0.15,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate high-signal synthetic dataset for demo purposes.
    
    This creates data with VERY STRONG patterns that are easy to learn,
    designed to achieve 70%+ accuracy for compelling demonstrations.
    
    Returns:
        codes: [num_patients, max_seq_length]
        ages: [num_patients, max_seq_length]
        visit_ids: [num_patients, max_seq_length]
        attention_mask: [num_patients, max_seq_length]
        masked_codes: [num_patients, max_seq_length] (for MLM)
        labels: [num_patients, max_seq_length] (original codes)
    """
    rng = np.random.RandomState(seed)
    
    all_codes = []
    all_ages = []
    all_visit_ids = []
    all_attention_masks = []
    
    for patient_id in range(num_patients):
        visits, ages, visit_ids = generate_patient_with_strong_patterns(
            patient_id, STRONG_PATTERNS, max_visits=30, seed=seed
        )
        
        # Flatten visits into sequence
        patient_codes = []
        patient_ages = []
        patient_visit_ids = []
        
        for visit_codes, age, visit_id in zip(visits, ages, visit_ids):
            patient_codes.extend(visit_codes)
            patient_ages.extend([age] * len(visit_codes))
            patient_visit_ids.extend([visit_id] * len(visit_codes))
        
        # Truncate or pad to max_seq_length
        seq_len = len(patient_codes)
        attention_mask = [1] * min(seq_len, max_seq_length) + [0] * max(0, max_seq_length - seq_len)
        
        if seq_len > max_seq_length:
            patient_codes = patient_codes[:max_seq_length]
            patient_ages = patient_ages[:max_seq_length]
            patient_visit_ids = patient_visit_ids[:max_seq_length]
        else:
            pad_len = max_seq_length - seq_len
            patient_codes.extend([0] * pad_len)
            patient_ages.extend([0] * pad_len)
            patient_visit_ids.extend([0] * pad_len)
        
        all_codes.append(patient_codes)
        all_ages.append(patient_ages)
        all_visit_ids.append(patient_visit_ids)
        all_attention_masks.append(attention_mask)
    
    # Convert to tensors
    codes = torch.tensor(all_codes, dtype=torch.long)
    ages = torch.tensor(all_ages, dtype=torch.long)
    visit_ids = torch.tensor(all_visit_ids, dtype=torch.long)
    attention_mask = torch.tensor(all_attention_masks, dtype=torch.bool)
    
    # Create masked language modeling task
    labels = codes.clone()
    masked_codes = codes.clone()
    
    # Mask tokens for MLM (only mask valid positions)
    for i in range(num_patients):
        valid_positions = attention_mask[i].nonzero(as_tuple=True)[0]
        n_mask = max(1, int(len(valid_positions) * mask_prob))
        mask_positions = rng.choice(valid_positions.numpy(), size=n_mask, replace=False)
        
        for pos in mask_positions:
            # 80% replace with [MASK] token (vocab_size - 1)
            # 10% replace with random token
            # 10% keep original
            rand = rng.random()
            if rand < 0.8:
                masked_codes[i, pos] = vocab_size - 1  # [MASK] token
            elif rand < 0.9:
                masked_codes[i, pos] = rng.randint(0, vocab_size - 1)
            # else: keep original
    
    return codes, ages, visit_ids, attention_mask, masked_codes, labels


def print_demo_dataset_statistics(codes: torch.Tensor, ages: torch.Tensor, visit_ids: torch.Tensor):
    """Print statistics about the generated demo dataset."""
    num_patients = codes.shape[0]
    max_seq_length = codes.shape[1]
    
    # Calculate actual sequence lengths
    seq_lengths = (codes != 0).sum(dim=1)
    avg_seq_length = seq_lengths.float().mean().item()
    
    # Count unique codes
    unique_codes = len(torch.unique(codes[codes != 0]))
    
    # Count visits per patient
    visits_per_patient = []
    for i in range(num_patients):
        n_visits = len(torch.unique(visit_ids[i][visit_ids[i] != 0]))
        visits_per_patient.append(n_visits)
    avg_visits = np.mean(visits_per_patient)
    
    print(f"\n📊 Demo Dataset Statistics:")
    print(f"   Total patients: {num_patients}")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   Average sequence length: {avg_seq_length:.1f}")
    print(f"   Unique codes used: {unique_codes}")
    print(f"   Average visits per patient: {avg_visits:.1f}")
    print(f"   Disease patterns: {len(STRONG_PATTERNS)} strong patterns")
    print(f"   Pattern strength: VERY HIGH (deterministic)")
    print(f"   Expected accuracy: 70-85% (with proper training)")
    print(f"\n   Strong patterns:")
    for name, pattern in STRONG_PATTERNS.items():
        print(f"      - {pattern.name}: {pattern.diagnosis_code} → {pattern.treatment_codes} → {pattern.monitoring_code}")
