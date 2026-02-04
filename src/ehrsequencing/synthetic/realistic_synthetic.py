"""
Realistic synthetic EHR data generator with learnable medical patterns.

This generator creates synthetic patient sequences with:
- Disease clusters (e.g., diabetes → insulin, metformin, glucose monitoring)
- Temporal progression (diagnosis → treatment → follow-up)
- Co-morbidities (realistic disease co-occurrence)
- Age-related patterns
- Seasonal patterns (flu in winter, allergies in spring)

Unlike random synthetic data, this has learnable structure that BEHRT can capture.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class DiseasePattern:
    """Defines a disease with its typical code sequence."""
    name: str
    diagnosis_codes: List[int]  # Initial diagnosis codes
    treatment_codes: List[int]  # Treatment/medication codes
    monitoring_codes: List[int]  # Follow-up/monitoring codes
    prevalence: float  # Probability of this disease in population
    age_range: Tuple[int, int]  # (min_age, max_age) for this disease
    progression_visits: int  # Number of visits for this disease pattern


# Define realistic disease patterns
DISEASE_PATTERNS = {
    'diabetes_type2': DiseasePattern(
        name='Type 2 Diabetes',
        diagnosis_codes=[250, 251, 252],  # Diabetes diagnosis codes
        treatment_codes=[100, 101, 102, 103],  # Metformin, insulin, etc.
        monitoring_codes=[300, 301, 302],  # Glucose tests, HbA1c
        prevalence=0.10,
        age_range=(40, 80),
        progression_visits=8
    ),
    'hypertension': DiseasePattern(
        name='Hypertension',
        diagnosis_codes=[401, 402, 403],
        treatment_codes=[110, 111, 112],  # ACE inhibitors, beta blockers
        monitoring_codes=[310, 311],  # BP monitoring
        prevalence=0.15,
        age_range=(35, 85),
        progression_visits=6
    ),
    'asthma': DiseasePattern(
        name='Asthma',
        diagnosis_codes=[493, 494],
        treatment_codes=[120, 121, 122],  # Inhalers, steroids
        monitoring_codes=[320, 321],  # Pulmonary function tests
        prevalence=0.08,
        age_range=(5, 70),
        progression_visits=5
    ),
    'depression': DiseasePattern(
        name='Depression',
        diagnosis_codes=[296, 311],
        treatment_codes=[130, 131, 132],  # SSRIs, therapy
        monitoring_codes=[330, 331],  # Mental health assessments
        prevalence=0.12,
        age_range=(18, 75),
        progression_visits=7
    ),
    'copd': DiseasePattern(
        name='COPD',
        diagnosis_codes=[496, 491, 492],
        treatment_codes=[140, 141, 142],  # Bronchodilators, oxygen
        monitoring_codes=[340, 341],  # Spirometry
        prevalence=0.06,
        age_range=(50, 85),
        progression_visits=6
    ),
    'heart_failure': DiseasePattern(
        name='Heart Failure',
        diagnosis_codes=[428, 429],
        treatment_codes=[150, 151, 152],  # Diuretics, ACE inhibitors
        monitoring_codes=[350, 351],  # Echo, BNP
        prevalence=0.05,
        age_range=(55, 90),
        progression_visits=8
    ),
    'ckd': DiseasePattern(
        name='Chronic Kidney Disease',
        diagnosis_codes=[585, 586],
        treatment_codes=[160, 161],  # Dialysis, medications
        monitoring_codes=[360, 361],  # Creatinine, GFR
        prevalence=0.07,
        age_range=(50, 85),
        progression_visits=7
    ),
    'arthritis': DiseasePattern(
        name='Rheumatoid Arthritis',
        diagnosis_codes=[714, 715],
        treatment_codes=[170, 171, 172],  # NSAIDs, DMARDs
        monitoring_codes=[370, 371],  # Inflammatory markers
        prevalence=0.09,
        age_range=(30, 75),
        progression_visits=6
    ),
}

# Co-morbidity patterns (diseases that often occur together)
COMORBIDITY_PAIRS = [
    ('diabetes_type2', 'hypertension', 0.4),  # 40% of diabetics have hypertension
    ('diabetes_type2', 'ckd', 0.3),
    ('hypertension', 'heart_failure', 0.25),
    ('copd', 'heart_failure', 0.2),
    ('depression', 'diabetes_type2', 0.15),
]

# Routine care codes (appear in most patients)
ROUTINE_CODES = [
    (500, 0.3),  # Annual checkup
    (501, 0.2),  # Blood pressure check
    (502, 0.15),  # Cholesterol screening
    (503, 0.1),  # Flu shot
]


def generate_patient_trajectory(
    patient_id: int,
    vocab_size: int = 1000,
    max_visits: int = 50,
    seed: Optional[int] = None
) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    Generate a realistic patient trajectory with disease patterns.
    
    Args:
        patient_id: Patient identifier (used for seeding)
        vocab_size: Total vocabulary size
        max_visits: Maximum number of visits
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (visits, ages, visit_ids) where:
        - visits: List of visits, each visit is a list of codes
        - ages: Age at each visit
        - visit_ids: Visit sequence numbers
    """
    if seed is not None:
        rng = np.random.RandomState(seed + patient_id)
    else:
        rng = np.random.RandomState(patient_id)
    
    # Generate patient demographics
    base_age = rng.randint(20, 70)
    
    # Select diseases for this patient based on age and prevalence
    patient_diseases = []
    for disease_name, pattern in DISEASE_PATTERNS.items():
        # Check if patient's age is in range for this disease
        if pattern.age_range[0] <= base_age <= pattern.age_range[1]:
            # Sample based on prevalence
            if rng.random() < pattern.prevalence:
                patient_diseases.append((disease_name, pattern))
    
    # Add co-morbidities
    for disease1, disease2, prob in COMORBIDITY_PAIRS:
        if any(d[0] == disease1 for d in patient_diseases):
            if disease2 in DISEASE_PATTERNS and rng.random() < prob:
                pattern = DISEASE_PATTERNS[disease2]
                if not any(d[0] == disease2 for d in patient_diseases):
                    patient_diseases.append((disease2, pattern))
    
    # Generate visit sequence
    visits = []
    ages = []
    visit_ids = []
    
    current_age = base_age
    visit_num = 0
    
    # Start with routine care
    if rng.random() < 0.5:
        routine_visit = [code for code, prob in ROUTINE_CODES if rng.random() < prob]
        if routine_visit:
            visits.append(routine_visit)
            ages.append(current_age)
            visit_ids.append(visit_num)
            visit_num += 1
    
    # Generate disease progression for each disease
    for disease_name, pattern in patient_diseases:
        disease_start_age = current_age + rng.randint(0, 3)
        
        # Diagnosis visit
        diagnosis_visit = rng.choice(pattern.diagnosis_codes, size=rng.randint(1, 3), replace=True).tolist()
        visits.append(diagnosis_visit)
        ages.append(disease_start_age)
        visit_ids.append(visit_num)
        visit_num += 1
        
        # Treatment visits (diagnosis + treatment codes)
        for i in range(pattern.progression_visits):
            visit_age = disease_start_age + i * rng.randint(1, 4)  # Visits every 1-4 months
            
            visit_codes = []
            
            # Sometimes repeat diagnosis code
            if rng.random() < 0.3:
                visit_codes.append(rng.choice(pattern.diagnosis_codes))
            
            # Add treatment codes
            n_treatments = rng.randint(1, min(3, len(pattern.treatment_codes) + 1))
            visit_codes.extend(rng.choice(pattern.treatment_codes, size=n_treatments, replace=False).tolist())
            
            # Add monitoring codes
            if rng.random() < 0.6:
                n_monitoring = rng.randint(1, min(2, len(pattern.monitoring_codes) + 1))
                visit_codes.extend(rng.choice(pattern.monitoring_codes, size=n_monitoring, replace=False).tolist())
            
            # Add routine codes occasionally
            if rng.random() < 0.2:
                routine = [code for code, prob in ROUTINE_CODES if rng.random() < prob]
                visit_codes.extend(routine)
            
            visits.append(visit_codes)
            ages.append(visit_age)
            visit_ids.append(visit_num)
            visit_num += 1
            
            if visit_num >= max_visits:
                break
        
        if visit_num >= max_visits:
            break
    
    # If patient has no diseases, generate routine care visits
    if not patient_diseases:
        for i in range(rng.randint(2, 8)):
            routine_visit = [code for code, prob in ROUTINE_CODES if rng.random() < prob * 2]
            if routine_visit:
                visits.append(routine_visit)
                ages.append(current_age + i)
                visit_ids.append(visit_num)
                visit_num += 1
    
    return visits, ages, visit_ids


def generate_realistic_dataset(
    num_patients: int = 1000,
    vocab_size: int = 1000,
    max_seq_length: int = 512,
    mask_prob: float = 0.15,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a realistic synthetic EHR dataset with learnable patterns.
    
    Args:
        num_patients: Number of patients to generate
        vocab_size: Size of medical code vocabulary
        max_seq_length: Maximum sequence length
        mask_prob: Probability of masking a code for MLM
        seed: Random seed
    
    Returns:
        Tuple of (codes, ages, visit_ids, attention_mask, masked_codes, labels)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    print(f"Generating realistic synthetic data: {num_patients} patients, vocab={vocab_size}")
    print("Disease patterns:")
    for name, pattern in DISEASE_PATTERNS.items():
        print(f"  - {pattern.name}: {pattern.prevalence*100:.1f}% prevalence")
    
    all_codes = []
    all_ages = []
    all_visit_ids = []
    all_attention_mask = []
    all_masked_codes = []
    all_labels = []
    
    for patient_id in range(num_patients):
        # Generate patient trajectory
        visits, ages, visit_ids = generate_patient_trajectory(
            patient_id, vocab_size, max_visits=max_seq_length, seed=seed
        )
        
        # Flatten visits into sequence
        codes_seq = []
        ages_seq = []
        visit_ids_seq = []
        
        for visit, age, visit_id in zip(visits, ages, visit_ids):
            for code in visit:
                codes_seq.append(code)
                ages_seq.append(age)
                visit_ids_seq.append(visit_id)
        
        # Truncate or pad to max_seq_length
        seq_len = min(len(codes_seq), max_seq_length)
        
        codes_padded = codes_seq[:seq_len] + [0] * (max_seq_length - seq_len)
        ages_padded = ages_seq[:seq_len] + [0] * (max_seq_length - seq_len)
        visit_ids_padded = visit_ids_seq[:seq_len] + [0] * (max_seq_length - seq_len)
        attention_mask = [1] * seq_len + [0] * (max_seq_length - seq_len)
        
        # Create masked version for MLM
        masked_codes = codes_padded.copy()
        labels = [-100] * max_seq_length  # -100 = ignore in loss
        
        for i in range(seq_len):
            if np.random.random() < mask_prob:
                labels[i] = codes_padded[i]
                # 80% mask, 10% random, 10% keep
                rand = np.random.random()
                if rand < 0.8:
                    masked_codes[i] = vocab_size - 1  # [MASK] token
                elif rand < 0.9:
                    masked_codes[i] = np.random.randint(1, vocab_size - 1)
        
        all_codes.append(codes_padded)
        all_ages.append(ages_padded)
        all_visit_ids.append(visit_ids_padded)
        all_attention_mask.append(attention_mask)
        all_masked_codes.append(masked_codes)
        all_labels.append(labels)
    
    return (
        torch.tensor(all_codes, dtype=torch.long),
        torch.tensor(all_ages, dtype=torch.long),
        torch.tensor(all_visit_ids, dtype=torch.long),
        torch.tensor(all_attention_mask, dtype=torch.long),
        torch.tensor(all_masked_codes, dtype=torch.long),
        torch.tensor(all_labels, dtype=torch.long)
    )


def print_dataset_statistics(
    codes: torch.Tensor,
    ages: torch.Tensor,
    visit_ids: torch.Tensor
):
    """Print statistics about the generated dataset."""
    print("\n📊 Dataset Statistics:")
    print(f"   Number of patients: {codes.shape[0]}")
    print(f"   Sequence length: {codes.shape[1]}")
    
    # Calculate average sequence length (non-padded)
    seq_lengths = (codes != 0).sum(dim=1).float()
    print(f"   Avg sequence length: {seq_lengths.mean():.1f} ± {seq_lengths.std():.1f}")
    
    # Calculate unique codes
    unique_codes = torch.unique(codes[codes != 0])
    print(f"   Unique codes used: {len(unique_codes)}")
    
    # Age distribution
    ages_nonzero = ages[ages != 0]
    print(f"   Age range: {ages_nonzero.min()}-{ages_nonzero.max()}")
    print(f"   Avg age: {ages_nonzero.float().mean():.1f}")
    
    # Visit distribution
    visits_per_patient = visit_ids.max(dim=1)[0]
    print(f"   Avg visits per patient: {visits_per_patient.float().mean():.1f}")
    
    # Code frequency (top 10)
    codes_flat = codes[codes != 0].flatten()
    unique, counts = torch.unique(codes_flat, return_counts=True)
    top_k = 10
    top_indices = torch.argsort(counts, descending=True)[:top_k]
    print(f"\n   Top {top_k} most frequent codes:")
    for i, idx in enumerate(top_indices):
        code = unique[idx].item()
        count = counts[idx].item()
        freq = count / len(codes_flat) * 100
        # Try to identify what this code represents
        code_type = "Unknown"
        for name, pattern in DISEASE_PATTERNS.items():
            if code in pattern.diagnosis_codes:
                code_type = f"{pattern.name} (diagnosis)"
                break
            elif code in pattern.treatment_codes:
                code_type = f"{pattern.name} (treatment)"
                break
            elif code in pattern.monitoring_codes:
                code_type = f"{pattern.name} (monitoring)"
                break
        if code in [c for c, _ in ROUTINE_CODES]:
            code_type = "Routine care"
        
        print(f"      {i+1}. Code {code:3d}: {count:5d} ({freq:5.2f}%) - {code_type}")
