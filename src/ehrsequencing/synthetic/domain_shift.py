"""
Domain-shifted synthetic EHR dataset generation for transfer learning evaluation.

This module provides pre-configured domain shift scenarios to test transfer learning
without requiring driver scripts to manage disease pattern modifications.

Domain shift simulates real-world scenarios like:
- Training on general population, deploying to elderly care
- Training on one hospital system, deploying to another
- Training on historical data (2010-2015), deploying to recent data (2016-2020)
"""

import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .realistic_synthetic import (
    generate_realistic_dataset,
    print_dataset_statistics,
    DISEASE_PATTERNS
)


@dataclass
class DomainConfig:
    """Configuration for a specific domain (source or target)."""
    name: str
    description: str
    prevalence_multiplier: float  # Multiply base prevalence by this factor
    age_shift: int  # Shift age range by this amount (positive = older)
    age_min: int  # Minimum age for this domain
    age_max: int  # Maximum age for this domain


# Pre-defined domain shift scenarios
DOMAIN_SCENARIOS = {
    'general_to_elderly': {
        'source': DomainConfig(
            name='General Population',
            description='Younger, healthier population (primary care)',
            prevalence_multiplier=0.6,  # 40% lower disease rates
            age_shift=-15,
            age_min=20,
            age_max=60
        ),
        'target': DomainConfig(
            name='Elderly Care',
            description='Older, sicker population (specialized care)',
            prevalence_multiplier=1.8,  # 80% higher disease rates
            age_shift=20,
            age_min=50,
            age_max=90
        )
    },
    'hospital_a_to_b': {
        'source': DomainConfig(
            name='Hospital A (Urban)',
            description='Urban hospital with diverse population',
            prevalence_multiplier=1.0,
            age_shift=0,
            age_min=20,
            age_max=85
        ),
        'target': DomainConfig(
            name='Hospital B (Rural)',
            description='Rural hospital with older, sicker population',
            prevalence_multiplier=1.3,  # 30% higher disease rates
            age_shift=10,
            age_min=30,
            age_max=90
        )
    },
    'historical_to_recent': {
        'source': DomainConfig(
            name='Historical Data (2010-2015)',
            description='Older treatment patterns and demographics',
            prevalence_multiplier=0.9,  # Slightly lower prevalence
            age_shift=-5,
            age_min=20,
            age_max=80
        ),
        'target': DomainConfig(
            name='Recent Data (2016-2020)',
            description='Modern treatment patterns, aging population',
            prevalence_multiplier=1.2,  # 20% higher prevalence
            age_shift=5,
            age_min=25,
            age_max=85
        )
    }
}


def apply_domain_config(config: DomainConfig) -> None:
    """
    Apply domain configuration to global DISEASE_PATTERNS.
    
    Args:
        config: Domain configuration to apply
    """
    for disease_name, pattern in DISEASE_PATTERNS.items():
        # Adjust prevalence
        pattern.prevalence *= config.prevalence_multiplier
        
        # Adjust age range
        min_age, max_age = pattern.age_range
        new_min = max(config.age_min, min_age + config.age_shift)
        new_max = min(config.age_max, max_age + config.age_shift)
        pattern.age_range = (new_min, new_max)


def restore_original_patterns(original_patterns: Dict) -> None:
    """
    Restore DISEASE_PATTERNS to original values.
    
    Args:
        original_patterns: Dictionary of original pattern values
    """
    for disease_name, pattern in DISEASE_PATTERNS.items():
        pattern.prevalence = original_patterns[disease_name]['prevalence']
        pattern.age_range = original_patterns[disease_name]['age_range']


def save_original_patterns() -> Dict:
    """
    Save current DISEASE_PATTERNS values.
    
    Returns:
        Dictionary of original pattern values
    """
    original_patterns = {}
    for disease_name, pattern in DISEASE_PATTERNS.items():
        original_patterns[disease_name] = {
            'prevalence': pattern.prevalence,
            'age_range': pattern.age_range
        }
    return original_patterns


def generate_domain_shifted_datasets(
    source_patients: int,
    target_patients: int,
    scenario: str = 'general_to_elderly',
    vocab_size: int = 1000,
    max_seq_length: int = 256,
    source_seed: int = 42,
    target_seed: int = 123,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Generate two datasets with pre-configured domain shift for transfer learning.
    
    This is the main API for transfer learning benchmarks. Driver scripts just need
    to call this function without worrying about disease pattern modifications.
    
    Args:
        source_patients: Number of patients in source dataset
        target_patients: Number of patients in target dataset
        scenario: Pre-defined scenario name (see DOMAIN_SCENARIOS)
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length
        source_seed: Random seed for source dataset
        target_seed: Random seed for target dataset
        verbose: Print detailed information
    
    Returns:
        Tuple of (source_data, target_data) dictionaries, each containing:
        - 'codes': Code sequences
        - 'ages': Age sequences
        - 'visit_ids': Visit ID sequences
        - 'attention_mask': Attention mask
        - 'labels': MLM labels
    
    Example:
        >>> source_data, target_data = generate_domain_shifted_datasets(
        ...     source_patients=10000,
        ...     target_patients=5000,
        ...     scenario='general_to_elderly'
        ... )
        >>> # Use for training
        >>> train_on_source(source_data)
        >>> evaluate_on_target(target_data)
    
    Available scenarios:
        - 'general_to_elderly': General population → Elderly care (default)
        - 'hospital_a_to_b': Urban hospital → Rural hospital
        - 'historical_to_recent': 2010-2015 data → 2016-2020 data
    """
    if scenario not in DOMAIN_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available: {list(DOMAIN_SCENARIOS.keys())}"
        )
    
    source_config = DOMAIN_SCENARIOS[scenario]['source']
    target_config = DOMAIN_SCENARIOS[scenario]['target']
    
    if verbose:
        print("\n" + "="*80)
        print("GENERATING DOMAIN-SHIFTED DATASETS")
        print("="*80)
        print(f"\n🔄 Scenario: {scenario}")
        print(f"   Source: {source_config.name}")
        print(f"           {source_config.description}")
        print(f"   Target: {target_config.name}")
        print(f"           {target_config.description}")
    
    # Save original patterns
    original_patterns = save_original_patterns()
    
    try:
        # Generate source dataset
        if verbose:
            print(f"\n📊 Source Dataset: {source_config.name}")
            print(f"   Patients: {source_patients}")
            print(f"   Age range: {source_config.age_min}-{source_config.age_max} years")
            print(f"   Disease prevalence: {source_config.prevalence_multiplier:.1f}x baseline")
        
        apply_domain_config(source_config)
        codes_src, ages_src, visit_ids_src, attention_mask_src, masked_codes_src, labels_src = generate_realistic_dataset(
            num_patients=source_patients,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            seed=source_seed
        )
        source_data = {
            'codes': codes_src,
            'ages': ages_src,
            'visit_ids': visit_ids_src,
            'attention_mask': attention_mask_src,
            'labels': labels_src
        }
        if verbose:
            print_dataset_statistics(codes_src, ages_src, visit_ids_src)
        
        # Restore and apply target configuration
        restore_original_patterns(original_patterns)
        
        # Generate target dataset
        if verbose:
            print(f"\n📊 Target Dataset: {target_config.name}")
            print(f"   Patients: {target_patients}")
            print(f"   Age range: {target_config.age_min}-{target_config.age_max} years")
            print(f"   Disease prevalence: {target_config.prevalence_multiplier:.1f}x baseline")
        
        apply_domain_config(target_config)
        codes_tgt, ages_tgt, visit_ids_tgt, attention_mask_tgt, masked_codes_tgt, labels_tgt = generate_realistic_dataset(
            num_patients=target_patients,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            seed=target_seed
        )
        target_data = {
            'codes': codes_tgt,
            'ages': ages_tgt,
            'visit_ids': visit_ids_tgt,
            'attention_mask': attention_mask_tgt,
            'labels': labels_tgt
        }
        if verbose:
            print_dataset_statistics(codes_tgt, ages_tgt, visit_ids_tgt)
        
        if verbose:
            print("\n✅ Domain shift created successfully!")
            print(f"   Expected transfer learning challenge: Medium-High")
            print(f"   Source and target have different demographics and disease patterns")
        
        return source_data, target_data
        
    finally:
        # Always restore original patterns
        restore_original_patterns(original_patterns)


def list_scenarios() -> None:
    """Print all available domain shift scenarios."""
    print("\n📋 Available Domain Shift Scenarios:")
    print("="*80)
    for scenario_name, configs in DOMAIN_SCENARIOS.items():
        source = configs['source']
        target = configs['target']
        print(f"\n🔹 {scenario_name}")
        print(f"   Source: {source.name}")
        print(f"           {source.description}")
        print(f"   Target: {target.name}")
        print(f"           {target.description}")
    print("\n" + "="*80)
