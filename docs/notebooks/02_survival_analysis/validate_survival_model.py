#!/usr/bin/env python3
"""
Validation script for discrete-time survival LSTM model.

This script provides quick validation and testing capabilities for the survival model
with options for:
- Patient subsampling for local testing
- Example patient sequence display
- Adjustable model complexity
- Memory estimation
- Synthetic outcome quality checks

Usage:
    # Quick validation with 200 patients on local system
    python validate_survival_model.py --max-patients 200 --show-examples 5
    
    # Full validation on cloud GPU
    python validate_survival_model.py --max-patients None --model-size large
    
    # Memory estimation only
    python validate_survival_model.py --estimate-memory-only
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
import scipy.stats as stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from ehrsequencing.data.adapters.synthea import SyntheaAdapter
from ehrsequencing.data.visit_grouper import VisitGrouper
from ehrsequencing.data.sequence_builder import PatientSequenceBuilder
from ehrsequencing.models.survival_lstm import DiscreteTimeSurvivalLSTM
from ehrsequencing.models.losses import DiscreteTimeSurvivalLoss
from ehrsequencing.synthetic.survival import DiscreteTimeSurvivalGenerator
from ehrsequencing.utils.sampling import (
    subsample_patients,
    get_recommended_batch_size,
    estimate_memory_gb,
    print_memory_recommendation,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate discrete-time survival LSTM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local testing with 200 patients
  python validate_survival_model.py --max-patients 200 --show-examples 5
  
  # Full dataset on cloud GPU
  python validate_survival_model.py --max-patients None --model-size large
  
  # Small model for quick iteration
  python validate_survival_model.py --max-patients 100 --model-size small --epochs 5
  
  # Memory estimation only (no training)
  python validate_survival_model.py --estimate-memory-only
        """
    )
    
    # Data options
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'large_cohort_1000',
        help='Path to Synthea data directory'
    )
    parser.add_argument(
        '--max-patients',
        type=str,
        default='200',
        help='Maximum number of patients to use (or "None" for all)'
    )
    parser.add_argument(
        '--min-visits',
        type=int,
        default=2,
        help='Minimum visits per patient'
    )
    
    # Model options
    parser.add_argument(
        '--model-size',
        choices=['small', 'medium', 'large'],
        default='medium',
        help='Model complexity: small (64/128), medium (128/256), large (256/512)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of LSTM layers'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate'
    )
    
    # Training options
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (auto-determined if not specified)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    # Synthetic outcome options
    parser.add_argument(
        '--censoring-rate',
        type=float,
        default=0.3,
        help='Censoring rate for synthetic outcomes'
    )
    parser.add_argument(
        '--time-scale',
        type=float,
        default=0.3,
        help='Time scale for synthetic outcomes'
    )
    
    # Display options
    parser.add_argument(
        '--show-examples',
        type=int,
        default=0,
        help='Number of example patient sequences to display'
    )
    parser.add_argument(
        '--check-outcomes',
        action='store_true',
        help='Run diagnostic checks on synthetic outcomes'
    )
    
    # Utility options
    parser.add_argument(
        '--estimate-memory-only',
        action='store_true',
        help='Only estimate memory requirements (no training)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'mps', 'cuda'],
        help='Device to use for training'
    )
    
    return parser.parse_args()


def get_model_config(model_size: str):
    """Get model configuration based on size."""
    configs = {
        'small': {'embedding_dim': 64, 'hidden_dim': 128},
        'medium': {'embedding_dim': 128, 'hidden_dim': 256},
        'large': {'embedding_dim': 256, 'hidden_dim': 512},
    }
    return configs[model_size]


def get_device(device_str: str):
    """Get torch device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def display_example_sequences(sequences, outcome, num_examples=5):
    """Display example patient sequences with their outcomes."""
    print("\n" + "="*80)
    print("EXAMPLE PATIENT SEQUENCES")
    print("="*80)
    
    for i in range(min(num_examples, len(sequences))):
        seq = sequences[i]
        event_time = outcome.event_times[i].item()
        event_indicator = outcome.event_indicators[i].item()
        risk_score = outcome.risk_scores[i].item()
        
        print(f"\nPatient {i+1}:")
        print(f"  • Patient ID: {seq.patient_id}")
        print(f"  • Number of visits: {len(seq.visits)}")
        print(f"  • Risk score: {risk_score:.3f}")
        print(f"  • Event time: {event_time} visits")
        print(f"  • Event indicator: {event_indicator} ({'Event' if event_indicator == 1 else 'Censored'})")
        
        print(f"\n  Visit timeline:")
        for j, visit in enumerate(seq.visits[:5]):  # Show first 5 visits
            codes = visit.get_all_codes()
            print(f"    Visit {j}: {len(codes)} codes - {', '.join(codes[:3])}{'...' if len(codes) > 3 else ''}")
        
        if len(seq.visits) > 5:
            print(f"    ... ({len(seq.visits) - 5} more visits)")
        
        if j + 1 == event_time and event_indicator == 1:
            print(f"    ⚠️  Event occurred at visit {event_time}")


def check_synthetic_outcomes(outcome, sequences):
    """Run diagnostic checks on synthetic outcome quality."""
    print("\n" + "="*80)
    print("SYNTHETIC OUTCOME QUALITY CHECKS")
    print("="*80)
    
    # Check correlation between risk scores and event times
    event_mask = outcome.event_indicators == 1
    event_risk_scores = outcome.risk_scores[event_mask].numpy()
    event_times = outcome.event_times[event_mask].numpy()
    
    if len(event_risk_scores) > 1:
        correlation, p_value = stats.pearsonr(event_risk_scores, event_times)
        
        print(f"\nCorrelation between risk score and event time:")
        print(f"  • Pearson r = {correlation:.3f} (p={p_value:.4f})")
        print(f"  • Expected: NEGATIVE correlation (high risk → early events)")
        
        if correlation > 0.1:
            print(f"  ⚠️  WARNING: POSITIVE correlation detected!")
            print(f"     High-risk patients have LATE events (inverse relationship)")
            print(f"     Model will learn backwards → C-index < 0.5")
            print(f"     FIX: Check synthetic outcome generator implementation")
        elif correlation < -0.2:
            print(f"  ✅ Good: Negative correlation detected")
            print(f"     High-risk patients have early events (correct relationship)")
        else:
            print(f"  ⚠️  Weak correlation: outcomes may be too random")
    
    # Event rate
    event_rate = outcome.event_indicators.float().mean().item()
    print(f"\nEvent statistics:")
    print(f"  • Event rate: {event_rate:.1%}")
    print(f"  • Censored rate: {1-event_rate:.1%}")
    
    if event_rate < 0.3:
        print(f"  ⚠️  Low event rate may make training difficult")
    elif event_rate > 0.9:
        print(f"  ⚠️  Very high event rate (little censoring)")
    else:
        print(f"  ✅ Reasonable event rate for survival analysis")
    
    # Risk score distribution
    print(f"\nRisk score distribution:")
    print(f"  • Mean: {outcome.risk_scores.mean():.3f}")
    print(f"  • Std: {outcome.risk_scores.std():.3f}")
    print(f"  • Min: {outcome.risk_scores.min():.3f}")
    print(f"  • Max: {outcome.risk_scores.max():.3f}")
    
    # Example patients
    print(f"\nExample patients (with events):")
    print(f"{'Risk Score':<12} {'Event Time':<12} {'Expected':<20}")
    print("-" * 50)
    for i in range(min(10, len(event_risk_scores))):
        expected = "Early" if event_risk_scores[i] > 0.6 else "Late"
        actual = "Early" if event_times[i] < np.median(event_times) else "Late"
        match = "✓" if expected == actual else "✗"
        print(f"{event_risk_scores[i]:.3f}        {event_times[i]:<12} {expected:<10} (actual: {actual}) {match}")


def load_and_prepare_data(args):
    """Load and prepare data for training."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    print(f"\nLoading Synthea data from: {args.data_dir}")
    adapter = SyntheaAdapter(args.data_dir)
    events = adapter.load_events()
    print(f"Loaded {len(events)} events")
    
    # Group into visits
    print("\nGrouping events into visits...")
    grouper = VisitGrouper(time_window_hours=24)
    visits = grouper.group_events(events)
    print(f"Created {len(visits)} visits")
    
    # Group by patient
    visits_by_patient = defaultdict(list)
    for visit in visits:
        visits_by_patient[visit.patient_id].append(visit)
    
    # Sort visits by timestamp
    for patient_id in visits_by_patient:
        visits_by_patient[patient_id].sort(key=lambda v: v.timestamp)
    
    print(f"\nData organized for {len(visits_by_patient)} patients")
    
    # Subsample patients
    max_patients = None if args.max_patients.lower() == 'none' else int(args.max_patients)
    visits_by_patient = subsample_patients(
        visits_by_patient,
        max_patients=max_patients,
        seed=args.seed,
        verbose=True
    )
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    builder = PatientSequenceBuilder()
    builder.build_vocabulary(list(visits_by_patient.values()))
    print(f"Vocabulary size: {builder.vocabulary_size}")
    
    # Build sequences
    print("\nBuilding patient sequences...")
    sequences = builder.build_sequences(list(visits_by_patient.values()), min_visits=args.min_visits)
    print(f"Built {len(sequences)} sequences")
    
    # Compute statistics
    avg_visits = np.mean([len(seq.visits) for seq in sequences])
    max_visits = max([len(seq.visits) for seq in sequences])
    avg_codes = np.mean([
        np.mean([visit.num_codes() for visit in seq.visits])
        for seq in sequences
    ])
    max_codes = max([
        max([visit.num_codes() for visit in seq.visits])
        for seq in sequences
    ])
    
    print(f"\nDataset statistics:")
    print(f"  • Patients: {len(sequences)}")
    print(f"  • Avg visits per patient: {avg_visits:.1f}")
    print(f"  • Max visits per patient: {max_visits}")
    print(f"  • Avg codes per visit: {avg_codes:.1f}")
    print(f"  • Max codes per visit: {max_codes}")
    
    return builder, sequences, {
        'num_patients': len(sequences),
        'avg_visits': avg_visits,
        'max_visits': max_visits,
        'avg_codes_per_visit': avg_codes,
        'max_codes_per_visit': max_codes,
    }


def estimate_memory(args, builder, stats, model_config):
    """Estimate memory requirements."""
    print("\n" + "="*80)
    print("MEMORY ESTIMATION")
    print("="*80)
    
    batch_size = args.batch_size or get_recommended_batch_size(
        stats['num_patients'],
        device=args.device,
        model_size=args.model_size
    )
    
    mem_est = estimate_memory_gb(
        num_patients=stats['num_patients'],
        avg_visits=stats['avg_visits'],
        max_visits=int(stats['max_visits']),
        avg_codes_per_visit=stats['avg_codes_per_visit'],
        max_codes_per_visit=int(stats['max_codes_per_visit']),
        vocab_size=builder.vocabulary_size,
        embedding_dim=model_config['embedding_dim'],
        hidden_dim=model_config['hidden_dim'],
        batch_size=batch_size,
    )
    
    print_memory_recommendation(mem_est, verbose=True)
    
    return mem_est


def generate_outcomes(args, sequences):
    """Generate synthetic survival outcomes."""
    print("\n" + "="*80)
    print("GENERATING SYNTHETIC OUTCOMES")
    print("="*80)
    
    generator = DiscreteTimeSurvivalGenerator(
        censoring_rate=args.censoring_rate,
        risk_weights={'comorbidity': 0.4, 'frequency': 0.4, 'diversity': 0.2},
        time_scale=args.time_scale,
        seed=args.seed
    )
    
    outcome = generator.generate(sequences)
    
    print(f"\nGenerated outcomes for {len(sequences)} patients")
    print(f"  • Event rate: {outcome.event_indicators.float().mean():.1%}")
    print(f"  • Median event/censoring time: {outcome.event_times.float().median():.1f} visits")
    print(f"  • Mean risk score: {outcome.risk_scores.mean():.3f}")
    
    return outcome


def main():
    """Main validation function."""
    args = parse_args()
    
    print("="*80)
    print("DISCRETE-TIME SURVIVAL LSTM VALIDATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  • Data directory: {args.data_dir}")
    print(f"  • Max patients: {args.max_patients}")
    print(f"  • Model size: {args.model_size}")
    print(f"  • Epochs: {args.epochs}")
    print(f"  • Device: {args.device}")
    
    # Get model configuration
    model_config = get_model_config(args.model_size)
    
    # Load data
    builder, sequences, stats = load_and_prepare_data(args)
    
    # Estimate memory
    mem_est = estimate_memory(args, builder, stats, model_config)
    
    if args.estimate_memory_only:
        print("\n✅ Memory estimation complete (--estimate-memory-only specified)")
        return
    
    # Generate outcomes
    outcome = generate_outcomes(args, sequences)
    
    # Show examples if requested
    if args.show_examples > 0:
        display_example_sequences(sequences, outcome, num_examples=args.show_examples)
    
    # Check outcome quality if requested
    if args.check_outcomes:
        check_synthetic_outcomes(outcome, sequences)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\n💡 Next steps:")
    print("  1. If memory estimate looks good, proceed with training")
    print("  2. If correlation check shows issues, fix synthetic generator")
    print("  3. Adjust --model-size or --max-patients if needed")
    print("  4. Use full notebook for complete training workflow")


if __name__ == '__main__':
    main()
