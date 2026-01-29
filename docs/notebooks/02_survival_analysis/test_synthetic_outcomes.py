#!/usr/bin/env python3
"""
Fast validation script for synthetic survival outcome generation.

This script quickly validates that the DiscreteTimeSurvivalGenerator produces
realistic outcomes with proper risk-time correlation, without running the full
notebook training pipeline.

Usage:
    # Quick test with 50 patients (default)
    python test_synthetic_outcomes.py
    
    # Test with more patients
    python test_synthetic_outcomes.py --max-patients 200
    
    # Test different parameters
    python test_synthetic_outcomes.py --censoring-rate 0.4 --time-scale 0.5
    
    # Show visualizations
    python test_synthetic_outcomes.py --plot
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import scipy.stats as stats
import torch
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from ehrsequencing.data.adapters.synthea import SyntheaAdapter
from ehrsequencing.data.visit_grouper import VisitGrouper
from ehrsequencing.data.sequence_builder import PatientSequenceBuilder
from ehrsequencing.synthetic.survival import DiscreteTimeSurvivalGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate synthetic survival outcome generation'
    )
    parser.add_argument(
        '--max-patients',
        type=int,
        default=50,
        help='Number of patients to test (default: 50 for speed)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'large_cohort_1000',
        help='Path to Synthea data directory'
    )
    parser.add_argument(
        '--censoring-rate',
        type=float,
        default=0.3,
        help='Censoring rate (default: 0.3)'
    )
    parser.add_argument(
        '--time-scale',
        type=float,
        default=0.3,
        help='Time scale parameter (default: 0.3)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Show matplotlib plots (requires display)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--save',
        type=Path,
        default=None,
        help='Save validated outcomes to file (e.g., synthetic_outcomes.pt)'
    )
    
    return parser.parse_args()


def load_data(args):
    """Load and prepare patient sequences."""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load Synthea data
    print(f"\nLoading data from: {args.data_dir}")
    adapter = SyntheaAdapter(args.data_dir)
    events = adapter.load_events()
    print(f"✓ Loaded {len(events)} events")
    
    # Group into visits
    grouper = VisitGrouper(time_window_hours=24)
    visits = grouper.group_events(events)
    print(f"✓ Created {len(visits)} visits")
    
    # Group by patient
    visits_by_patient = defaultdict(list)
    for visit in visits:
        visits_by_patient[visit.patient_id].append(visit)
    
    # Sort visits by timestamp
    for patient_id in visits_by_patient:
        visits_by_patient[patient_id].sort(key=lambda v: v.timestamp)
    
    total_patients = len(visits_by_patient)
    print(f"✓ Organized {total_patients} patients")
    
    # Subsample if needed
    if args.max_patients and total_patients > args.max_patients:
        print(f"\n⚠️  Subsampling to {args.max_patients} patients for speed")
        np.random.seed(args.seed)
        sampled_patient_ids = np.random.choice(
            list(visits_by_patient.keys()),
            size=args.max_patients,
            replace=False
        )
        visits_by_patient = {
            pid: visits_by_patient[pid]
            for pid in sampled_patient_ids
        }
        print(f"✓ Using {len(visits_by_patient)} patients")
    
    # Build sequences
    builder = PatientSequenceBuilder()
    builder.build_vocabulary(list(visits_by_patient.values()))
    sequences = builder.build_sequences(list(visits_by_patient.values()), min_visits=2)
    
    print(f"✓ Built {len(sequences)} sequences")
    print(f"✓ Vocabulary size: {builder.vocabulary_size}")
    
    return sequences


def generate_outcomes(args, sequences):
    """Generate synthetic outcomes."""
    print("\n" + "="*70)
    print("GENERATING SYNTHETIC OUTCOMES")
    print("="*70)
    
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
    print(f"  • Risk score range: [{outcome.risk_scores.min():.3f}, {outcome.risk_scores.max():.3f}]")
    
    return outcome


def validate_correlation(outcome, sequences, verbose=False):
    """Validate risk-time correlation."""
    print("\n" + "="*70)
    print("CORRELATION VALIDATION")
    print("="*70)
    
    # Only look at patients with events (not censored)
    event_mask = outcome.event_indicators == 1
    event_risk_scores = outcome.risk_scores[event_mask].numpy()
    event_times = outcome.event_times[event_mask].numpy()
    
    if len(event_risk_scores) < 2:
        print("⚠️  Not enough events to compute correlation")
        return False
    
    # Compute correlation
    correlation, p_value = stats.pearsonr(event_risk_scores, event_times)
    
    print(f"\nCorrelation between risk score and event time:")
    print(f"  • Pearson r = {correlation:.3f} (p={p_value:.4f})")
    print(f"  • Expected: NEGATIVE correlation (high risk → early events)")
    print(f"  • Sample size: {len(event_risk_scores)} events")
    
    # Assess correlation strength
    success = False
    if correlation > 0:
        print(f"\n❌ FAIL: POSITIVE correlation detected!")
        print(f"   High-risk patients have LATE events (backwards)")
        print(f"   The model will learn the wrong relationship")
    elif correlation < -0.5:
        print(f"\n✅ PASS: Strong negative correlation")
        print(f"   High-risk patients have early events (correct)")
        success = True
    elif correlation < -0.3:
        print(f"\n⚠️  WARNING: Moderate negative correlation")
        print(f"   Correlation exists but could be stronger")
        print(f"   Consider adjusting time_scale or noise parameters")
        success = True
    else:
        print(f"\n❌ FAIL: Weak correlation")
        print(f"   Outcomes are too random, risk signal is lost")
    
    # Show examples
    if verbose or not success:
        print(f"\nExample patients (with events):")
        print(f"{'Risk Score':<12} {'Event Time':<12} {'Expected':<20}")
        print("-" * 50)
        
        num_examples = min(15, len(event_risk_scores))
        median_time = np.median(event_times)
        
        for i in range(num_examples):
            expected = "Early" if event_risk_scores[i] > 0.6 else "Late"
            actual = "Early" if event_times[i] < median_time else "Late"
            match = "✓" if expected == actual else "✗"
            print(f"{event_risk_scores[i]:.3f}        {event_times[i]:<12} "
                  f"{expected:<10} (actual: {actual}) {match}")
    
    return success


def validate_distribution(outcome, verbose=False):
    """Validate outcome distributions."""
    print("\n" + "="*70)
    print("DISTRIBUTION VALIDATION")
    print("="*70)
    
    event_mask = outcome.event_indicators == 1
    censored_mask = outcome.event_indicators == 0
    
    # Check event rate
    event_rate = outcome.event_indicators.float().mean().item()
    print(f"\nEvent rate: {event_rate:.1%}")
    
    if abs(event_rate - (1 - 0.3)) < 0.15:  # Within 15% of expected
        print(f"  ✓ Close to expected {1-0.3:.1%} (censoring_rate=0.3)")
    else:
        print(f"  ⚠️  Differs from expected {1-0.3:.1%}")
    
    # Check risk score distribution
    event_risk = outcome.risk_scores[event_mask].mean().item()
    censored_risk = outcome.risk_scores[censored_mask].mean().item() if censored_mask.sum() > 0 else 0
    
    print(f"\nRisk scores:")
    print(f"  • Events:   mean={event_risk:.3f}")
    print(f"  • Censored: mean={censored_risk:.3f}")
    
    if event_risk > censored_risk:
        print(f"  ✓ Events have higher risk (correct)")
    else:
        print(f"  ⚠️  Censored patients have higher risk (unexpected)")
    
    # Check time distribution
    event_times = outcome.event_times[event_mask]
    censored_times = outcome.event_times[censored_mask]
    
    if len(event_times) > 0:
        print(f"\nEvent times:")
        print(f"  • Mean: {event_times.float().mean():.1f} visits")
        print(f"  • Median: {event_times.float().median():.1f} visits")
        print(f"  • Range: [{event_times.min()}, {event_times.max()}]")
    
    if verbose and len(censored_times) > 0:
        print(f"\nCensoring times:")
        print(f"  • Mean: {censored_times.float().mean():.1f} visits")
        print(f"  • Median: {censored_times.float().median():.1f} visits")


def plot_results(outcome, sequences):
    """Plot outcome distributions."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('whitegrid')
    except ImportError:
        print("\n⚠️  Matplotlib not available, skipping plots")
        return
    
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    event_mask = outcome.event_indicators == 1
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Risk score distribution
    axes[0, 0].hist(outcome.risk_scores[event_mask].numpy(), bins=20, 
                    alpha=0.7, color='red', label='Events')
    axes[0, 0].hist(outcome.risk_scores[~event_mask].numpy(), bins=20,
                    alpha=0.7, color='gray', label='Censored')
    axes[0, 0].set_xlabel('Risk Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Risk Score Distribution')
    axes[0, 0].legend()
    
    # 2. Event time distribution
    axes[0, 1].hist(outcome.event_times[event_mask].numpy(), bins=20,
                    alpha=0.7, color='red', label='Events')
    axes[0, 1].hist(outcome.event_times[~event_mask].numpy(), bins=20,
                    alpha=0.7, color='gray', label='Censored')
    axes[0, 1].set_xlabel('Time (visits)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Event Time Distribution')
    axes[0, 1].legend()
    
    # 3. Risk vs Time scatter
    colors = ['red' if e == 1 else 'gray' for e in outcome.event_indicators]
    axes[1, 0].scatter(outcome.risk_scores.numpy(), outcome.event_times.numpy(),
                       c=colors, alpha=0.6, s=30)
    axes[1, 0].set_xlabel('Risk Score')
    axes[1, 0].set_ylabel('Event Time (visits)')
    axes[1, 0].set_title('Risk vs Event Time\n(Red=Event, Gray=Censored)')
    
    # Add correlation line for events only
    event_risk = outcome.risk_scores[event_mask].numpy()
    event_time = outcome.event_times[event_mask].numpy()
    if len(event_risk) > 1:
        z = np.polyfit(event_risk, event_time, 1)
        p = np.poly1d(z)
        x_line = np.linspace(event_risk.min(), event_risk.max(), 100)
        axes[1, 0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
                        label=f'Trend (events only)')
        axes[1, 0].legend()
    
    # 4. Sequence length distribution
    seq_lengths = [len(seq.visits) for seq in sequences]
    axes[1, 1].hist(seq_lengths, bins=20, alpha=0.7, color='blue')
    axes[1, 1].set_xlabel('Number of Visits')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Patient Sequence Lengths')
    
    plt.tight_layout()
    plt.savefig('synthetic_outcomes_validation.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved plot to: synthetic_outcomes_validation.png")
    plt.show()


def save_outcomes(outcome, sequences, filepath, args):
    """Save validated outcomes and sequences to file."""
    print(f"\nSaving outcomes to: {filepath}")
    
    # Save as a dictionary with all necessary data
    save_data = {
        'outcome': {
            'event_times': outcome.event_times,
            'event_indicators': outcome.event_indicators,
            'risk_scores': outcome.risk_scores,
            'metadata': outcome.metadata,
        },
        'sequences': sequences,
        'config': {
            'max_patients': args.max_patients,
            'censoring_rate': args.censoring_rate,
            'time_scale': args.time_scale,
            'seed': args.seed,
        },
        'num_patients': len(sequences),
    }
    
    torch.save(save_data, filepath)
    print(f"✓ Saved {len(sequences)} sequences with outcomes")


def main():
    """Main validation function."""
    args = parse_args()
    
    print("="*70)
    print("SYNTHETIC OUTCOME VALIDATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  • Max patients: {args.max_patients}")
    print(f"  • Censoring rate: {args.censoring_rate}")
    print(f"  • Time scale: {args.time_scale}")
    print(f"  • Seed: {args.seed}")
    
    # Load data
    sequences = load_data(args)
    
    # Generate outcomes
    outcome = generate_outcomes(args, sequences)
    
    # Validate correlation
    correlation_ok = validate_correlation(outcome, sequences, verbose=args.verbose)
    
    # Validate distributions
    validate_distribution(outcome, verbose=args.verbose)
    
    # Plot if requested
    if args.plot:
        plot_results(outcome, sequences)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if correlation_ok:
        print("\n✅ PASS: Synthetic outcomes look good!")
        print("   • Strong negative correlation between risk and event time")
        print("   • Ready to use for model training")
        
        # Save if requested and validation passed
        if args.save:
            save_outcomes(outcome, sequences, args.save, args)
            print(f"\n💡 To use in notebook:")
            print(f"   LOAD_PREGENERATED = '{args.save}'")
    else:
        print("\n❌ FAIL: Synthetic outcomes need improvement")
        print("   • Weak or incorrect correlation")
        print("   • Check generator parameters or implementation")
        print("\n💡 Suggestions:")
        print("   • Reduce noise_std in _simulate_event_time()")
        print("   • Adjust time_scale parameter")
        print("   • Check risk score normalization")
        
        if args.save:
            print(f"\n⚠️  Not saving outcomes (validation failed)")
    
    return 0 if correlation_ok else 1


if __name__ == '__main__':
    sys.exit(main())
