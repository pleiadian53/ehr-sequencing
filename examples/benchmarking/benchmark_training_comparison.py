"""
Simple example demonstrating the reusable benchmarking utilities.

This script shows how to use the ehrsequencing.benchmarks module to:
1. Track multiple training runs
2. Compare their performance
3. Generate visualizations and reports

Usage:
    python examples/benchmarking/benchmark_training_comparison.py
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from ehrsequencing.models.behrt import BEHRTForMLM, BEHRTConfig
from ehrsequencing.models.lora import apply_lora_to_behrt
from ehrsequencing.synthetic.demo_data import generate_demo_dataset

# Import reusable benchmarking utilities
from ehrsequencing.benchmarks import (
    BenchmarkTracker,
    BenchmarkVisualizer,
    train_model,
    compute_roc_curve,
    compute_pr_curve
)


def main():
    print("="*80)
    print("Benchmarking Example: Comparing Training Configurations")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Generate data
    print("\n🔬 Generating synthetic data...")
    vocab_size = 1000
    num_patients = 2000
    
    codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_demo_dataset(
        num_patients=num_patients,
        vocab_size=vocab_size,
        max_seq_length=512,
        seed=42
    )
    
    # Create dataloaders
    train_size = int(0.7 * num_patients)
    val_size = int(0.15 * num_patients)
    
    train_dataset = TensorDataset(
        masked_codes[:train_size],
        ages[:train_size],
        visit_ids[:train_size],
        attention_mask[:train_size],
        labels[:train_size]
    )
    
    val_dataset = TensorDataset(
        masked_codes[train_size:train_size+val_size],
        ages[train_size:train_size+val_size],
        visit_ids[train_size:train_size+val_size],
        attention_mask[train_size:train_size+val_size],
        labels[train_size:train_size+val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"   Train: {len(train_dataset)} patients")
    print(f"   Val: {len(val_dataset)} patients")
    
    # Initialize benchmark tracker
    tracker = BenchmarkTracker(output_dir='examples/benchmarking/results/comparison')
    
    # Configuration 1: Small model with LoRA
    print("\n" + "="*80)
    print("Configuration 1: Small Model with LoRA")
    print("="*80)
    
    config_small = BEHRTConfig(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.2,
        max_position=512,
        num_age_bins=100,
        num_visit_bins=50,
        num_segments=2
    )
    
    model_small = BEHRTForMLM(config_small).to(device)
    model_small = apply_lora_to_behrt(model_small, rank=8, train_embeddings=True, train_head=True)
    
    optimizer_small = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_small.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    
    tracker.add_run('Small-LoRA', config={
        'model_size': 'small',
        'lora_rank': 8,
        'trainable_params': sum(p.numel() for p in model_small.parameters() if p.requires_grad)
    })
    
    probs_small, labels_small = train_model(
        name='Small-LoRA',
        model=model_small,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_small,
        device=device,
        epochs=30,
        tracker=tracker,
        vocab_size=vocab_size,
        patience=10
    )
    
    # Configuration 2: Medium model with LoRA
    print("\n" + "="*80)
    print("Configuration 2: Medium Model with LoRA")
    print("="*80)
    
    config_medium = BEHRTConfig(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.2,
        max_position=512,
        num_age_bins=100,
        num_visit_bins=50,
        num_segments=2
    )
    
    model_medium = BEHRTForMLM(config_medium).to(device)
    model_medium = apply_lora_to_behrt(model_medium, rank=16, train_embeddings=True, train_head=True)
    
    optimizer_medium = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_medium.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    
    tracker.add_run('Medium-LoRA', config={
        'model_size': 'medium',
        'lora_rank': 16,
        'trainable_params': sum(p.numel() for p in model_medium.parameters() if p.requires_grad)
    })
    
    probs_medium, labels_medium = train_model(
        name='Medium-LoRA',
        model=model_medium,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_medium,
        device=device,
        epochs=30,
        tracker=tracker,
        vocab_size=vocab_size,
        patience=10
    )
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations and Reports")
    print("="*80)
    
    visualizer = BenchmarkVisualizer(output_dir='examples/benchmarking/results/comparison')
    
    # Compute ROC and PR curves
    roc_data = {
        'Small-LoRA': {
            'fpr': compute_roc_curve(probs_small, labels_small, vocab_size)[0],
            'tpr': compute_roc_curve(probs_small, labels_small, vocab_size)[1],
            'auc': compute_roc_curve(probs_small, labels_small, vocab_size)[2]
        },
        'Medium-LoRA': {
            'fpr': compute_roc_curve(probs_medium, labels_medium, vocab_size)[0],
            'tpr': compute_roc_curve(probs_medium, labels_medium, vocab_size)[1],
            'auc': compute_roc_curve(probs_medium, labels_medium, vocab_size)[2]
        }
    }
    
    pr_data = {
        'Small-LoRA': {
            'precision': compute_pr_curve(probs_small, labels_small, vocab_size)[0],
            'recall': compute_pr_curve(probs_small, labels_small, vocab_size)[1],
            'auc': compute_pr_curve(probs_small, labels_small, vocab_size)[2]
        },
        'Medium-LoRA': {
            'precision': compute_pr_curve(probs_medium, labels_medium, vocab_size)[0],
            'recall': compute_pr_curve(probs_medium, labels_medium, vocab_size)[1],
            'auc': compute_pr_curve(probs_medium, labels_medium, vocab_size)[2]
        }
    }
    
    # Generate all plots
    visualizer.plot_all(tracker.get_all_runs(), roc_data=roc_data, pr_data=pr_data)
    
    # Generate summary table
    summary = tracker.generate_summary_table()
    
    print("\n" + "="*80)
    print("✅ Benchmark Complete!")
    print("="*80)
    print(f"\n📁 Results saved to: examples/benchmarking/results/comparison/")
    print("\nFiles generated:")
    print("  - SUMMARY.txt          # Text summary table")
    print("  - summary.json         # JSON summary")
    print("  - summary.csv          # CSV summary")
    print("  - training_curves.png  # Training/validation curves")
    print("  - performance_metrics.png  # Performance bar chart")
    print("  - roc_curves.png       # ROC curves")
    print("  - pr_curves.png        # Precision-Recall curves")
    print("  - convergence_*.png    # Convergence plots")
    print("  - training_time.png    # Training time comparison")


if __name__ == '__main__':
    main()
