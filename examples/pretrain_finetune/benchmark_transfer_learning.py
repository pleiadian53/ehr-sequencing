"""
Benchmark: Transfer Learning Across Datasets

This script tests whether embeddings learned on one dataset transfer to another.
This is the proper test for pre-trained embedding quality.

EXPERIMENT DESIGN:
- Generate two synthetic datasets with different distributions (Dataset A & B)
- Dataset A: Source domain (e.g., 2010-2015 patients, different demographics)
- Dataset B: Target domain (e.g., 2016-2020 patients, different demographics)

4-WAY COMPARISON:
1. Train on A, test on A (source domain baseline)
2. Train on A, test on B (zero-shot transfer - no adaptation)
3. Train on A, fine-tune on B, test on B (transfer learning)
4. Train on B from scratch, test on B (target domain upper bound)

QUESTIONS ANSWERED:
- Do embeddings generalize across different data distributions?
- Is transfer learning better than training from scratch on limited data?
- How much does fine-tuning help vs zero-shot transfer?
- What's the performance gap between transfer and training from scratch?

This is the real test of whether pre-trained embeddings are valuable.

Usage:

# Full-scale test (A40 pod)
python benchmark_transfer_learning.py \
    --model-size large \
    --source-patients 10000 \
    --target-patients 5000 \
    --epochs 100 \
    --finetune-epochs 20 \
    --batch-size 128 \
    --output-dir experiments/transfer_learning

# Quick test (local)
python benchmark_transfer_learning.py \
    --model-size small \
    --source-patients 1000 \
    --target-patients 500 \
    --epochs 20 \
    --finetune-epochs 10 \
    --batch-size 32
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import argparse
import time
from typing import Dict, Tuple

from ehrsequencing.models.behrt import BEHRT, BEHRTConfig, BEHRTForMLM
from ehrsequencing.models.pretrained_embeddings import (
    save_embeddings, 
    load_embeddings,
    initialize_embedding_layer
)
from ehrsequencing.data.realistic_synthetic import generate_realistic_dataset, print_dataset_statistics
from ehrsequencing.benchmarks import (
    BenchmarkTracker,
    BenchmarkVisualizer,
    train_epoch,
    evaluate,
    compute_metrics
)


def generate_domain_shifted_datasets(
    source_patients: int,
    target_patients: int,
    vocab_size: int = 1000,
    source_seed: int = 42,
    target_seed: int = 123
) -> Tuple[Dict, Dict]:
    """
    Generate two datasets with REAL domain shift to test transfer learning.
    
    Domain Shift Strategy:
    - Source: Younger population (20-60), lower chronic disease prevalence
    - Target: Older population (50-90), higher chronic disease prevalence
    - Different code frequency distributions
    - Different temporal patterns
    
    This simulates real-world scenarios like:
    - Training on general population, deploying to elderly care
    - Training on one hospital system, deploying to another
    - Training on historical data, deploying to recent data
    
    Args:
        source_patients: Number of patients in source dataset
        target_patients: Number of patients in target dataset
        vocab_size: Vocabulary size
        source_seed: Random seed for source dataset
        target_seed: Random seed for target dataset
    
    Returns:
        Tuple of (source_data, target_data) dictionaries
    """
    print("\n" + "="*80)
    print("GENERATING DOMAIN-SHIFTED DATASETS")
    print("="*80)
    print("\n🔄 Domain Shift Strategy:")
    print("   Source: Younger population (20-60 yrs), lower chronic disease rates")
    print("   Target: Older population (50-90 yrs), higher chronic disease rates")
    print("   This simulates deploying a model trained on general population to elderly care")
    
    # Source domain: Younger, healthier population
    print(f"\n📊 Source Dataset (seed={source_seed}) - YOUNGER POPULATION:")
    print("   Age range: 20-60 years")
    print("   Disease prevalence: Lower (general population)")
    
    # Temporarily modify disease patterns for source domain
    from ehrsequencing.data.realistic_synthetic import DISEASE_PATTERNS
    original_patterns = {}
    for disease_name, pattern in DISEASE_PATTERNS.items():
        original_patterns[disease_name] = {
            'prevalence': pattern.prevalence,
            'age_range': pattern.age_range
        }
        # Reduce prevalence for younger population
        pattern.prevalence *= 0.6  # 40% reduction
        # Shift age range younger
        pattern.age_range = (max(20, pattern.age_range[0] - 15), min(60, pattern.age_range[1] - 20))
    
    codes_src, ages_src, visit_ids_src, attention_mask_src, masked_codes_src, labels_src = generate_realistic_dataset(
        num_patients=source_patients,
        vocab_size=vocab_size,
        max_seq_length=256,
        seed=source_seed
    )
    source_data = {
        'codes': codes_src,
        'ages': ages_src,
        'visit_ids': visit_ids_src,
        'attention_mask': attention_mask_src,
        'labels': labels_src
    }
    print_dataset_statistics(codes_src, ages_src, visit_ids_src)
    
    # Target domain: Older, sicker population
    print(f"\n📊 Target Dataset (seed={target_seed}) - OLDER POPULATION:")
    print("   Age range: 50-90 years")
    print("   Disease prevalence: Higher (elderly care)")
    
    # Modify disease patterns for target domain
    for disease_name, pattern in DISEASE_PATTERNS.items():
        # Increase prevalence for older population
        pattern.prevalence = original_patterns[disease_name]['prevalence'] * 1.8  # 80% increase
        # Shift age range older
        pattern.age_range = (max(50, pattern.age_range[0] + 20), min(90, pattern.age_range[1] + 15))
    
    codes_tgt, ages_tgt, visit_ids_tgt, attention_mask_tgt, masked_codes_tgt, labels_tgt = generate_realistic_dataset(
        num_patients=target_patients,
        vocab_size=vocab_size,
        max_seq_length=256,
        seed=target_seed
    )
    target_data = {
        'codes': codes_tgt,
        'ages': ages_tgt,
        'visit_ids': visit_ids_tgt,
        'attention_mask': attention_mask_tgt,
        'labels': labels_tgt
    }
    print_dataset_statistics(codes_tgt, ages_tgt, visit_ids_tgt)
    
    # Restore original patterns
    for disease_name, pattern in DISEASE_PATTERNS.items():
        pattern.prevalence = original_patterns[disease_name]['prevalence']
        pattern.age_range = original_patterns[disease_name]['age_range']
    
    print("\n✅ Domain shift created successfully!")
    print(f"   Expected transfer learning challenge: Medium-High")
    print(f"   Source and target have different age demographics and disease patterns")
    
    return source_data, target_data


def create_dataloaders(data: Dict, batch_size: int, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders from dataset."""
    codes = data['codes']
    ages = data['ages']
    visit_ids = data['visit_ids']
    attention_mask = data['attention_mask']
    labels = data['labels']
    
    # Split into train/val
    n_train = int(len(codes) * train_split)
    
    train_dataset = TensorDataset(
        codes[:n_train],
        ages[:n_train],
        visit_ids[:n_train],
        attention_mask[:n_train],
        labels[:n_train]
    )
    
    val_dataset = TensorDataset(
        codes[n_train:],
        ages[n_train:],
        visit_ids[n_train:],
        attention_mask[n_train:],
        labels[n_train:]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_with_early_stopping(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    tracker: BenchmarkTracker,
    vocab_size: int,
    patience: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Train model with early stopping and metric tracking."""
    print(f"\n{'='*80}")
    print(f"Training: {name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_probs, val_labels = evaluate(
            model, val_loader, device, return_predictions=True
        )
        
        tracker.log_epoch(name, epoch, train_loss, train_acc, val_loss, val_acc)
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            trophy = "🏆"
        else:
            patience_counter += 1
            trophy = ""
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} {trophy} | "
              f"Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    tracker.set_training_time(name, training_time)
    
    # Final evaluation
    print(f"\n📊 Computing final metrics for {name}...")
    _, _, final_probs, final_labels = evaluate(
        model, val_loader, device, return_predictions=True
    )
    metrics = compute_metrics(final_probs, final_labels, vocab_size)
    tracker.set_final_metrics(name, metrics)
    
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"   Avg Precision: {metrics['average_precision']:.4f}")
    
    return final_probs, final_labels


def main():
    parser = argparse.ArgumentParser(description='Benchmark: Transfer Learning Across Datasets')
    parser.add_argument('--model-size', type=str, default='large', choices=['small', 'medium', 'large'],
                       help='Model size')
    parser.add_argument('--source-patients', type=int, default=10000,
                       help='Number of patients in source dataset')
    parser.add_argument('--target-patients', type=int, default=5000,
                       help='Number of patients in target dataset')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs for source domain')
    parser.add_argument('--finetune-epochs', type=int, default=20,
                       help='Fine-tuning epochs on target domain')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='experiments/transfer_learning',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    vocab_size = 1000
    
    # Generate domain-shifted datasets
    source_data, target_data = generate_domain_shifted_datasets(
        source_patients=args.source_patients,
        target_patients=args.target_patients,
        vocab_size=vocab_size
    )
    
    # Create dataloaders
    source_train_loader, source_val_loader = create_dataloaders(source_data, args.batch_size)
    target_train_loader, target_val_loader = create_dataloaders(target_data, args.batch_size)
    
    # Initialize tracker
    tracker = BenchmarkTracker(output_dir=args.output_dir)
    
    # Model config
    if args.model_size == 'small':
        config = BEHRTConfig.small(vocab_size=vocab_size)
    elif args.model_size == 'medium':
        config = BEHRTConfig.medium(vocab_size=vocab_size)
    else:
        config = BEHRTConfig.large(vocab_size=vocab_size)
    
    print(f"\n🔧 Model Configuration: {args.model_size}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Embedding dim: {config.embedding_dim}")
    print(f"   Hidden dim: {config.hidden_dim}")
    print(f"   Layers: {config.num_layers}")
    print(f"   Heads: {config.num_heads}")
    
    # ============================================================================
    # RUN 1: Train on Source, Test on Source (baseline)
    # ============================================================================
    print("\n" + "="*80)
    print("RUN 1: Train on Source, Test on Source (Baseline)")
    print("="*80)
    
    model1 = BEHRTForMLM(config).to(device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=args.learning_rate)
    
    tracker.add_run('Source→Source', {
        'model_size': args.model_size,
        'dataset': 'source',
        'strategy': 'baseline',
        'trainable_params': f"{sum(p.numel() for p in model1.parameters() if p.requires_grad):,}"
    })
    
    train_with_early_stopping(
        'Source→Source', model1, source_train_loader, source_val_loader,
        optimizer1, device, args.epochs, tracker, vocab_size
    )
    
    # Save source model embeddings
    embedding_path = Path(args.output_dir) / 'source_embeddings.pt'
    embeddings = model1.behrt.embeddings.code_embedding.weight.data
    save_embeddings(
        embeddings, 
        embedding_path,
        metadata={'vocab_size': vocab_size, 'embedding_dim': config.embedding_dim}
    )
    print(f"\n💾 Saved source embeddings: {embedding_path}")
    
    # ============================================================================
    # RUN 2: Train on Source, Test on Target (zero-shot transfer)
    # ============================================================================
    print("\n" + "="*80)
    print("RUN 2: Train on Source, Test on Target (Zero-shot Transfer)")
    print("="*80)
    
    # Use model1 (trained on source) to evaluate on target
    print("\n📊 Evaluating source model on target domain (zero-shot)...")
    _, _, probs2, labels2 = evaluate(
        model1, target_val_loader, device, return_predictions=True
    )
    metrics2 = compute_metrics(probs2, labels2, vocab_size)
    
    tracker.add_run('Source→Target (zero-shot)', {
        'model_size': args.model_size,
        'dataset': 'target',
        'strategy': 'zero-shot',
        'trainable_params': '0 (frozen)'
    })
    tracker.set_training_time('Source→Target (zero-shot)', 0.0)
    tracker.set_final_metrics('Source→Target (zero-shot)', metrics2)
    
    print(f"   ROC-AUC: {metrics2['roc_auc']:.4f}")
    print(f"   PR-AUC: {metrics2['pr_auc']:.4f}")
    print(f"   Avg Precision: {metrics2['average_precision']:.4f}")
    
    # ============================================================================
    # RUN 3: Train on Source, Fine-tune on Target, Test on Target
    # ============================================================================
    print("\n" + "="*80)
    print("RUN 3: Train on Source, Fine-tune on Target (Transfer Learning)")
    print("="*80)
    
    model3 = BEHRTForMLM(config).to(device)
    pretrained_emb, _ = load_embeddings(embedding_path)
    initialize_embedding_layer(model3.behrt.embeddings.code_embedding, pretrained_emb, freeze=False)
    print(f"✅ Loaded and initialized source embeddings from {embedding_path}")
    
    optimizer3 = torch.optim.AdamW(model3.parameters(), lr=args.learning_rate * 0.1)  # Lower LR for fine-tuning
    
    tracker.add_run('Source→Target (fine-tuned)', {
        'model_size': args.model_size,
        'dataset': 'target',
        'strategy': 'transfer_learning',
        'trainable_params': f"{sum(p.numel() for p in model3.parameters() if p.requires_grad):,}"
    })
    
    train_with_early_stopping(
        'Source→Target (fine-tuned)', model3, target_train_loader, target_val_loader,
        optimizer3, device, args.finetune_epochs, tracker, vocab_size, patience=5
    )
    
    # ============================================================================
    # RUN 4: Train on Target from Scratch, Test on Target (upper bound)
    # ============================================================================
    print("\n" + "="*80)
    print("RUN 4: Train on Target from Scratch (Upper Bound)")
    print("="*80)
    
    model4 = BEHRTForMLM(config).to(device)
    optimizer4 = torch.optim.AdamW(model4.parameters(), lr=args.learning_rate)
    
    tracker.add_run('Target (from scratch)', {
        'model_size': args.model_size,
        'dataset': 'target',
        'strategy': 'from_scratch',
        'trainable_params': f"{sum(p.numel() for p in model4.parameters() if p.requires_grad):,}"
    })
    
    train_with_early_stopping(
        'Target (from scratch)', model4, target_train_loader, target_val_loader,
        optimizer4, device, args.epochs, tracker, vocab_size
    )
    
    # ============================================================================
    # Generate Summary
    # ============================================================================
    print("\n" + "="*80)
    print("GENERATING BENCHMARK SUMMARY")
    print("="*80)
    
    summary = tracker.generate_summary_table()
    
    # Use BenchmarkVisualizer for plots
    visualizer = BenchmarkVisualizer(output_dir=args.output_dir)
    visualizer.plot_all(tracker.get_all_runs())
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)
    print(f"\n📁 Results saved to: {args.output_dir}")
    print(f"   - SUMMARY.txt: Performance comparison table")
    print(f"   - summary.json: Machine-readable results")
    print(f"   - *.png: Visualization plots")
    print(f"   - source_embeddings.pt: Trained embeddings from source domain")


if __name__ == '__main__':
    main()
