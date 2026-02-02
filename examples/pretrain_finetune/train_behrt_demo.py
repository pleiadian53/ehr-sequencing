"""
BEHRT Pre-training Demo Script

Demonstrates:
1. BEHRT architecture with 3 size configs (small/medium/large)
2. LoRA for efficient fine-tuning (enabled by default)
3. Comprehensive experiment tracking for ephemeral pods
4. MLM pre-training objective
5. Checkpointing and visualization
6. Comprehensive metrics (Accuracy, Top-5, F1, Precision, Recall, Perplexity)
7. **Auto resource detection** - automatically optimizes parameters for your hardware

Auto Resource Detection (NEW!):
- Detects GPU type (A40, A100, V100, T4, local GPU, or CPU)
- Detects VRAM capacity and system RAM
- Automatically sets optimal: model_size, batch_size, num_patients, epochs, lora_rank
- User can override any parameter via command-line
- Enabled by default, disable with --no_auto_resources

Supported Platforms:
- Local CPU (small model, 100 patients, batch 4)
- Local Laptop GPU (small model, 500 patients, batch 16)
- Local Workstation (medium model, 2000 patients, batch 64)
- Cloud T4 (medium model, 3000 patients, batch 64)
- Cloud V100 (large model, 5000 patients, batch 96)
- Cloud A40 (large model, 5000 patients, batch 128)
- Cloud A100 (large model, 10000 patients, batch 256)

Data Options:
- Random data (default): For quick syntax testing only (~0.1% accuracy)
- Realistic data (--realistic_data): Realistic patterns (~30-60% accuracy)
- Demo data (--demo_data): Very strong patterns for compelling demos (~70-85% accuracy)

Metrics:
- Accuracy: Standard accuracy (can be misleading for imbalanced data)
- Top-5 Accuracy: Is correct code in top 5 predictions? (more forgiving)
- Macro F1: F1 averaged across all codes (treats rare codes equally)
- Weighted F1: F1 weighted by code frequency
- Perplexity: Exp(cross-entropy loss)

Usage:

# Auto-detect resources (RECOMMENDED - works anywhere!)
python examples/pretrain_finetune/train_behrt_demo.py --demo_data

# Auto-detect with realistic data
python examples/pretrain_finetune/train_behrt_demo.py --realistic_data

# Override specific parameters (auto-detect fills the rest)
python examples/pretrain_finetune/train_behrt_demo.py \
    --demo_data \
    --batch_size 64 \
    --epochs 50

# Force specific model size (auto-detect adjusts other params)
python examples/pretrain_finetune/train_behrt_demo.py \
    --demo_data \
    --model_size large

# Disable auto-detection (use fixed defaults)
python examples/pretrain_finetune/train_behrt_demo.py \
    --no_auto_resources \
    --model_size large \
    --demo_data

# Train full model without LoRA
python examples/pretrain_finetune/train_behrt_demo.py \
    --demo_data \
    --no_lora
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import argparse

from ehrsequencing.models.behrt import BEHRT, BEHRTConfig, BEHRTForMLM
from ehrsequencing.models.lora import apply_lora_to_behrt, count_parameters
from ehrsequencing.utils.experiment_tracker import ExperimentTracker
from ehrsequencing.data import (
    generate_realistic_dataset,
    print_dataset_statistics,
    generate_demo_dataset,
    print_demo_dataset_statistics,
    generate_random_dataset
)
from ehrsequencing.utils.metrics import compute_mlm_metrics, print_metrics_summary, get_metrics_for_logging
from ehrsequencing.utils.resource_manager import get_recommended_config


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    for batch in dataloader:
        masked_codes, ages, visit_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        logits, loss = model(masked_codes, ages, visit_ids, attention_mask, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        mask = labels != -100
        if mask.any():
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions[mask] == labels[mask]).sum().item()
            total_masked += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    
    return avg_loss, accuracy


def validate(model, dataloader, device, vocab_size):
    """Validate model with comprehensive metrics."""
    model.eval()
    total_loss = 0
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            masked_codes, ages, visit_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            logits, loss = model(masked_codes, ages, visit_ids, attention_mask, labels)
            
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    avg_loss = total_loss / len(dataloader)
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute comprehensive metrics
    metrics = compute_mlm_metrics(all_logits, all_labels, vocab_size, top_k=5)
    metrics['loss'] = avg_loss
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='BEHRT Pre-training Demo')
    
    # Resource management
    parser.add_argument('--auto_resources', action='store_true', default=True,
                       help='Auto-detect resources and set optimal defaults (default: True)')
    parser.add_argument('--no_auto_resources', action='store_true',
                       help='Disable auto resource detection, use fixed defaults')
    
    # Model configuration
    parser.add_argument('--model_size', type=str, default=None, choices=['small', 'medium', 'large'],
                       help='Model size (auto-detected if not specified)')
    parser.add_argument('--use_lora', action='store_true', default=None,
                       help='Use LoRA for efficient fine-tuning (auto-detected if not specified)')
    parser.add_argument('--no_lora', action='store_true',
                       help='Disable LoRA (train full model)')
    parser.add_argument('--lora_rank', type=int, default=None,
                       help='LoRA rank (auto-detected if not specified)')
    
    # Training parameters
    parser.add_argument('--num_patients', type=int, default=None,
                       help='Number of synthetic patients (auto-detected if not specified)')
    parser.add_argument('--vocab_size', type=int, default=1000,
                       help='Vocabulary size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (auto-detected if not specified)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (auto-detected if not specified)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for AdamW optimizer')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout probability')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--realistic_data', action='store_true',
                       help='Use realistic synthetic data with disease patterns (recommended for showcasing)')
    parser.add_argument('--demo_data', action='store_true',
                       help='Use high-signal demo data with very strong patterns (70%+ accuracy, best for demos)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Handle --no_auto_resources flag
    if args.no_auto_resources:
        args.auto_resources = False
    
    # Auto-detect resources and set defaults for unspecified parameters
    if args.auto_resources:
        # Determine task type from data flags
        if args.demo_data:
            task = 'demo'
        elif args.realistic_data:
            task = 'realistic'
        else:
            task = 'demo'  # Default to demo
        
        # Get recommended config
        recommended_config, resources = get_recommended_config(
            task=task,
            model_size_override=args.model_size,
            verbose=True
        )
        
        # Fill in None parameters with recommendations
        if args.model_size is None:
            args.model_size = recommended_config.model_size
        if args.batch_size is None:
            args.batch_size = recommended_config.batch_size
        if args.num_patients is None:
            args.num_patients = recommended_config.num_patients
        if args.epochs is None:
            args.epochs = recommended_config.epochs
        if args.use_lora is None:
            args.use_lora = recommended_config.use_lora
        if args.lora_rank is None:
            args.lora_rank = recommended_config.lora_rank
    else:
        # Use fixed defaults when auto-detection is disabled
        if args.model_size is None:
            args.model_size = 'large'
        if args.batch_size is None:
            args.batch_size = 128
        if args.num_patients is None:
            args.num_patients = 5000
        if args.epochs is None:
            args.epochs = 100
        if args.use_lora is None:
            args.use_lora = True
        if args.lora_rank is None:
            args.lora_rank = 16
    
    # Handle --no_lora flag to override
    if args.no_lora:
        args.use_lora = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.experiment_name is None:
        lora_suffix = f"_lora{args.lora_rank}" if args.use_lora else ""
        args.experiment_name = f"behrt_{args.model_size}_mlm{lora_suffix}"
    
    tracker = ExperimentTracker(args.experiment_name, output_dir=args.output_dir)
    
    print("\n" + "="*80)
    print(f"BEHRT Pre-training Demo: {args.experiment_name}")
    print("="*80)
    
    if args.model_size == 'small':
        config = BEHRTConfig.small(vocab_size=args.vocab_size)
        config.dropout = args.dropout
        print("📱 Small model (for M1 MacBook Pro 16GB)")
    elif args.model_size == 'medium':
        config = BEHRTConfig.medium(vocab_size=args.vocab_size)
        config.dropout = args.dropout
        print("💻 Medium model (for local/small GPU)")
    else:
        config = BEHRTConfig.large(vocab_size=args.vocab_size)
        config.dropout = args.dropout
        print("☁️  Large model (for A40 cloud GPU)")
    
    tracker.log_hyperparameters({
        'model_size': args.model_size,
        'vocab_size': args.vocab_size,
        'embedding_dim': config.embedding_dim,
        'hidden_dim': config.hidden_dim,
        'num_layers': config.num_layers,
        'num_heads': config.num_heads,
        'dropout': args.dropout,
        'use_lora': args.use_lora,
        'lora_rank': args.lora_rank if args.use_lora else None,
        'num_patients': args.num_patients,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'device': str(device)
    })
    
    model = BEHRTForMLM(config).to(device)
    
    if args.use_lora:
        print(f"\n🔧 Applying LoRA (rank={args.lora_rank})...")
        # Apply LoRA to the full model (not just behrt) so embeddings and MLM head are handled
        model = apply_lora_to_behrt(
            model, 
            rank=args.lora_rank, 
            lora_attention=True,
            train_embeddings=True,  # Critical: embeddings must be trainable when training from scratch
            train_head=True         # Critical: MLM head must be trainable
        )
    
    param_counts = count_parameters(model)
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {param_counts['total']:,}")
    print(f"   Trainable: {param_counts['trainable']:,} ({param_counts['trainable_percent']:.1f}%)")
    print(f"   Frozen: {param_counts['frozen']:,}")
    if args.use_lora:
        print(f"   LoRA: {param_counts['lora']:,} ({param_counts['lora_percent']:.1f}%)")
    print(f"   Embeddings: {param_counts['embedding_trainable']:,}/{param_counts['embedding_total']:,} trainable")
    print(f"   Head: {param_counts['head_trainable']:,}/{param_counts['head_total']:,} trainable")
    
    tracker.log_metadata({
        'total_parameters': param_counts['total'],
        'trainable_parameters': param_counts['trainable'],
        'trainable_percent': param_counts['trainable_percent']
    })
    
    print(f"\n🔬 Generating synthetic data...")
    if args.demo_data:
        print("Using HIGH-SIGNAL demo data with very strong patterns (70%+ accuracy expected)...")
        codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_demo_dataset(
            num_patients=args.num_patients,
            vocab_size=args.vocab_size,
            max_seq_length=config.max_position,
            seed=42
        )
        print_demo_dataset_statistics(codes, ages, visit_ids)
    elif args.realistic_data:
        print("Using realistic synthetic data with disease patterns...")
        codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_realistic_dataset(
            num_patients=args.num_patients,
            vocab_size=args.vocab_size,
            max_seq_length=config.max_position,
            seed=42
        )
        print_dataset_statistics(codes, ages, visit_ids)
    else:
        print("Using random synthetic data (for testing only)...")
        codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_random_dataset(
            num_patients=args.num_patients,
            vocab_size=args.vocab_size,
            max_seq_length=config.max_position,
            seed=42
        )
    
    train_size = int(0.8 * args.num_patients)
    train_dataset = TensorDataset(
        masked_codes[:train_size], ages[:train_size], visit_ids[:train_size],
        attention_mask[:train_size], labels[:train_size]
    )
    val_dataset = TensorDataset(
        masked_codes[train_size:], ages[train_size:], visit_ids[train_size:],
        attention_mask[train_size:], labels[train_size:]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print(f"\n🚀 Starting training...")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Early stopping patience: {args.early_stopping_patience} epochs")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device, args.vocab_size)
        
        # Extract key metrics
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_top5 = val_metrics['top_5_accuracy']
        val_f1 = val_metrics['macro_f1']
        
        # Log all metrics
        tracker.log_metrics(epoch, {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_top_5_accuracy': val_top5,
            'val_macro_f1': val_f1,
            'val_weighted_f1': val_metrics['weighted_f1'],
            'val_perplexity': val_metrics['perplexity']
        })
        
        # Check if this is a significant improvement (>0.5% relative improvement)
        is_best = val_loss < best_val_loss
        is_significant = False
        if is_best:
            improvement = (best_val_loss - val_loss) / best_val_loss if best_val_loss != float('inf') else 1.0
            is_significant = improvement > 0.005  # 0.5% improvement threshold
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        
        if args.use_lora:
            tracker.save_lora_checkpoint(model, epoch, 
                                        {'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1},
                                        is_best=is_best)
        else:
            tracker.save_checkpoint(model, optimizer, epoch,
                                   {'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1},
                                   is_best=is_best)
        
        # Show trophy only on significant improvements
        status_icon = ' 🏆' if is_significant else (' ✓' if is_best else '')
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Top5: {val_top5:.4f} F1: {val_f1:.4f}"
              f"{status_icon}"
              f" | Patience: {patience_counter}/{args.early_stopping_patience}")
        
        if patience_counter >= args.early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            print(f"   Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break
    
    print(f"\n📈 Generating plots...")
    tracker.plot_training_curves()
    
    print(f"\n💾 Saving final summary...")
    tracker.save_summary()
    
    print(f"\n✅ Training complete!")
    print(f"📁 All outputs saved to: {tracker.output_dir}")
    print(f"\nKey files:")
    print(f"   - Best model: {tracker.output_dir}/checkpoints/best_{'lora' if args.use_lora else 'model'}.pt")
    print(f"   - Training curves: {tracker.output_dir}/plots/")
    print(f"   - Summary: {tracker.output_dir}/SUMMARY.txt")


if __name__ == '__main__':
    main()
