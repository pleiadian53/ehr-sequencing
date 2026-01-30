"""
BEHRT Pre-training Demo Script

Demonstrates:
1. BEHRT architecture with 3 size configs (small/medium/large)
2. LoRA for efficient fine-tuning
3. Comprehensive experiment tracking for ephemeral pods
4. MLM pre-training objective
5. Checkpointing and visualization

This script is designed for quick testing and demonstration.
For production training, use train_behrt.py

Usage:

# Test locally (M1 16GB) with LoRA
python examples/pretrain_finetune/train_behrt_demo.py \
    --model_size small \
    --use_lora \
    --lora_rank 8 \
    --num_patients 100 \
    --epochs 10

# Train on A40 pod
python examples/pretrain_finetune/train_behrt_demo.py \
    --model_size large \
    --use_lora \
    --lora_rank 16 \
    --num_patients 5000 \
    --epochs 100 \
    --batch_size 128
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
from ehrsequencing.data.realistic_synthetic import generate_realistic_dataset, print_dataset_statistics


def generate_synthetic_data(
    num_patients: int = 100,
    vocab_size: int = 1000,
    max_seq_length: int = 50,
    mask_prob: float = 0.15
):
    """
    Generate synthetic EHR data for testing.
    
    Returns:
        codes: Medical code IDs [num_patients, max_seq_length]
        ages: Patient ages [num_patients, max_seq_length]
        visit_ids: Visit IDs [num_patients, max_seq_length]
        attention_mask: Valid positions [num_patients, max_seq_length]
        masked_codes: Codes with masking applied
        labels: Original codes for masked positions
    """
    print(f"Generating synthetic data: {num_patients} patients, vocab={vocab_size}")
    
    codes = torch.randint(1, vocab_size, (num_patients, max_seq_length))
    ages = torch.randint(20, 80, (num_patients, max_seq_length))
    visit_ids = torch.arange(max_seq_length).unsqueeze(0).expand(num_patients, -1)
    attention_mask = torch.ones(num_patients, max_seq_length, dtype=torch.bool)
    
    masked_codes = codes.clone()
    labels = torch.full_like(codes, -100)
    
    mask_token_id = 0
    mask = torch.rand(num_patients, max_seq_length) < mask_prob
    labels[mask] = codes[mask]
    masked_codes[mask] = mask_token_id
    
    return codes, ages, visit_ids, attention_mask, masked_codes, labels


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


def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    with torch.no_grad():
        for batch in dataloader:
            masked_codes, ages, visit_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            logits, loss = model(masked_codes, ages, visit_ids, attention_mask, labels)
            
            total_loss += loss.item()
            
            mask = labels != -100
            if mask.any():
                predictions = logits.argmax(dim=-1)
                total_correct += (predictions[mask] == labels[mask]).sum().item()
                total_masked += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='BEHRT Pre-training Demo')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large'],
                       help='Model size (small for local, large for pod)')
    parser.add_argument('--use_lora', action='store_true',
                       help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank (lower = fewer parameters)')
    parser.add_argument('--num_patients', type=int, default=100,
                       help='Number of synthetic patients')
    parser.add_argument('--vocab_size', type=int, default=1000,
                       help='Vocabulary size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
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
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory')
    
    args = parser.parse_args()
    
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
    if args.realistic_data:
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
        codes, ages, visit_ids, attention_mask, masked_codes, labels = generate_synthetic_data(
            num_patients=args.num_patients,
            vocab_size=args.vocab_size,
            max_seq_length=config.max_position
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
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        
        tracker.log_metrics(epoch, {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if args.use_lora:
            tracker.save_lora_checkpoint(model, epoch, 
                                        {'val_loss': val_loss, 'val_acc': val_acc},
                                        is_best=is_best)
        else:
            tracker.save_checkpoint(model, optimizer, epoch,
                                   {'val_loss': val_loss, 'val_acc': val_acc},
                                   is_best=is_best)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
              f"{' 🏆' if is_best else ''}"
              f" | Patience: {patience_counter}/{args.early_stopping_patience}")
        
        if patience_counter >= args.early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            print(f"   Best val loss: {best_val_loss:.4f} at epoch {epoch+1-patience_counter}")
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
