"""
BEHRT Fine-tuning Script with Pre-trained Embeddings

This script demonstrates fine-tuning BEHRT with pre-trained medical code embeddings.
Use this when you have pre-trained embeddings (e.g., from Med2Vec, Word2Vec) and want
to fine-tune on a specific task with limited data.

Key differences from train_behrt_demo.py:
- Loads pre-trained embeddings and freezes them
- Requires fewer patients (1K-10K vs 100K+)
- Only trains LoRA adapters + task head
- Faster convergence due to pre-trained representations

Workflow:
1. Pre-train embeddings (Phase 2: Med2Vec) OR use existing embeddings
2. Fine-tune BEHRT with frozen embeddings (this script)
3. Evaluate on downstream tasks

Usage:

# Fine-tune with Med2Vec embeddings
python examples/pretrain_finetune/train_behrt_finetune.py \
    --model_size large \
    --embedding_path pretrained/med2vec_embeddings.pt \
    --use_lora \
    --lora_rank 16 \
    --num_patients 2000 \
    --epochs 50

# Fine-tune with realistic synthetic data
python examples/pretrain_finetune/train_behrt_finetune.py \
    --model_size medium \
    --embedding_path pretrained/behrt_embeddings.pt \
    --realistic_data \
    --num_patients 5000 \
    --epochs 100
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import argparse

from ehrsequencing.models.behrt import BEHRT, BEHRTConfig, BEHRTForMLM
from ehrsequencing.models.lora import apply_lora_to_behrt, count_parameters
from ehrsequencing.models.pretrained_embeddings import (
    load_embeddings,
    initialize_embedding_layer,
    print_embedding_statistics
)
from ehrsequencing.utils.experiment_tracker import ExperimentTracker
from ehrsequencing.data.realistic_synthetic import generate_realistic_dataset, print_dataset_statistics


def generate_synthetic_data(
    num_patients: int = 100,
    vocab_size: int = 1000,
    max_seq_length: int = 50,
    mask_prob: float = 0.15
):
    """Generate random synthetic EHR data for testing (fallback)."""
    print(f"Generating synthetic data: {num_patients} patients, vocab={vocab_size}")
    
    codes = torch.randint(1, vocab_size, (num_patients, max_seq_length))
    ages = torch.randint(20, 80, (num_patients, max_seq_length))
    visit_ids = torch.arange(max_seq_length).unsqueeze(0).repeat(num_patients, 1)
    attention_mask = torch.ones(num_patients, max_seq_length)
    
    masked_codes = codes.clone()
    labels = torch.full((num_patients, max_seq_length), -100)
    
    mask = torch.rand(num_patients, max_seq_length) < mask_prob
    labels[mask] = codes[mask]
    
    rand_mask = torch.rand(num_patients, max_seq_length)
    masked_codes[mask & (rand_mask < 0.8)] = vocab_size - 1
    masked_codes[mask & (rand_mask >= 0.8) & (rand_mask < 0.9)] = torch.randint(1, vocab_size - 1, (mask.sum(),))
    
    return codes, ages, visit_ids, attention_mask, masked_codes, labels


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    for batch in dataloader:
        codes, ages, visit_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(codes, ages=ages, visit_ids=visit_ids, attention_mask=attention_mask)
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        mask = labels != -100
        predictions = outputs.argmax(dim=-1)
        total_correct += (predictions[mask] == labels[mask]).sum().item()
        total_masked += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    with torch.no_grad():
        for batch in dataloader:
            codes, ages, visit_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(codes, ages=ages, visit_ids=visit_ids, attention_mask=attention_mask)
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            
            mask = labels != -100
            predictions = outputs.argmax(dim=-1)
            total_correct += (predictions[mask] == labels[mask]).sum().item()
            total_masked += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='BEHRT Fine-tuning with Pre-trained Embeddings')
    parser.add_argument('--model_size', type=str, default='medium', choices=['small', 'medium', 'large'],
                       help='Model size')
    parser.add_argument('--embedding_path', type=str, required=True,
                       help='Path to pre-trained embeddings (.pt file)')
    parser.add_argument('--use_lora', action='store_true',
                       help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--num_patients', type=int, default=2000,
                       help='Number of patients (can be smaller with pre-trained embeddings)')
    parser.add_argument('--vocab_size', type=int, default=1000,
                       help='Vocabulary size (must match pre-trained embeddings)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout probability')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--realistic_data', action='store_true',
                       help='Use realistic synthetic data with disease patterns')
    parser.add_argument('--freeze_embeddings', action='store_true', default=True,
                       help='Freeze pre-trained embeddings (recommended)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pre-trained embeddings
    print(f"\n📂 Loading pre-trained embeddings from {args.embedding_path}")
    pretrained_embeddings, metadata = load_embeddings(args.embedding_path)
    print_embedding_statistics(pretrained_embeddings, "Pre-trained Embeddings")
    
    # Validate embedding dimensions
    if metadata:
        if 'vocab_size' in metadata and metadata['vocab_size'] != args.vocab_size:
            print(f"⚠️  Warning: vocab_size mismatch. Using {metadata['vocab_size']} from embeddings.")
            args.vocab_size = metadata['vocab_size']
        if 'embedding_dim' in metadata:
            embedding_dim = metadata['embedding_dim']
        else:
            embedding_dim = pretrained_embeddings.shape[1]
    else:
        embedding_dim = pretrained_embeddings.shape[1]
    
    if args.experiment_name is None:
        lora_suffix = f"_lora{args.lora_rank}" if args.use_lora else ""
        args.experiment_name = f"behrt_{args.model_size}_finetune{lora_suffix}"
    
    tracker = ExperimentTracker(args.experiment_name, output_dir=args.output_dir)
    
    print("\n" + "="*80)
    print(f"BEHRT Fine-tuning: {args.experiment_name}")
    print("="*80)
    print(f"🔧 Using pre-trained embeddings (frozen: {args.freeze_embeddings})")
    
    # Create model config
    if args.model_size == 'small':
        config = BEHRTConfig.small(vocab_size=args.vocab_size)
    elif args.model_size == 'medium':
        config = BEHRTConfig.medium(vocab_size=args.vocab_size)
    else:
        config = BEHRTConfig.large(vocab_size=args.vocab_size)
    
    # Override embedding_dim to match pre-trained embeddings
    config.embedding_dim = embedding_dim
    config.dropout = args.dropout
    
    tracker.log_hyperparameters({
        'model_size': args.model_size,
        'vocab_size': config.vocab_size,
        'embedding_dim': config.embedding_dim,
        'hidden_dim': config.hidden_dim,
        'num_layers': config.num_layers,
        'num_heads': config.num_heads,
        'dropout': args.dropout,
        'use_lora': args.use_lora,
        'lora_rank': args.lora_rank if args.use_lora else None,
        'freeze_embeddings': args.freeze_embeddings,
        'num_patients': args.num_patients,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'device': str(device)
    })
    
    model = BEHRTForMLM(config).to(device)
    
    # Initialize code embeddings with pre-trained weights
    print(f"\n🔧 Initializing model with pre-trained embeddings...")
    initialize_embedding_layer(
        model.behrt.embeddings.code_embedding,
        pretrained_embeddings,
        freeze=args.freeze_embeddings
    )
    
    if args.use_lora:
        print(f"\n🔧 Applying LoRA (rank={args.lora_rank})...")
        model = apply_lora_to_behrt(
            model,
            rank=args.lora_rank,
            lora_attention=True,
            train_embeddings=not args.freeze_embeddings,  # Only train if not frozen
            train_head=True  # Always train task head
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
    
    print(f"\n🔬 Generating data...")
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
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    print(f"\n🚀 Starting fine-tuning...")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Early stopping patience: {args.early_stopping_patience} epochs")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        tracker.log_metrics({
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            if args.use_lora:
                tracker.save_lora_checkpoint(model, f'best_lora.pt')
            else:
                tracker.save_checkpoint(model, 'best_model.pt')
            trophy = "🏆"
        else:
            patience_counter += 1
            trophy = ""
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} {trophy} | "
              f"Patience: {patience_counter}/{args.early_stopping_patience}")
        
        if patience_counter >= args.early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            print(f"   Best val loss: {best_val_loss:.4f} at epoch {epoch+1-patience_counter}")
            break
    
    print("\n📈 Generating plots...")
    tracker.plot_training_curves()
    
    print("\n💾 Saving final summary...")
    tracker.save_summary()
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"📁 All outputs saved to: {tracker.output_dir}")
    print(f"⏱️  Duration: {tracker.get_duration():.2f} hours")
    print(f"🏆 Best metrics:")
    for key, value in tracker.best_metrics.items():
        print(f"   {key}: {value:.4f}")


if __name__ == '__main__':
    main()
