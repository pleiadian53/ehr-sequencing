"""
Training script for BEHRTForSurvival model.

Supports three loss functions:
1. NLL (standard) - Optimizes calibration
2. Pairwise Ranking - Directly optimizes C-index
3. Hybrid (NLL + Ranking) - Best of both worlds

Usage:
    # NLL loss (standard)
    python train_behrt_survival.py --loss nll --epochs 100
    
    # Pairwise ranking loss (direct C-index optimization)
    python train_behrt_survival.py --loss ranking --margin 0.1 --epochs 100
    
    # Hybrid loss
    python train_behrt_survival.py --loss hybrid --lambda-rank 0.1 --epochs 100
"""

import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from ehrsequencing.models.behrt_survival import BEHRTForSurvival, BEHRTSurvivalConfig
from ehrsequencing.models.behrt import BEHRTForMLM
from ehrsequencing.models.losses import (
    DiscreteTimeSurvivalLoss,
    PairwiseRankingLoss,
    HybridSurvivalLoss,
    concordance_index
)
from ehrsequencing.data.behrt_survival_dataset import (
    BEHRTSurvivalDataset,
    collate_behrt_survival,
    prepare_behrt_survival_data
)
from ehrsequencing.synthetic.realistic_synthetic import generate_realistic_dataset
from ehrsequencing.synthetic.survival import DiscreteTimeSurvivalGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Train BEHRTForSurvival')
    
    # Data
    parser.add_argument('--num-patients', type=int, default=5000,
                        help='Number of synthetic patients')
    parser.add_argument('--vocab-size', type=int, default=1000,
                        help='Vocabulary size')
    parser.add_argument('--max-seq-length', type=int, default=512,
                        help='Maximum sequence length')
    
    # Model
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'medium', 'large'],
                        help='BEHRT model size')
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='Path to pre-trained BEHRT checkpoint')
    parser.add_argument('--freeze-behrt', action='store_true',
                        help='Freeze BEHRT encoder (train only head)')
    parser.add_argument('--use-lora', action='store_true',
                        help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora-rank', type=int, default=16,
                        help='LoRA rank')
    
    # Loss function
    parser.add_argument('--loss', type=str, default='nll',
                        choices=['nll', 'ranking', 'hybrid'],
                        help='Loss function type')
    parser.add_argument('--lambda-rank', type=float, default=0.1,
                        help='Weight for ranking loss in hybrid')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Margin for ranking loss')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='experiments/behrt_survival',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(args) -> Tuple[list, int]:
    """Generate synthetic EHR data with survival outcomes."""
    print(f"\n{'='*80}")
    print("Generating synthetic data...")
    print(f"{'='*80}")
    
    # Generate realistic sequences
    codes, ages, visit_ids, attention_mask, _, _ = generate_realistic_dataset(
        num_patients=args.num_patients,
        vocab_size=args.vocab_size,
        max_seq_length=args.max_seq_length,
        seed=args.seed
    )
    
    # Convert to visit-grouped format
    patient_sequences = []
    for i in range(args.num_patients):
        # Get codes and ages for this patient
        patient_codes = codes[i].tolist()
        patient_ages = ages[i].tolist()
        patient_visit_ids = visit_ids[i].tolist()
        patient_mask = attention_mask[i].tolist()
        
        # Group by visit
        visits = []
        max_visit_id = max(patient_visit_ids)
        for v in range(max_visit_id + 1):
            visit_codes = [
                c for c, vid, m in zip(patient_codes, patient_visit_ids, patient_mask)
                if vid == v and m == 1
            ]
            visit_age = [
                a for a, vid, m in zip(patient_ages, patient_visit_ids, patient_mask)
                if vid == v and m == 1
            ][0] if any(vid == v and m == 1 for vid, m in zip(patient_visit_ids, patient_mask)) else 0
            
            if visit_codes:
                visits.append({'codes': visit_codes, 'age': visit_age})
        
        patient_sequences.append({'visits': visits})
    
    # Generate survival outcomes
    generator = DiscreteTimeSurvivalGenerator(
        risk_correlation=-0.5,
        censoring_rate=0.3,
        time_scale=10,
        seed=args.seed
    )
    
    outcomes = generator.generate_outcomes(patient_sequences)
    
    # Add outcomes to sequences
    for seq, outcome in zip(patient_sequences, outcomes):
        seq['outcome'] = outcome
    
    print(f"Generated {len(patient_sequences)} patients")
    print(f"Average visits per patient: {np.mean([len(s['visits']) for s in patient_sequences]):.1f}")
    print(f"Event rate: {np.mean([s['outcome']['event_indicator'] for s in patient_sequences]):.2%}")
    
    return patient_sequences, args.vocab_size


def create_model(args, vocab_size: int) -> BEHRTForSurvival:
    """Create BEHRTForSurvival model."""
    print(f"\n{'='*80}")
    print("Creating model...")
    print(f"{'='*80}")
    
    # Create config
    if args.model_size == 'small':
        config = BEHRTSurvivalConfig.from_pretrained_small(
            vocab_size=vocab_size,
            freeze_behrt=args.freeze_behrt
        )
    elif args.model_size == 'medium':
        config = BEHRTSurvivalConfig.from_pretrained_medium(
            vocab_size=vocab_size,
            freeze_behrt=args.freeze_behrt
        )
    else:
        config = BEHRTSurvivalConfig.from_pretrained_large(
            vocab_size=vocab_size,
            freeze_behrt=args.freeze_behrt
        )
    
    # Add LoRA config
    if args.use_lora:
        config.use_lora = True
        config.lora_rank = args.lora_rank
    
    # Load pre-trained BEHRT if specified
    pretrained_behrt = None
    if args.pretrained_path:
        print(f"Loading pre-trained BEHRT from {args.pretrained_path}")
        pretrained_model = BEHRTForMLM.from_pretrained(args.pretrained_path)
        pretrained_behrt = pretrained_model.behrt
    
    # Create model
    model = BEHRTForSurvival(config, pretrained_behrt=pretrained_behrt)
    
    # Print parameter counts
    param_counts = model.get_trainable_parameters()
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,} ({param_counts['trainable_percentage']:.1f}%)")
    print(f"BEHRT trainable: {param_counts['behrt_trainable']:,}")
    print(f"Head trainable: {param_counts['head_trainable']:,}")
    
    return model


def create_loss_function(args):
    """Create loss function based on args."""
    if args.loss == 'nll':
        print(f"Using NLL loss (standard)")
        return DiscreteTimeSurvivalLoss()
    elif args.loss == 'ranking':
        print(f"Using Pairwise Ranking loss (margin={args.margin})")
        return PairwiseRankingLoss(margin=args.margin)
    else:  # hybrid
        print(f"Using Hybrid loss (lambda_rank={args.lambda_rank}, margin={args.margin})")
        return HybridSurvivalLoss(
            lambda_nll=1.0,
            lambda_rank=args.lambda_rank,
            margin=args.margin
        )


def train_epoch(
    model: BEHRTForSurvival,
    train_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_nll = 0
    total_rank = 0
    n_batches = 0
    
    for batch in train_loader:
        # Move to device
        codes = batch['codes'].to(device)
        ages = batch['ages'].to(device)
        visit_ids = batch['visit_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        event_times = batch['event_time'].to(device)
        event_indicators = batch['event_indicator'].to(device)
        
        # Forward pass
        hazards = model(codes, ages, visit_ids, attention_mask)
        
        # Compute loss
        if loss_type == 'nll':
            # NLL loss needs sequence mask (derived from visit_ids and attention_mask)
            max_visits = visit_ids.max().item() + 1
            sequence_mask = torch.zeros(codes.size(0), max_visits, device=device)
            for b in range(codes.size(0)):
                for v in range(max_visits):
                    if ((visit_ids[b] == v) & (attention_mask[b] == 1)).any():
                        sequence_mask[b, v] = 1
            
            loss = loss_fn(hazards, event_times, event_indicators, sequence_mask)
            total_loss += loss.item()
        
        elif loss_type == 'ranking':
            # Ranking loss needs risk scores
            risk_scores = model.compute_risk_score(hazards)
            loss = loss_fn(risk_scores, event_times, event_indicators)
            total_loss += loss.item()
        
        else:  # hybrid
            # Hybrid needs both
            max_visits = visit_ids.max().item() + 1
            sequence_mask = torch.zeros(codes.size(0), max_visits, device=device)
            for b in range(codes.size(0)):
                for v in range(max_visits):
                    if ((visit_ids[b] == v) & (attention_mask[b] == 1)).any():
                        sequence_mask[b, v] = 1
            
            risk_scores = model.compute_risk_score(hazards)
            loss, loss_dict = loss_fn(
                hazards, risk_scores, event_times, event_indicators, sequence_mask
            )
            total_loss += loss_dict['total']
            total_nll += loss_dict['nll']
            total_rank += loss_dict['rank']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        n_batches += 1
    
    metrics = {'loss': total_loss / n_batches}
    if loss_type == 'hybrid':
        metrics['nll'] = total_nll / n_batches
        metrics['rank'] = total_rank / n_batches
    
    return metrics


def evaluate(
    model: BEHRTForSurvival,
    data_loader: DataLoader,
    loss_fn,
    device: torch.device,
    loss_type: str
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_risk_scores = []
    all_event_times = []
    all_event_indicators = []
    n_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # Move to device
            codes = batch['codes'].to(device)
            ages = batch['ages'].to(device)
            visit_ids = batch['visit_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            event_times = batch['event_time'].to(device)
            event_indicators = batch['event_indicator'].to(device)
            
            # Forward pass
            hazards = model(codes, ages, visit_ids, attention_mask)
            risk_scores = model.compute_risk_score(hazards)
            
            # Compute loss
            if loss_type == 'nll':
                max_visits = visit_ids.max().item() + 1
                sequence_mask = torch.zeros(codes.size(0), max_visits, device=device)
                for b in range(codes.size(0)):
                    for v in range(max_visits):
                        if ((visit_ids[b] == v) & (attention_mask[b] == 1)).any():
                            sequence_mask[b, v] = 1
                
                loss = loss_fn(hazards, event_times, event_indicators, sequence_mask)
                total_loss += loss.item()
            
            elif loss_type == 'ranking':
                loss = loss_fn(risk_scores, event_times, event_indicators)
                total_loss += loss.item()
            
            else:  # hybrid
                max_visits = visit_ids.max().item() + 1
                sequence_mask = torch.zeros(codes.size(0), max_visits, device=device)
                for b in range(codes.size(0)):
                    for v in range(max_visits):
                        if ((visit_ids[b] == v) & (attention_mask[b] == 1)).any():
                            sequence_mask[b, v] = 1
                
                loss, loss_dict = loss_fn(
                    hazards, risk_scores, event_times, event_indicators, sequence_mask
                )
                total_loss += loss_dict['total']
            
            # Collect for C-index
            all_risk_scores.append(risk_scores.cpu())
            all_event_times.append(event_times.cpu())
            all_event_indicators.append(event_indicators.cpu())
            
            n_batches += 1
    
    # Compute C-index
    all_risk_scores = torch.cat(all_risk_scores)
    all_event_times = torch.cat(all_event_times)
    all_event_indicators = torch.cat(all_event_indicators)
    
    # Convert risk scores to hazards format for concordance_index function
    # (it expects hazards but we can use risk scores directly)
    c_index = concordance_index_from_risk(
        all_risk_scores, all_event_times, all_event_indicators
    )
    
    return {
        'loss': total_loss / n_batches,
        'c_index': c_index
    }


def concordance_index_from_risk(
    risk_scores: torch.Tensor,
    event_times: torch.Tensor,
    event_indicators: torch.Tensor
) -> float:
    """Compute C-index from risk scores."""
    batch_size = len(risk_scores)
    concordant = 0
    total = 0
    
    for i in range(batch_size):
        if event_indicators[i] == 0:
            continue
        
        for j in range(batch_size):
            if i == j:
                continue
            
            if event_times[j] > event_times[i]:
                total += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
    
    return concordant / total if total > 0 else 0.5


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"{args.loss}_{args.model_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    patient_sequences, vocab_size = generate_synthetic_data(args)
    
    # Prepare datasets
    train_ds, val_ds, test_ds = prepare_behrt_survival_data(
        patient_sequences,
        vocab_size=vocab_size,
        max_seq_length=args.max_seq_length,
        seed=args.seed
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_behrt_survival
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_behrt_survival
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_behrt_survival
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Create model
    model = create_model(args, vocab_size).to(device)
    
    # Create loss function
    loss_fn = create_loss_function(args)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print(f"\n{'='*80}")
    print("Training...")
    print(f"{'='*80}\n")
    
    best_val_c_index = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_c_index': []}
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device, args.loss
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, loss_fn, device, args.loss)
        
        # Log
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_c_index'].append(val_metrics['c_index'])
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        if args.loss == 'hybrid':
            print(f"    NLL: {train_metrics['nll']:.4f}, Rank: {train_metrics['rank']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, C-index: {val_metrics['c_index']:.4f}")
        
        # Early stopping
        if val_metrics['c_index'] > best_val_c_index:
            best_val_c_index = val_metrics['c_index']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"  ✓ New best C-index: {best_val_c_index:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_metrics = evaluate(model, test_loader, loss_fn, device, args.loss)
    
    print(f"\n{'='*80}")
    print("Final Results")
    print(f"{'='*80}")
    print(f"Best Val C-index: {best_val_c_index:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test C-index: {test_metrics['c_index']:.4f}")
    print(f"Training time: {training_time:.1f}s")
    
    # Save results
    results = {
        'best_val_c_index': best_val_c_index,
        'test_loss': test_metrics['loss'],
        'test_c_index': test_metrics['c_index'],
        'training_time': training_time,
        'history': history
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
