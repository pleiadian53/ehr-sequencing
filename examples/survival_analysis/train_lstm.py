"""
Production training script for discrete-time survival LSTM.

This is the full-featured training script with proper safeguards for production use.
It works on both local machines (with large datasets) and cloud GPUs.

Key features:
- Early stopping to prevent overfitting
- Learning rate scheduling
- Gradient clipping for stability
- Comprehensive logging and checkpointing
- Configurable hyperparameters
- Training history export

Usage (local):
    python examples/survival_analysis/train_lstm.py \
        --data_dir ~/work/loinc-predictor/data/synthea/large_cohort_1000/ \
        --outcome synthetic \
        --epochs 100 \
        --batch_size 32 \
        --early_stopping_patience 10

Usage (cloud GPU - RunPods, Vast.ai, etc.):
    python examples/survival_analysis/train_lstm.py \
        --data_dir /workspace/loinc-predictor/data/synthea/large_cohort_1000/ \
        --outcome synthetic \
        --epochs 100 \
        --batch_size 64 \
        --early_stopping_patience 10

Differences from train_lstm_demo.py:
- Uses full dataset (no patient limit)
- Early stopping and LR scheduling
- Gradient clipping
- Comprehensive logging
- Production-ready hyperparameters
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

from ehrsequencing.data.adapters.synthea import SyntheaAdapter
from ehrsequencing.data.visit_grouper import VisitGrouper
from ehrsequencing.data.sequence_builder import PatientSequenceBuilder
from ehrsequencing.models.survival_lstm import DiscreteTimeSurvivalLSTM
from ehrsequencing.models.losses import DiscreteTimeSurvivalLoss, concordance_index
from ehrsequencing.synthetic.survival import DiscreteTimeSurvivalGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Train discrete-time survival LSTM on RunPods')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to Synthea data directory')
    parser.add_argument('--outcome', type=str, default='synthetic',
                       choices=['synthetic', 'ckd_progression', 'random'],
                       help='Prediction outcome')
    parser.add_argument('--min_visits', type=int, default=3,
                       help='Minimum visits per patient')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Dimension of code embeddings')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Dimension of LSTM hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (use 64 for A40/RTX 4090)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    
    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Epochs to wait before early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001,
                       help='Minimum improvement for early stopping')
    
    # Learning rate scheduling
    parser.add_argument('--lr_scheduler', type=str, default='reduce_on_plateau',
                       choices=['reduce_on_plateau', 'cosine', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--lr_patience', type=int, default=5,
                       help='Patience for ReduceLROnPlateau')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                       help='Factor for ReduceLROnPlateau')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Get torch device based on argument."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def create_synthetic_survival_labels(sequences, seed=42):
    """Create synthetic survival labels using DiscreteTimeSurvivalGenerator."""
    generator = DiscreteTimeSurvivalGenerator(
        censoring_rate=0.3,
        risk_weights={'comorbidity': 0.4, 'frequency': 0.4, 'diversity': 0.2},
        time_scale=0.3,
        seed=seed
    )
    
    outcome = generator.generate(sequences)
    
    return outcome.event_times, outcome.event_indicators


def create_survival_labels(sequences, outcome='synthetic'):
    """Create survival labels from patient sequences."""
    if outcome == 'synthetic':
        return create_synthetic_survival_labels(sequences)
    
    elif outcome == 'ckd_progression':
        event_times = []
        event_indicators = []
        
        for seq in sequences:
            num_visits = len(seq.visits)
            has_event = False
            event_time = num_visits - 1
            
            for i, visit in enumerate(seq.visits):
                ckd_codes = {'585.4', '585.5', 'N18.4', 'N18.5'}
                visit_codes = set(visit.get_all_codes())
                
                if visit_codes & ckd_codes:
                    has_event = True
                    event_time = i
                    break
            
            event_times.append(event_time)
            event_indicators.append(1 if has_event else 0)
        
        return torch.tensor(event_times), torch.tensor(event_indicators)
    
    else:  # random
        event_times = []
        event_indicators = []
        
        for seq in sequences:
            num_visits = len(seq.visits)
            event_time = np.random.randint(0, num_visits)
            event_indicator = np.random.binomial(1, 0.3)
            event_times.append(event_time)
            event_indicators.append(event_indicator)
        
        return torch.tensor(event_times), torch.tensor(event_indicators)


def create_collate_fn(builder):
    """Create collate function with access to vocabulary."""
    def collate_fn(batch):
        sequences = [item['sequence'] for item in batch]
        event_times = torch.tensor([item['event_time'] for item in batch])
        event_indicators = torch.tensor([item['event_indicator'] for item in batch])
        
        max_visits = max(len(seq.visits) for seq in sequences)
        max_codes_per_visit = max(
            max(visit.num_codes() for visit in seq.visits)
            for seq in sequences
        )
        
        batch_size = len(sequences)
        
        visit_codes = torch.zeros(batch_size, max_visits, max_codes_per_visit, dtype=torch.long)
        visit_mask = torch.zeros(batch_size, max_visits, max_codes_per_visit, dtype=torch.bool)
        sequence_mask = torch.zeros(batch_size, max_visits, dtype=torch.bool)
        
        for i, seq in enumerate(sequences):
            num_visits = len(seq.visits)
            sequence_mask[i, :num_visits] = True
            
            for j, visit in enumerate(seq.visits):
                codes = visit.get_all_codes()
                encoded_codes = [
                    builder.vocab.get(code, builder.unk_id)
                    for code in codes[:max_codes_per_visit]
                ]
                
                num_codes = len(encoded_codes)
                
                if num_codes > 0:
                    visit_codes[i, j, :num_codes] = torch.tensor(encoded_codes)
                    visit_mask[i, j, :num_codes] = True
        
        return {
            'visit_codes': visit_codes,
            'visit_mask': visit_mask,
            'sequence_mask': sequence_mask,
            'event_times': event_times,
            'event_indicators': event_indicators,
        }
    
    return collate_fn


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        visit_codes = batch['visit_codes'].to(device)
        visit_mask = batch['visit_mask'].to(device)
        sequence_mask = batch['sequence_mask'].to(device)
        event_times = batch['event_times'].to(device)
        event_indicators = batch['event_indicators'].to(device)
        
        hazards = model(visit_codes, visit_mask, sequence_mask)
        loss = criterion(hazards, event_times, event_indicators, sequence_mask)
        
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_hazards = []
    all_event_times = []
    all_event_indicators = []
    all_sequence_masks = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            visit_codes = batch['visit_codes'].to(device)
            visit_mask = batch['visit_mask'].to(device)
            sequence_mask = batch['sequence_mask'].to(device)
            event_times = batch['event_times'].to(device)
            event_indicators = batch['event_indicators'].to(device)
            
            hazards = model(visit_codes, visit_mask, sequence_mask)
            loss = criterion(hazards, event_times, event_indicators, sequence_mask)
            total_loss += loss.item()
            
            all_hazards.append(hazards.cpu())
            all_event_times.append(event_times.cpu())
            all_event_indicators.append(event_indicators.cpu())
            all_sequence_masks.append(sequence_mask.cpu())
    
    avg_loss = total_loss / len(dataloader)
    
    all_hazards = torch.cat(all_hazards, dim=0)
    all_event_times = torch.cat(all_event_times, dim=0)
    all_event_indicators = torch.cat(all_event_indicators, dim=0)
    all_sequence_masks = torch.cat(all_sequence_masks, dim=0)
    
    c_index = concordance_index(all_hazards, all_event_times, all_event_indicators, all_sequence_masks)
    
    return avg_loss, c_index


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def main():
    args = parse_args()
    
    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    adapter = SyntheaAdapter(args.data_dir)
    events = adapter.load_events()
    print(f"Loaded {len(events)} events from {events['patient_id'].nunique()} patients")
    
    # Group into visits
    print("Grouping events into visits...")
    grouper = VisitGrouper(time_window_hours=24)
    all_visits = grouper.group_events(events)
    
    # Group visits by patient
    from collections import defaultdict
    patient_visits = defaultdict(list)
    for visit in all_visits:
        patient_visits[visit.patient_id].append(visit)
    
    # Sort visits by timestamp for each patient
    for patient_id in patient_visits:
        patient_visits[patient_id].sort(key=lambda v: v.timestamp)
    
    patient_visits_list = list(patient_visits.values())
    
    # Build vocabulary first
    print("Building vocabulary...")
    builder = PatientSequenceBuilder()
    vocab = builder.build_vocabulary(patient_visits_list, min_frequency=2)
    print(f"Vocabulary size: {builder.vocabulary_size}")
    
    # Build sequences
    print("Building patient sequences...")
    sequences = builder.build_sequences(patient_visits_list, min_visits=args.min_visits)
    
    print(f"Loaded {len(sequences)} patient sequences")
    
    # Create survival labels
    print(f"Creating survival labels for outcome: {args.outcome}")
    event_times, event_indicators = create_survival_labels(sequences, args.outcome)
    
    print(f"Event rate: {event_indicators.float().mean():.2%}")
    print(f"Median event/censoring time: {event_times.float().median():.1f} visits")
    
    # Create dataset
    dataset = [
        {
            'sequence': seq,
            'event_time': event_time.item(),
            'event_indicator': event_indicator.item()
        }
        for seq, event_time, event_indicator in zip(sequences, event_times, event_indicators)
    ]
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Create collate function with vocabulary
    collate_fn = create_collate_fn(builder)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    print("Creating model...")
    vocab_size = builder.vocabulary_size
    model = DiscreteTimeSurvivalLSTM(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Parameters per patient: {num_params / len(train_dataset):.1f}")
    
    # Loss and optimizer
    criterion = DiscreteTimeSurvivalLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.lr_scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience,
            verbose=True
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        mode='min'
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_val_c_index = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_c_index': [],
        'lr': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, val_c_index = evaluate(model, val_loader, criterion, device)
        print(f"Val loss: {val_loss:.4f}, C-index: {val_c_index:.4f}")
        
        # Record history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_c_index'].append(val_c_index)
        history['lr'].append(current_lr)
        
        # Learning rate scheduling
        if scheduler is not None:
            if args.lr_scheduler == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_c_index = val_c_index
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_c_index': val_c_index,
                'args': vars(args),
                'vocab_size': vocab_size
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_c_index': val_c_index,
            }, checkpoint_path)
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Best C-index: {best_val_c_index:.4f}")
            break
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best C-index: {best_val_c_index:.4f}")
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    main()
