"""
Example script for training LSTM baseline model on EHR sequences.

This script demonstrates:
1. Loading and preprocessing Synthea data
2. Creating train/val/test splits
3. Training LSTM baseline model
4. Evaluating model performance
5. Saving trained model

Usage:
    python examples/train_lstm_baseline.py --data_dir /path/to/synthea --output_dir ./outputs
"""

import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ehrsequencing.data import SyntheaAdapter, VisitGrouper, PatientSequenceBuilder
from ehrsequencing.models import create_lstm_baseline
from ehrsequencing.training import Trainer, binary_accuracy, auroc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LSTM baseline model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to Synthea data directory')
    parser.add_argument('--max_patients', type=int, default=None,
                       help='Maximum number of patients to load')
    
    # Model arguments
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Model size')
    parser.add_argument('--visit_aggregation', type=str, default='mean',
                       choices=['mean', 'sum', 'max', 'attention'],
                       help='Visit aggregation strategy')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Get appropriate device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_arg


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Handles variable-length sequences and creates proper masks.
    """
    # Extract data
    visit_codes = [item['visit_codes'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    
    # Get dimensions
    batch_size = len(visit_codes)
    max_visits = max(len(seq) for seq in visit_codes)
    max_codes = max(max(len(visit) for visit in seq) for seq in visit_codes)
    
    # Create padded tensors
    padded_codes = torch.zeros(batch_size, max_visits, max_codes, dtype=torch.long)
    visit_mask = torch.zeros(batch_size, max_visits, max_codes, dtype=torch.bool)
    sequence_mask = torch.zeros(batch_size, max_visits, dtype=torch.bool)
    
    # Fill tensors
    for i, seq in enumerate(visit_codes):
        sequence_mask[i, :len(seq)] = 1
        for j, visit in enumerate(seq):
            padded_codes[i, j, :len(visit)] = torch.tensor(visit)
            visit_mask[i, j, :len(visit)] = 1
    
    return {
        'visit_codes': padded_codes,
        'visit_mask': visit_mask,
        'sequence_mask': sequence_mask,
        'labels': labels.unsqueeze(1)
    }


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    adapter = SyntheaAdapter(args.data_dir)
    
    # Load patients and events
    patients = adapter.load_patients(limit=args.max_patients)
    logger.info(f"Loaded {len(patients)} patients")
    
    events = adapter.load_events()
    logger.info(f"Loaded {len(events)} events")
    
    # Group into visits
    logger.info("Grouping events into visits")
    grouper = VisitGrouper(strategy='hybrid')
    visits_by_patient = grouper.group_by_patient(events)
    logger.info(f"Created visits for {len(visits_by_patient)} patients")
    
    # Build sequences
    logger.info("Building patient sequences")
    builder = PatientSequenceBuilder(
        max_visits=50,
        max_codes_per_visit=100
    )
    sequences = builder.build_sequences(visits_by_patient)
    vocab = builder.build_vocabulary(visits_by_patient)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset
    # For this example, we'll create synthetic labels (predict if patient has diabetes)
    # In practice, you'd extract real labels from the data
    dataset_items = []
    for seq in sequences:
        # Encode sequence
        encoded = builder.encode_sequence(seq, vocab)
        
        # Create synthetic label (1 if patient has diabetes code)
        diabetes_codes = {'44054006', 'E11.9', '73211009'}  # Example diabetes codes
        has_diabetes = any(
            any(code in diabetes_codes for code in visit.codes)
            for visit in seq.visits
        )
        
        dataset_items.append({
            'visit_codes': encoded['visit_codes'],
            'label': 1 if has_diabetes else 0
        })
    
    logger.info(f"Created {len(dataset_items)} sequences")
    logger.info(f"Positive labels: {sum(item['label'] for item in dataset_items)}")
    
    # Split into train/val/test
    train_size = int(0.7 * len(dataset_items))
    val_size = int(0.15 * len(dataset_items))
    test_size = len(dataset_items) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset_items,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    logger.info(f"Creating {args.model_size} LSTM model")
    model = create_lstm_baseline(
        vocab_size=len(vocab),
        task='binary_classification',
        model_size=args.model_size
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        metrics={'accuracy': binary_accuracy, 'auroc': auroc},
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=output_dir / 'checkpoints'
    )
    
    # Train
    logger.info("Starting training")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save vocabulary
    vocab_path = output_dir / 'vocabulary.txt'
    with open(vocab_path, 'w') as f:
        for code, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{code}\t{idx}\n")
    logger.info(f"Vocabulary saved to {vocab_path}")
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
