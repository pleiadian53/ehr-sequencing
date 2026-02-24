"""
Training script for HierarchicalBEHRTForSurvival model.

Two-level hierarchical encoder: codes → visit embeddings (attention pooling)
→ patient timeline (transformer) → discrete-time hazard per visit.

Supports three loss functions:
1. NLL (standard) - Optimizes calibration
2. Pairwise Ranking - Directly optimizes C-index
3. Hybrid (NLL + Ranking) - Best of both worlds

Usage:
    # NLL loss (standard)
    python train_hierarchical_survival.py --loss nll --epochs 100

    # Hybrid loss
    python train_hierarchical_survival.py --loss hybrid --lambda-rank 0.05 --epochs 100

    # Compare sizes
    python train_hierarchical_survival.py --model-size medium --loss hybrid --epochs 100

See docs/methods/discrete_time_survival_analysis/09_hierarchical_architecture.md
for architecture details.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.data import DataLoader
import numpy as np

from ehrsequencing.models.hierarchical_survival import (
    HierarchicalBEHRTForSurvival,
    HierarchicalSurvivalConfig,
)
from ehrsequencing.models.losses import (
    DiscreteTimeSurvivalLoss,
    PairwiseRankingLoss,
    HybridSurvivalLoss,
    concordance_index,
)
from ehrsequencing.data.hierarchical_survival_dataset import (
    HierarchicalSurvivalDataset,
    collate_hierarchical_survival,
    prepare_hierarchical_survival_data,
)
from ehrsequencing.synthetic.survival import generate_survival_patient_sequences


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train HierarchicalBEHRTForSurvival')

    # Data
    parser.add_argument('--num-patients', type=int, default=5000,
                        help='Number of synthetic patients')
    parser.add_argument('--vocab-size', type=int, default=1000,
                        help='Vocabulary size')
    parser.add_argument('--max-visits', type=int, default=50,
                        help='Maximum visits per patient (V dimension)')
    parser.add_argument('--max-codes-per-visit', type=int, default=30,
                        help='Maximum codes per visit (C dimension)')

    # Model
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'medium', 'large'],
                        help='Model size (hidden_dim, n_heads, n_layers)')

    # Loss function
    parser.add_argument('--loss', type=str, default='nll',
                        choices=['nll', 'ranking', 'hybrid'],
                        help='Loss function type')
    parser.add_argument('--lambda-rank', type=float, default=0.05,
                        help='Weight for ranking loss in hybrid')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Margin for pairwise ranking loss')

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
                        help='Early stopping patience (epochs without val C-index improvement)')

    # Output
    parser.add_argument('--output-dir', type=str,
                        default='experiments/hierarchical_survival',
                        help='Output directory for checkpoints and results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(args) -> list:
    """Generate synthetic EHR data with survival outcomes in visit-grouped format."""
    print(f"\n{'='*80}")
    print("Generating synthetic data...")
    print(f"{'='*80}")

    patient_sequences = generate_survival_patient_sequences(
        num_patients=args.num_patients,
        vocab_size=args.vocab_size,
        max_visits=args.max_visits,
        max_codes_per_visit=args.max_codes_per_visit,
        seed=args.seed,
    )

    print(f"Generated {len(patient_sequences)} patients")
    print(f"Average visits: {np.mean([len(s['visits']) for s in patient_sequences]):.1f}")
    print(f"Event rate: {np.mean([s['outcome']['event_indicator'] for s in patient_sequences]):.2%}")

    return patient_sequences


def create_model(args, vocab_size: int) -> HierarchicalBEHRTForSurvival:
    print(f"\n{'='*80}")
    print("Creating model...")
    print(f"{'='*80}")

    if args.model_size == 'small':
        config = HierarchicalSurvivalConfig.small(vocab_size)
    elif args.model_size == 'medium':
        config = HierarchicalSurvivalConfig.medium(vocab_size)
    else:
        config = HierarchicalSurvivalConfig.large(vocab_size)

    config.max_visits = args.max_visits
    config.max_codes_per_visit = args.max_codes_per_visit

    model = HierarchicalBEHRTForSurvival(config)

    param_counts = model.get_trainable_parameters()
    print(f"Total parameters:     {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,} ({param_counts['trainable_percentage']:.1f}%)")

    return model


def create_loss_function(args):
    if args.loss == 'nll':
        print("Using NLL loss")
        return DiscreteTimeSurvivalLoss()
    elif args.loss == 'ranking':
        print(f"Using Pairwise Ranking loss (margin={args.margin})")
        return PairwiseRankingLoss(margin=args.margin)
    else:
        print(f"Using Hybrid loss (lambda_rank={args.lambda_rank}, margin={args.margin})")
        return HybridSurvivalLoss(
            lambda_nll=1.0,
            lambda_rank=args.lambda_rank,
            margin=args.margin,
        )


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_epoch(
    model: HierarchicalBEHRTForSurvival,
    train_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_nll = 0.0
    total_rank = 0.0
    n_batches = 0

    for batch in train_loader:
        codes = batch['codes'].to(device)                  # (B, V, C)
        ages = batch['ages'].to(device)                    # (B, V)
        time_deltas = batch['time_deltas'].to(device)      # (B, V)
        code_mask = batch['code_mask'].to(device)          # (B, V, C)
        visit_mask = batch['visit_mask'].to(device)        # (B, V)
        event_times = batch['event_time'].to(device)       # (B,)
        event_indicators = batch['event_indicator'].to(device)  # (B,)

        hazards = model(codes, ages, time_deltas, code_mask, visit_mask)  # (B, V)

        if loss_type == 'nll':
            loss = loss_fn(hazards, event_times, event_indicators, visit_mask.float())
            total_loss += loss.item()

        elif loss_type == 'ranking':
            risk_scores = model.compute_risk_score(hazards, visit_mask=visit_mask)
            loss = loss_fn(risk_scores, event_times, event_indicators)
            total_loss += loss.item()

        else:  # hybrid
            risk_scores = model.compute_risk_score(hazards, visit_mask=visit_mask)
            loss, loss_dict = loss_fn(
                hazards, risk_scores, event_times, event_indicators, visit_mask.float()
            )
            total_loss += loss_dict['total']
            total_nll += loss_dict['nll']
            total_rank += loss_dict['rank']

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
    model: HierarchicalBEHRTForSurvival,
    data_loader: DataLoader,
    loss_fn,
    device: torch.device,
    loss_type: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_risk_scores = []
    all_event_times = []
    all_event_indicators = []
    n_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            codes = batch['codes'].to(device)
            ages = batch['ages'].to(device)
            time_deltas = batch['time_deltas'].to(device)
            code_mask = batch['code_mask'].to(device)
            visit_mask = batch['visit_mask'].to(device)
            event_times = batch['event_time'].to(device)
            event_indicators = batch['event_indicator'].to(device)

            hazards = model(codes, ages, time_deltas, code_mask, visit_mask)
            risk_scores = model.compute_risk_score(hazards, visit_mask=visit_mask)

            if loss_type == 'nll':
                loss = loss_fn(hazards, event_times, event_indicators, visit_mask.float())
                total_loss += loss.item()

            elif loss_type == 'ranking':
                loss = loss_fn(risk_scores, event_times, event_indicators)
                total_loss += loss.item()

            else:  # hybrid
                loss, loss_dict = loss_fn(
                    hazards, risk_scores, event_times, event_indicators, visit_mask.float()
                )
                total_loss += loss_dict['total']

            all_risk_scores.append(risk_scores.cpu())
            all_event_times.append(event_times.cpu())
            all_event_indicators.append(event_indicators.cpu())
            n_batches += 1

    all_risk_scores = torch.cat(all_risk_scores)
    all_event_times = torch.cat(all_event_times)
    all_event_indicators = torch.cat(all_event_indicators)

    c_index = _concordance_index_from_risk(all_risk_scores, all_event_times, all_event_indicators)

    return {'loss': total_loss / n_batches, 'c_index': c_index}


def _concordance_index_from_risk(
    risk_scores: torch.Tensor,
    event_times: torch.Tensor,
    event_indicators: torch.Tensor,
) -> float:
    """Compute C-index from scalar risk scores."""
    concordant = 0
    total = 0
    n = len(risk_scores)
    for i in range(n):
        if event_indicators[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if event_times[j] > event_times[i]:
                total += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
    return concordant / total if total > 0 else 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir) / f"{args.loss}_{args.model_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    patient_sequences = generate_synthetic_data(args)

    train_ds, val_ds, test_ds = prepare_hierarchical_survival_data(
        patient_sequences,
        vocab_size=args.vocab_size,
        max_visits=args.max_visits,
        max_codes_per_visit=args.max_codes_per_visit,
        seed=args.seed,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_hierarchical_survival)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_hierarchical_survival)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_hierarchical_survival)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    model = create_model(args, args.vocab_size).to(device)
    loss_fn = create_loss_function(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"\n{'='*80}")
    print("Training...")
    print(f"{'='*80}\n")

    best_val_c_index = 0.0
    patience_counter = 0
    history: Dict[str, list] = {'train_loss': [], 'val_loss': [], 'val_c_index': []}
    start_time = time.time()

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, args.loss)
        val_metrics = evaluate(model, val_loader, loss_fn, device, args.loss)

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_c_index'].append(val_metrics['c_index'])

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"train_loss={train_metrics['loss']:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_c_index={val_metrics['c_index']:.4f}")
        if args.loss == 'hybrid':
            print(f"  nll={train_metrics['nll']:.4f}  rank={train_metrics['rank']:.4f}")

        if val_metrics['c_index'] > best_val_c_index:
            best_val_c_index = val_metrics['c_index']
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"  -> New best C-index: {best_val_c_index:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time

    model.load_state_dict(torch.load(output_dir / 'best_model.pt', map_location=device))
    test_metrics = evaluate(model, test_loader, loss_fn, device, args.loss)

    print(f"\n{'='*80}")
    print("Final Results")
    print(f"{'='*80}")
    print(f"Best Val C-index: {best_val_c_index:.4f}")
    print(f"Test Loss:        {test_metrics['loss']:.4f}")
    print(f"Test C-index:     {test_metrics['c_index']:.4f}")
    print(f"Training time:    {training_time:.1f}s")

    results = {
        'model': 'HierarchicalBEHRTForSurvival',
        'model_size': args.model_size,
        'loss': args.loss,
        'best_val_c_index': best_val_c_index,
        'test_loss': test_metrics['loss'],
        'test_c_index': test_metrics['c_index'],
        'training_time': training_time,
        'history': history,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
