"""
Benchmark: HierarchicalBEHRTForSurvival vs BEHRTForSurvival (flat).

Trains both models on the same synthetic data split and compares:
  - Test C-index
  - Test NLL loss
  - Parameter count
  - Training time

Usage:
    python benchmark_hierarchical_vs_flat.py --num-patients 2000 --epochs 30
    python benchmark_hierarchical_vs_flat.py --num-patients 5000 --epochs 50 --model-size medium
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.data import DataLoader
import numpy as np

from ehrsequencing.models.hierarchical_survival import (
    HierarchicalBEHRTForSurvival,
    HierarchicalSurvivalConfig,
)
from ehrsequencing.models.behrt_survival import BEHRTForSurvival, BEHRTSurvivalConfig
from ehrsequencing.models.losses import DiscreteTimeSurvivalLoss
from ehrsequencing.data.hierarchical_survival_dataset import (
    HierarchicalSurvivalDataset,
    collate_hierarchical_survival,
)
from ehrsequencing.data.behrt_survival_dataset import (
    BEHRTSurvivalDataset,
    collate_behrt_survival,
)
from ehrsequencing.synthetic.survival import generate_survival_patient_sequences


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _concordance_index(risk_scores, event_times, event_indicators) -> float:
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


def split_sequences(sequences: List[Dict], train_r=0.7, val_r=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sequences))
    n_train = int(len(idx) * train_r)
    n_val = int(len(idx) * val_r)
    return (
        [sequences[i] for i in idx[:n_train]],
        [sequences[i] for i in idx[n_train:n_train + n_val]],
        [sequences[i] for i in idx[n_train + n_val:]],
    )


# ---------------------------------------------------------------------------
# Hierarchical model training
# ---------------------------------------------------------------------------

def make_hier_loaders(train_seqs, val_seqs, test_seqs, args):
    kwargs = dict(vocab_size=args.vocab_size,
                  max_visits=args.max_visits,
                  max_codes_per_visit=args.max_codes_per_visit)
    train_ds = HierarchicalSurvivalDataset(train_seqs, **kwargs)
    val_ds   = HierarchicalSurvivalDataset(val_seqs,   **kwargs)
    test_ds  = HierarchicalSurvivalDataset(test_seqs,  **kwargs)
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size,
                                        shuffle=shuffle,
                                        collate_fn=collate_hierarchical_survival)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)


def hier_step(model, batch, loss_fn, device):
    codes       = batch['codes'].to(device)
    ages        = batch['ages'].to(device)
    time_deltas = batch['time_deltas'].to(device)
    code_mask   = batch['code_mask'].to(device)
    visit_mask  = batch['visit_mask'].to(device)
    event_times = batch['event_time'].to(device)
    event_inds  = batch['event_indicator'].to(device)

    hazards = model(codes, ages, time_deltas, code_mask, visit_mask)
    # Clamp event_times to valid hazard range
    max_v = hazards.shape[1]
    last_visit = visit_mask.long().sum(dim=1) - 1          # (B,)
    event_times = torch.min(event_times, last_visit.clamp(min=0))
    event_times = event_times.clamp(max=max_v - 1)
    loss = loss_fn(hazards, event_times, event_inds, visit_mask.float())
    risk  = model.compute_risk_score(hazards, visit_mask=visit_mask)
    return loss, risk, event_times, event_inds


def train_hierarchical(train_loader, val_loader, args, device) -> Tuple[HierarchicalBEHRTForSurvival, Dict]:
    if args.model_size == 'small':
        config = HierarchicalSurvivalConfig.small(args.vocab_size)
    elif args.model_size == 'medium':
        config = HierarchicalSurvivalConfig.medium(args.vocab_size)
    else:
        config = HierarchicalSurvivalConfig.large(args.vocab_size)
    config.max_visits = args.max_visits
    config.max_codes_per_visit = args.max_codes_per_visit

    model = HierarchicalBEHRTForSurvival(config).to(device)
    loss_fn = DiscreteTimeSurvivalLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_c = 0.0
    best_state = None
    patience = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            loss, _, _, _ = hier_step(model, batch, loss_fn, device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_c = _eval_c_index(model, val_loader, device, 'hier')
        if val_c > best_val_c:
            best_val_c = val_c
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [hier] epoch {epoch+1:3d}  val_c={val_c:.4f}  best={best_val_c:.4f}")

    training_time = time.time() - t0
    model.load_state_dict(best_state)
    return model, {'training_time': training_time, 'best_val_c_index': best_val_c}


# ---------------------------------------------------------------------------
# Flat BEHRT training
# ---------------------------------------------------------------------------

FLAT_MAX_SEQ = 512  # flat BEHRT positional embedding cap

def make_flat_loaders(train_seqs, val_seqs, test_seqs, args):
    kwargs = dict(vocab_size=args.vocab_size,
                  max_seq_length=min(args.max_visits * args.max_codes_per_visit, FLAT_MAX_SEQ))
    train_ds = BEHRTSurvivalDataset(train_seqs, **kwargs)
    val_ds   = BEHRTSurvivalDataset(val_seqs,   **kwargs)
    test_ds  = BEHRTSurvivalDataset(test_seqs,  **kwargs)
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size,
                                        shuffle=shuffle,
                                        collate_fn=collate_behrt_survival)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)


def flat_step(model, batch, loss_fn, device):
    codes       = batch['codes'].to(device)
    ages        = batch['ages'].to(device)
    visit_ids   = batch['visit_ids'].to(device)
    attn_mask   = batch['attention_mask'].to(device)
    event_times = batch['event_time'].to(device)
    event_inds  = batch['event_indicator'].to(device)

    # Guard: ensure every sample has at least one real code (avoid all-zero attn)
    has_real = attn_mask.bool().any(dim=1)  # (B,)
    if not has_real.all():
        codes     = codes[has_real]
        ages      = ages[has_real]
        visit_ids = visit_ids[has_real]
        attn_mask = attn_mask[has_real]
        event_times = event_times[has_real]
        event_inds  = event_inds[has_real]

    hazards = model(codes, ages, visit_ids, attn_mask)  # (B', max_visits)
    max_v = hazards.shape[1]

    # Replace any NaN hazards (degenerate batches) with zeros
    hazards = torch.nan_to_num(hazards, nan=0.0)

    # Build visit_mask vectorized: (B', max_v)
    vid_exp = visit_ids.unsqueeze(2)                               # (B', L, 1)
    v_range = torch.arange(max_v, device=device).view(1, 1, max_v)  # (1,1,V)
    valid   = attn_mask.bool().unsqueeze(2)                        # (B', L, 1)
    visit_mask = ((vid_exp == v_range) & valid).any(dim=1).float()  # (B', V)

    # Clamp event_times per sample to that sample's last real visit
    last_visit = visit_mask.long().sum(dim=1) - 1                  # (B',)
    event_times = torch.min(event_times, last_visit.clamp(min=0))

    loss = loss_fn(hazards, event_times, event_inds, visit_mask)
    risk = model.compute_risk_score(hazards)
    return loss, risk, event_times, event_inds


def train_flat(train_loader, val_loader, args, device) -> Tuple[BEHRTForSurvival, Dict]:
    if args.model_size == 'small':
        config = BEHRTSurvivalConfig.from_pretrained_small(vocab_size=args.vocab_size)
    elif args.model_size == 'medium':
        config = BEHRTSurvivalConfig.from_pretrained_medium(vocab_size=args.vocab_size)
    else:
        config = BEHRTSurvivalConfig.from_pretrained_large(vocab_size=args.vocab_size)

    model = BEHRTForSurvival(config).to(device)
    loss_fn = DiscreteTimeSurvivalLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_c = 0.0
    best_state = None
    patience = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            loss, _, _, _ = flat_step(model, batch, loss_fn, device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_c = _eval_c_index(model, val_loader, device, 'flat')
        if val_c > best_val_c:
            best_val_c = val_c
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [flat] epoch {epoch+1:3d}  val_c={val_c:.4f}  best={best_val_c:.4f}")

    training_time = time.time() - t0
    model.load_state_dict(best_state)
    return model, {'training_time': training_time, 'best_val_c_index': best_val_c}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _eval_c_index(model, loader, device, mode: str) -> float:
    model.eval()
    all_risk, all_et, all_ei = [], [], []
    with torch.no_grad():
        for batch in loader:
            if mode == 'hier':
                _, risk, et, ei = hier_step(model, batch, DiscreteTimeSurvivalLoss(), device)
            else:
                _, risk, et, ei = flat_step(model, batch, DiscreteTimeSurvivalLoss(), device)
            all_risk.append(risk.cpu())
            all_et.append(et.cpu())
            all_ei.append(ei.cpu())
    return _concordance_index(
        torch.cat(all_risk), torch.cat(all_et), torch.cat(all_ei)
    )


def eval_nll(model, loader, device, mode: str) -> float:
    model.eval()
    loss_fn = DiscreteTimeSurvivalLoss()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if mode == 'hier':
                loss, _, _, _ = hier_step(model, batch, loss_fn, device)
            else:
                loss, _, _, _ = flat_step(model, batch, loss_fn, device)
            total += loss.item()
            n += 1
    return total / n if n > 0 else float('nan')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark hierarchical vs flat BEHRT')
    parser.add_argument('--num-patients', type=int, default=2000)
    parser.add_argument('--vocab-size', type=int, default=1000)
    parser.add_argument('--max-visits', type=int, default=50)
    parser.add_argument('--max-codes-per-visit', type=int, default=30)
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'medium', 'large'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--output-dir', type=str, default='experiments/benchmark')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}  |  model_size={args.model_size}  |  patients={args.num_patients}")

    print("\nGenerating data...")
    sequences = generate_survival_patient_sequences(
        num_patients=args.num_patients,
        vocab_size=args.vocab_size,
        max_visits=args.max_visits,
        max_codes_per_visit=args.max_codes_per_visit,
        seed=args.seed,
    )
    train_seqs, val_seqs, test_seqs = split_sequences(sequences, seed=args.seed)
    print(f"Split: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")

    # ---- Hierarchical ----
    print("\n" + "="*60)
    print("Training: HierarchicalBEHRTForSurvival")
    print("="*60)
    hier_train, hier_val, hier_test = make_hier_loaders(train_seqs, val_seqs, test_seqs, args)
    hier_model, hier_meta = train_hierarchical(hier_train, hier_val, args, device)
    hier_test_c   = _eval_c_index(hier_model, hier_test, device, 'hier')
    hier_test_nll = eval_nll(hier_model, hier_test, device, 'hier')
    hier_params   = hier_model.get_trainable_parameters()['total']

    # ---- Flat BEHRT ----
    print("\n" + "="*60)
    print("Training: BEHRTForSurvival (flat)")
    print("="*60)
    flat_train, flat_val, flat_test = make_flat_loaders(train_seqs, val_seqs, test_seqs, args)
    flat_model, flat_meta = train_flat(flat_train, flat_val, args, device)
    flat_test_c   = _eval_c_index(flat_model, flat_test, device, 'flat')
    flat_test_nll = eval_nll(flat_model, flat_test, device, 'flat')
    flat_params   = flat_model.get_trainable_parameters()['total']

    # ---- Results table ----
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    header = f"{'Metric':<25} {'Hierarchical':>14} {'Flat BEHRT':>14}"
    print(header)
    print("-" * len(header))
    rows = [
        ("Test C-index",       f"{hier_test_c:.4f}",              f"{flat_test_c:.4f}"),
        ("Test NLL",           f"{hier_test_nll:.4f}",            f"{flat_test_nll:.4f}"),
        ("Parameters",         f"{hier_params:,}",                f"{flat_params:,}"),
        ("Training time (s)",  f"{hier_meta['training_time']:.1f}", f"{flat_meta['training_time']:.1f}"),
        ("Best val C-index",   f"{hier_meta['best_val_c_index']:.4f}", f"{flat_meta['best_val_c_index']:.4f}"),
    ]
    for label, h, f in rows:
        print(f"{label:<25} {h:>14} {f:>14}")

    results = {
        'config': vars(args),
        'hierarchical': {
            'test_c_index': hier_test_c,
            'test_nll': hier_test_nll,
            'parameters': hier_params,
            **hier_meta,
        },
        'flat': {
            'test_c_index': flat_test_c,
            'test_nll': flat_test_nll,
            'parameters': flat_params,
            **flat_meta,
        },
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out / 'benchmark_results.json'}")


if __name__ == '__main__':
    main()
