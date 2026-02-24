# Architecture Decisions: BEHRT vs LSTM and Flat vs Hierarchical

**Topics:** LSTM baseline architecture, BEHRT vs LSTM comparison, flat vs hierarchical design tradeoffs, benchmark framework

**Reference code:** `src/ehrsequencing/models/survival_lstm.py`, `src/ehrsequencing/models/behrt_survival.py`, `examples/survival_analysis/`

---

## Table of Contents

1. [LSTM Baseline Architecture](#lstm-baseline-architecture)
2. [BEHRT for Survival Architecture](#behrt-for-survival-architecture)
3. [BEHRT vs LSTM: Key Differences](#behrt-vs-lstm-key-differences)
4. [Flat vs Hierarchical Transformer](#flat-vs-hierarchical-transformer)
5. [Benchmark Framework](#benchmark-framework)
6. [Expected Performance](#expected-performance)

---

## LSTM Baseline Architecture

The LSTM baseline provides a strong sequential model without pretraining:

```
Input Codes → Embedding Layer → Mean Pool per Visit → LSTM → Hazard Head
```

**Components:**

```python
self.embedding   = nn.Embedding(vocab_size, embedding_dim)
self.lstm        = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                           dropout=dropout, batch_first=True)
self.hazard_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, 1),
    nn.Sigmoid()
)
```

**Forward pass:**

```python
def forward(self, codes, visit_mask, sequence_mask):
    # codes: (B, max_visits, max_codes_per_visit)
    embedded = self.embedding(codes)                          # (B, V, C, d)
    visit_emb = (embedded * sequence_mask.unsqueeze(-1)).sum(dim=2)
    visit_emb /= sequence_mask.sum(dim=2, keepdim=True) + 1e-7  # mean pool codes
    lstm_out, _ = self.lstm(visit_emb)                        # (B, V, H)
    hazards = self.hazard_head(lstm_out).squeeze(-1)          # (B, V)
    return hazards * visit_mask
```

**Key characteristics:**
- Operates at visit level natively (no flattening needed)
- Unidirectional — each visit only sees prior history
- No pretraining — learns from survival labels only
- Simpler data pipeline (hierarchical input format)

---

## BEHRT for Survival Architecture

```
Input Codes → BEHRT Encoder (pre-trained) → scatter_add Aggregation → Hazard Head
```

**Key characteristics:**
- Operates at token level, then aggregates to visit level
- Bidirectional — each code attends to all other codes in the sequence
- Pretrained on MLM (and optionally NVP) before survival fine-tuning
- Requires flattened input format with `visit_ids`

---

## BEHRT vs LSTM: Key Differences

| Aspect | LSTM Baseline | BEHRT for Survival |
|--------|--------------|-------------------|
| Context | Unidirectional (past only) | Bidirectional (full sequence) |
| Pretraining | None | MLM + optional NVP |
| Input format | Hierarchical `(B, V, C)` | Flat `(B, L)` with `visit_ids` |
| Visit aggregation | Mean pool before LSTM | scatter_add after transformer |
| Parameter count | ~5M | ~495K (small config) |
| Trainable (frozen) | All | Head only (~17K) |
| Trainable (LoRA) | All | ~50–100K |
| Convergence speed | 50–100 epochs | 20–50 epochs (pretrained) |
| Overfitting risk | Moderate | Lower (pretrained representations) |

**When LSTM may be preferred:**
- No pretrained BEHRT available for the target domain
- Very small cohort where bidirectional attention overfits
- Interpretability requirements favor sequential hidden states
- Simpler deployment pipeline

**When BEHRT is preferred:**
- Pretrained BEHRT available on similar EHR data
- Medium to large cohort
- Cross-visit context is clinically meaningful (e.g., comorbidity interactions)
- Parameter efficiency matters (frozen or LoRA)

---

## Flat vs Hierarchical Transformer

The current BEHRT survival implementation uses a **flat** architecture. An alternative is a **hierarchical transformer**:

```
Hierarchical:
  Visit 1 codes → [Code Encoder] → visit_1_repr
  Visit 2 codes → [Code Encoder] → visit_2_repr   (shared weights)
  [visit_1_repr, visit_2_repr, ...] → [Visit Encoder] → hazards

Flat (current):
  [code_1_v1, code_2_v1, code_1_v2, ...] → [BEHRT] → scatter_add → hazards
```

| Aspect | Flat (current) | Hierarchical |
|--------|---------------|--------------|
| Cross-visit attention | Full (all tokens attend to all) | None at code level |
| Pre-training compatibility | Direct BEHRT transfer | Requires separate pretraining |
| Attention complexity | O(L²) over all tokens | O(V²) over visits + O(C²) per visit |
| Batching | Simple | Ragged tensors or double padding |
| Mask discipline | Load-bearing (single mask) | Structurally isolated per visit |
| Scalability | Bottleneck at long sequences | Better for dense visits |

**Current choice rationale:** Flat is pragmatic for the current data scale (7–10 visits avg, moderate codes/visit). The flat design enables direct BEHRT pretraining transfer and full cross-visit attention with a single transformer.

**When hierarchical becomes worth considering:**
- Visits are long (> 20 codes/visit) AND sequences are long (> 30 visits)
- You want to pretrain a visit-level encoder separately on claim-level data
- Interpretability at the visit level during attention is required
- Sequence length hits transformer memory limits

---

## Benchmark Framework

To compare BEHRT variants against the LSTM baseline fairly:

**Experimental controls:**
- Identical train/val/test splits
- Same synthetic data generation (controlled risk correlation)
- Same evaluation metrics (C-index, Brier score)
- Same loss function (hybrid, `lambda_rank=0.05`)

**BEHRT variants to compare:**

```python
# A: Frozen encoder
model_frozen = BEHRTForSurvival(config, freeze_behrt=True)

# B: LoRA fine-tuning
model_lora = BEHRTForSurvival(config, use_lora=True, lora_rank=16)

# C: Full fine-tuning
model_full = BEHRTForSurvival(config)
```

**Benchmark script:** `examples/survival_analysis/benchmark_loss_functions.py`

**Research hypotheses:**

| Hypothesis | Test | Expected |
|------------|------|----------|
| H1: Pretraining helps | BEHRT pretrained vs from scratch | Pretrained → higher C-index |
| H2: BEHRT > LSTM | Best BEHRT vs LSTM | BEHRT → 5–10% higher C-index |
| H3: LoRA is efficient | LoRA vs Full fine-tune | Similar C-index, 80–90% fewer params |
| H4: BEHRT generalizes | Train-val gap comparison | BEHRT → smaller gap |

---

## Expected Performance

Based on design analysis (to be validated empirically):

| Model | C-index | Convergence | Trainable Params |
|-------|---------|-------------|-----------------|
| LSTM Baseline | 0.67–0.72 | 50–100 epochs | ~5M |
| BEHRT (Frozen) | 0.70–0.75 | 20–30 epochs | ~17K |
| BEHRT (LoRA r=16) | 0.73–0.78 | 30–50 epochs | ~100K |
| BEHRT (Full) | 0.74–0.80 | 40–80 epochs | ~495K |

**Note:** These are projections based on architecture properties. Actual results depend on data quality, cohort size, and pretraining quality. Run `validate_behrt_survival.py` and `train_behrt_survival.py` to obtain empirical numbers.

---

**See also:** `dev/workflow/BEHRT_SURVIVAL_ANALYSIS_DESIGN.md` for the original design specification and `examples/survival_analysis/README.md` for training script usage.
