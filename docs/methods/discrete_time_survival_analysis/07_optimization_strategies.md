# Optimization Strategies: Frozen vs LoRA vs Full Fine-Tuning

**Topics:** Training strategy comparison, parameter tradeoffs, practical recipes by data regime

**Reference code:** `src/ehrsequencing/models/behrt_survival.py`, `src/ehrsequencing/models/lora.py`, `examples/survival_analysis/train_behrt_survival.py`

---

## Table of Contents

1. [The Three Strategies](#the-three-strategies)
2. [Parameter Tradeoff Intuition](#parameter-tradeoff-intuition)
3. [Practical Recipe by Data Regime](#practical-recipe-by-data-regime)
4. [Operational Notes](#operational-notes)
5. [Decision Summary](#decision-summary)

---

## The Three Strategies

### A) Frozen Encoder (`freeze_behrt=True`)

Train only the survival head. All BEHRT encoder weights are fixed.

```python
for param in model.behrt.parameters():
    param.requires_grad = False
# Only hazard_head parameters are updated
```

**Use when:**
- Very small cohorts (< 2K patients)
- Fast baselines and ablations
- Verifying the pipeline before committing to longer training

**Characteristics:**
- Fastest training, most stable
- Lowest adaptation capacity — relies entirely on pretrained representations
- Best when pretrained BEHRT was trained on similar EHR data

### B) LoRA (`use_lora=True`)

Freeze base BEHRT weights; inject trainable low-rank adapter matrices into attention layers. Optionally also train embeddings and the hazard head.

```python
from ehrsequencing.models.lora import apply_lora_to_behrt
model.behrt = apply_lora_to_behrt(model.behrt, rank=16, alpha=32)
```

**Use when:**
- Medium data scale (2K–20K patients)
- Limited GPU memory
- Need better adaptation than frozen but want to avoid full fine-tuning risk

**Characteristics:**
- Strong efficiency/performance tradeoff
- Typically 10–20% of full fine-tuning parameter count
- Usually the best default for practical transfer learning

### C) Full Fine-Tuning

All BEHRT parameters are trainable.

```python
# All parameters trainable by default — no special setup needed
```

**Use when:**
- Large cohorts (> 20K patients) with strong regularization
- Sufficient compute and tuning budget
- LoRA has saturated and you need more adaptation capacity

**Characteristics:**
- Highest capacity, highest compute, highest overfitting risk
- Requires careful early stopping and validation monitoring

---

## Parameter Tradeoff Intuition

```
Trainable parameters:  Frozen << LoRA << Full

As trainable count rises:
  ↑ adaptation flexibility
  ↑ compute and memory
  ↑ overfitting risk
```

Use `model.get_trainable_parameters()` to verify the actual training footprint before starting a run:

```python
params = model.get_trainable_parameters()
print(f"Trainable: {params['trainable']:,} / {params['total']:,}")
```

---

## Practical Recipe by Data Regime

### Small cohort (< 2K patients)

```
Strategy:   Frozen encoder
Loss:       Hybrid (lambda_rank=0.05)
Epochs:     50–100 with strong early stopping (patience=10)
Batch size: 16–32
If underfitting: move to LoRA rank 8–16
```

### Medium cohort (2K–20K patients)

```
Strategy:   LoRA rank 16
Loss:       Hybrid (lambda_rank=0.1 as initial)
Epochs:     100 with early stopping (patience=15)
Batch size: 32–64
Regularization: gradient clipping (max_norm=1.0) + weight decay (1e-5)
```

### Large cohort (> 20K patients)

```
Strategy:   Compare LoRA vs full fine-tune
Loss:       Hybrid (lambda_rank=0.1–0.2)
Epochs:     100–200 with checkpoint selection by C-index + calibration
Batch size: 64–128
If LoRA saturates: escalate to full fine-tune
```

---

## Operational Notes

- **Verify training footprint** before each run with `get_trainable_parameters()`
- **Keep preprocessing consistent** across train/val/test — never compare strategies with different truncation or masking pipelines
- **Monitor both metrics** — C-index (discrimination) and calibration proxy (Brier score or risk curve sanity)
- **Early stopping criterion** — use validation C-index, not validation loss, as the primary stopping signal
- **Checkpoint selection** — save the checkpoint with best val C-index, not the final epoch

---

## Decision Summary

| Strategy | Data Scale | Trainable Params | Risk | Recommended When |
|----------|-----------|-----------------|------|-----------------|
| Frozen | < 2K | Head only (~17K) | Low | Fast baseline, tiny data |
| LoRA rank 16 | 2K–20K | ~10–20% of full | Medium | **Default choice** |
| Full fine-tune | > 20K | All (~495K+) | High | Large data, LoRA saturated |

**Default recommendation for this project: LoRA-first.** Escalate to full fine-tune only when data scale and validation evidence justify it.

---

**Next:** `08_architecture_decisions.md` — BEHRT vs LSTM comparison and broader architectural tradeoffs.
