# BEHRT Survival Tutorial Series (6 sessions)

This sequence is designed for a workshop/tutorial format.

## Sessions

1. [Session 1 — EHR -> tokens -> tensors](./session_01_ehr_to_tokens_to_tensors.md)
2. [Session 2 — BEHRT embeddings as structured inductive bias](./session_02_behrt_embeddings_and_inductive_bias.md)
3. [Session 3 — BEHRT pretraining objectives](./session_03_pretraining_objectives.md)
4. [Session 4 — Survival head + visit aggregation](./session_04_survival_head_and_visit_aggregation.md)
5. [Session 5 — Losses as value systems](./session_05_losses_as_value_systems.md)
6. [Session 6 — Optimization strategy (frozen vs LoRA vs full)](./session_06_optimization_strategies.md)

## Suggested pacing

- Session 1–2: data/representation foundations
- Session 3–4: objective and architecture bridge to survival
- Session 5–6: optimization philosophy and practical training choices

## Relevant code paths

- `src/ehrsequencing/data/behrt_survival_dataset.py`
- `src/ehrsequencing/models/embeddings.py`
- `src/ehrsequencing/models/behrt.py`
- `src/ehrsequencing/models/behrt_survival.py`
- `src/ehrsequencing/models/losses.py`
- `src/ehrsequencing/models/lora.py`
- `examples/survival_analysis/train_behrt_survival.py`
