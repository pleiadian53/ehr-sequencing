# Methods Documentation

This directory contains detailed documentation on the methodological approaches used in the EHR sequencing framework.

## Disease Progression Modeling

### Causal Survival Analysis (2-Part Tutorial)

**Why this matters:** Simple classification tasks like "does this patient have diabetes?" can often be solved with rule-based methods. The real power of sequence modeling emerges when predicting **disease progression** over time.

#### [Part 1: Causal Progression Labels](causal-survival-analysis-1.md)

**Topics covered:**
- The most dangerous data leakage pattern in temporal prediction
- Why patient-level labels + visit-level inputs = temporal leakage
- Three diagnostic tests to detect leakage
- Designing progression labels that respect causality
- Three approaches: fixed-horizon, discrete-time survival, continuous-time survival

**Key insight:** If a model can "predict" an outcome before clinical evidence exists, you have leakage. High AUC is not a virtue—causality is.

#### [Part 2: Discrete-Time Survival Modeling](causal-survival-analysis-2.md)

**Topics covered:**
- What discrete-time survival modeling actually means
- Understanding censoring (conceptually and operationally)
- Deriving the likelihood formula from first principles
- PyTorch implementation of the survival loss
- Why this is causal by construction

**Key insight:** Visits are the natural discretization for EHR data. Discrete-time survival modeling fits perfectly with visit-based sequences.

**Implementation:**
- Loss function: `src/ehrsequencing/models/losses.py::DiscreteTimeSurvivalLoss`
- Model: `src/ehrsequencing/models/survival_lstm.py::DiscreteTimeSurvivalLSTM`
- Training script: `examples/train_survival_lstm.py`

---

## Other Methodological Topics

### [Within-Visit Structure](within-visit-structure.md)

How to handle multiple medical codes within a single visit:
- Bag-of-codes (mean pooling)
- Attention mechanisms
- Hierarchical encoding
- Set-based representations

### [LSTM Variable-Length Analysis](lstm-variable-length-analysis.md)

Technical details on handling variable-length sequences in LSTMs:
- Padding and masking strategies
- PackedSequence optimization
- Memory considerations
- Batch processing

---

## Quick Reference

### When to Use Each Approach

| Task | Approach | Why |
|------|----------|-----|
| **Static classification** | Logistic regression, simple NN | No temporal dynamics needed |
| **Fixed-horizon prediction** | LSTM + BCE loss | Simple, interpretable, requires careful censoring |
| **Disease progression** | LSTM + discrete-time survival | Natural for visits, handles censoring, causal |
| **Irregular timing** | Cox-style continuous-time | Flexible timing, standard in epidemiology |
| **Multiple outcomes** | Competing risks survival | Multiple event types, only one can occur first |

### Evaluation Metrics

- **Classification**: AUC, precision, recall, calibration
- **Survival**: Concordance index (C-index), calibration, survival curves
- **Temporal**: Time-dependent AUC, Brier score

### Common Pitfalls

1. **Temporal leakage**: Using future information in predictions
2. **Censoring as negative**: Treating censored patients as "no event"
3. **Ignoring visit frequency**: Confounding surveillance with risk
4. **Patient-level labels**: Losing temporal resolution
5. **No diagnostic tests**: Not verifying causality

---

## Getting Started

### For Researchers

1. Read [Part 1](causal-survival-analysis-1.md) to understand the leakage problem
2. Read [Part 2](causal-survival-analysis-2.md) for implementation details
3. Review the training script: `examples/train_survival_lstm.py`
4. Adapt to your specific outcome and dataset

### For Practitioners

1. Start with the training script: `examples/train_survival_lstm.py`
2. Modify the `create_survival_labels()` function for your outcome
3. Adjust model hyperparameters as needed
4. Evaluate with C-index and calibration plots

### For Students

1. Work through the tutorials in order
2. Implement the loss function from scratch (good exercise!)
3. Compare discrete-time vs. fixed-horizon on the same data
4. Run the diagnostic tests on your model

---

## References

### Survival Analysis

- Singer, J. D., & Willett, J. B. (2003). *Applied Longitudinal Data Analysis: Modeling Change and Event Occurrence*. Oxford University Press.
- Tutz, G., & Schmid, M. (2016). *Modeling Discrete Time-to-Event Data*. Springer.
- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187-202.

### EHR Sequence Modeling

- Choi, E., et al. (2016). RETAIN: An interpretable predictive model for healthcare using reverse time attention mechanism. *NeurIPS*.
- Rajkomar, A., et al. (2018). Scalable and accurate deep learning with electronic health records. *npj Digital Medicine*.
- Steinberg, E., et al. (2021). Language models are an effective representation learning technique for electronic health record data. *Journal of Biomedical Informatics*.

### Temporal Causality

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.

---

## Contributing

Found an error or have a suggestion? Please open an issue or submit a pull request.

When adding new methods documentation:
1. Include mathematical derivations with intuition
2. Provide PyTorch implementation examples
3. Explain when to use vs. not use the method
4. Add references to key papers
5. Include common pitfalls and debugging tips
