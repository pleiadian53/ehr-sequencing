```
src/ehrsequencing/
├── data/              # Real EHR data adapters ONLY
│   ├── adapters/      # Synthea, MIMIC, etc.
│   │   ├── base.py
│   │   └── synthea.py
│   ├── visit_grouper.py
│   └── sequence_builder.py
│
├── synthetic/         # All synthetic data generation
│   ├── survival.py              # Survival analysis
│   ├── realistic_synthetic.py   # Medical LLM training
│   ├── domain_shift.py          # Transfer learning
│   ├── demo_synthetic.py        # Quick demos
│   └── random_synthetic.py      # Baseline comparison
```



Future 

```
src/ehrsequencing/synthetic/
├── medical_llm/       # Medical LLM-specific generators
├── survival/          # Expanded survival analysis
├── phenotyping/       # Disease phenotyping datasets
└── fairness/          # Bias evaluation datasets
```



---

