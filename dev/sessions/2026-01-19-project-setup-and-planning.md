# Session Context: EHR Sequencing + LOINC Predictor

**Date:** January 19, 2026  
**Session:** Parallel Development Setup

---

## Current Session Overview

Working on **two projects simultaneously** in this workspace:

1. **loinc-predictor** - LOINC code prediction and error correction
2. **ehr-sequencing** - Disease progression modeling (biological language model for EHR)

---

## Project Status Summary

### loinc-predictor (Primary Project)

**Location:** `/Users/pleiadian53/work/loinc-predictor`

**Current Phase:** Phase 3 - Hybrid Multi-Modal Feature Engineering

**Recent Accomplishments:**
- ✅ Phase 2 complete (Text processing, LOINC management, TF-IDF, String distance)
- ✅ Legacy code archived to `legacy/`
- ✅ Modern package structure established
- ✅ Phase 3 planning documents created
- ✅ Matchmaker & Siamese Networks methodology documented

**Next Steps:**
- Implement multi-column hybrid feature extractor
- Implement LOINC code grouping algorithm
- Build Classifier Array framework
- Integrate BioBERT embeddings
- Implement Siamese network for error correction

**Key Documents:**
- `dev/workflow/PHASE3_PLAN.md` - Detailed 6-8 week plan
- `dev/workflow/PHASE3_ROADMAP.md` - Experimental routes comparison
- `docs/methods/matchmaker-and-siamese-networks.md` - Methodology analysis

---

### ehr-sequencing (New Project)

**Location:** `/Users/pleiadian53/work/ehr-sequencing`

**Current Phase:** Phase 1 - Foundation & Data Pipeline

**Recent Accomplishments:**
- ✅ Project renamed from `temporal-phenotyping`
- ✅ Added to workspace
- ✅ Initial planning documents created (`PROJECT_SETUP.md`, `ROADMAP.md`)
- ✅ Session context documented

**Next Steps:**
- Create modern directory structure
- Set up `pyproject.toml` and `environment.yml`
- Implement data adapters (duplicate from loinc-predictor)
- Build sequence construction pipeline
- Create first exploration notebook

**Key Documents:**
- `dev/workflow/PROJECT_SETUP.md` - Complete project structure and setup
- `dev/workflow/ROADMAP.md` - 10-week development roadmap
- `dev/workflow/SESSION_CONTEXT.md` - This file

**Legacy Code:**
- Preserved in place (seqmaker/, cluster/, batchpheno/, etc.)
- Will build modern structure alongside legacy

---

## Parallel Development Strategy

### Independent but Complementary

**Shared Components (Duplicated):**
- Data adapters (Synthea, MIMIC-III)
- Clinical event schema
- Basic preprocessing utilities

**Why Duplicate?**
- Projects can evolve independently
- No circular dependencies
- Different optimization priorities
- Simpler maintenance

**loinc-predictor Focus:**
- Cross-sectional: test → LOINC code mapping
- Classification and matching
- Error correction
- Models: Classifier Array, Matchmaker, Siamese networks

**ehr-sequencing Focus:**
- Longitudinal: patient timelines
- Sequence modeling
- Disease progression
- Models: LSTM, Transformers, BEHRT, Med2Vec

---

## Development Timeline (Parallel)

| Week | loinc-predictor | ehr-sequencing |
|------|----------------|----------------|
| **1-2** | Phase 3: Feature engineering | Foundation & data pipeline |
| **3-4** | Classifier Array implementation | Code embeddings (Med2Vec) |
| **5-6** | Siamese networks | Sequence models (LSTM, Transformer) |
| **7-8** | Matchmaker completion | Disease progression models |
| **9-10** | Hybrid ensemble | Disease subtyping |

---

## Technology Stack Comparison

### loinc-predictor
```
- scikit-learn (Random Forest, ensemble methods)
- LightGBM, XGBoost (gradient boosting)
- PyTorch (Siamese networks, BioBERT)
- NLTK, spaCy (text processing)
- rapidfuzz (string distance)
```

### ehr-sequencing
```
- PyTorch (LSTM, Transformers, BEHRT)
- gensim (Med2Vec embeddings)
- lifelines (survival analysis)
- umap-learn (dimensionality reduction)
- biopython (sequence utilities)
```

---

## Key Decisions Made

### 1. Duplicate Data Adapters
**Decision:** Copy data adapter code to both projects  
**Rationale:** Independent evolution, no dependencies, simpler maintenance  
**Trade-off:** Some code duplication vs. coupling

### 2. Parallel Development
**Decision:** Develop both projects simultaneously in same session  
**Rationale:** Shared context, consistent patterns, efficient iteration  
**Capability:** AI can handle multiple projects concurrently

### 3. Project Naming
**Decision:** Rename `temporal-phenotyping` → `ehr-sequencing`  
**Rationale:** Better captures "biological language model" concept  
**Analogy:** DNA sequences → NLP → Medical code sequences

### 4. Structure Consistency
**Decision:** Use same project structure for both (mamba/conda + poetry)  
**Rationale:** Familiar patterns, easier to switch context  
**Pattern:** `src/`, `docs/`, `dev/`, `notebooks/`, `tests/`

---

## Communication Protocol

### When Working on loinc-predictor
- Reference files as `@/Users/pleiadian53/work/loinc-predictor/...`
- Update `loinc-predictor` TODO list
- Document in `loinc-predictor/dev/workflow/`

### When Working on ehr-sequencing
- Reference files as `@/Users/pleiadian53/work/ehr-sequencing/...`
- Update `ehr-sequencing` TODO list
- Document in `ehr-sequencing/dev/workflow/`

### Context Switching
- Clearly state which project is being worked on
- Summarize changes before switching
- Update both project roadmaps as needed

---

## Integration Points

### Data Flow
```
Synthea/MIMIC Data
    ↓
┌───────────────────┬───────────────────┐
│ loinc-predictor   │ ehr-sequencing    │
│ (Clean LOINC)     │ (Build Sequences) │
└───────────────────┴───────────────────┘
    ↓                       ↓
Corrected Codes  →  Enhanced Sequences
```

### Potential Synergies
- Use corrected LOINC codes from predictor in sequences
- Share evaluation metrics and visualization tools
- Cross-project benchmarking on same datasets
- Unified documentation for methods

---

## File Organization

### loinc-predictor
```
src/loincpredictor/
├── data/           # Adapters, schema
├── features/       # TF-IDF, string distance, LOINC features
├── models/         # Classifier Array, Matchmaker, Siamese
├── evaluation/     # Metrics, error analysis
└── utils/          # Text processing, helpers
```

### ehr-sequencing
```
src/ehrsequencing/
├── data/           # Adapters (duplicated), sequences
├── embeddings/     # Med2Vec, graph embeddings
├── models/         # LSTM, Transformer, BEHRT, trajectory
├── clustering/     # Disease subtyping
├── evaluation/     # Metrics, visualization
└── utils/          # Tokenization, temporal encoding
```

---

## Success Metrics

### loinc-predictor
- Accuracy > 90% on test set
- MRR > 0.85 for top-5 predictions
- Error correction > 85% of errors fixed
- Inference < 500ms per test

### ehr-sequencing
- Embedding quality: NN accuracy > 80%
- Prediction AUC > 0.85
- Clustering: Silhouette > 0.5
- Training < 1 hour for 10K patients

---

## Git Strategy

### loinc-predictor
- Branch: `main`
- Recent commit: "adding documentation"
- Next: Phase 3 feature implementations

### ehr-sequencing
- Branch: TBD (will initialize)
- Status: Modernization in progress
- Next: Initial structure commit

---

## Environment Setup

### loinc-predictor
```bash
cd ~/work/loinc-predictor
mamba activate loincpredictor
poetry install
```

### ehr-sequencing
```bash
cd ~/work/ehr-sequencing
mamba activate ehrsequencing  # To be created
poetry install  # To be set up
```

---

## Next Immediate Actions

### For ehr-sequencing (Priority)
1. Create directory structure
2. Write `pyproject.toml`
3. Write `environment.yml`
4. Create `.gitignore`
5. Initialize package structure

### For loinc-predictor (Ongoing)
1. Continue Phase 3 planning
2. Begin feature engineering implementation
3. Set up experiment tracking

---

## Notes for Future Sessions

### If Starting New Session
1. Read this file (`SESSION_CONTEXT.md`)
2. Read `PROJECT_SETUP.md` for ehr-sequencing structure
3. Read `PHASE3_PLAN.md` for loinc-predictor status
4. Check TODO lists in both projects
5. Review recent git commits

### Key Context to Remember
- Both projects share similar data sources (Synthea, MIMIC)
- Data adapters are intentionally duplicated
- Projects have different goals but complementary approaches
- Development is happening in parallel
- AI assistant can handle both projects simultaneously

---

## Questions & Decisions Log

**Q: Should we create shared data package?**  
A: No, duplicate adapters for independent evolution

**Q: Can AI handle both projects?**  
A: Yes, parallel development is manageable

**Q: Why rename to ehr-sequencing?**  
A: Better captures biological language model concept

**Q: Same structure for both?**  
A: Yes, consistency helps with context switching

---

**Document Version:** 1.0  
**Last Updated:** January 19, 2026  
**Next Review:** After Phase 1 completion (ehr-sequencing)
