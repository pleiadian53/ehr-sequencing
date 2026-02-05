```
/Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/data/domain_shift.py
```

```
src/ehrsequencing/
├── data/              # ✅ Main location for all data generation
│   ├── realistic_synthetic.py   # Base synthetic data with disease patterns
│   ├── domain_shift.py          # NEW: Domain shift scenarios
│   ├── demo_synthetic.py        # Demo data for quick tests
│   └── random_synthetic.py      # Random data (testing only)
├── synthetic/         # Separate purpose (survival analysis)
│   └── survival.py
```

---

You're right - the synthetic data generation logic is scattered and the driver script shouldn't handle domain shift configuration. Let me refactor this into a clean, reusable module structure.









Perfect! Let me create a clean, reusable module for domain-shifted datasets. Here's my design:

## Design: Clean Module Structure

**Current Issues:**
- [synthetic/](cci:9://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/synthetic:0:0-0:0) has survival data (different purpose)
- [data/](cci:9://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/data:0:0-0:0) has realistic_synthetic.py (good location)
- Domain shift logic is in benchmark script (bad - not reusable)

**Solution:**
- Keep [data/](cci:9://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/data:0:0-0:0) as the main location for synthetic data
- Create `data/domain_shift.py` for domain shift configurations
- Provide clean API: [generate_domain_shifted_datasets()](cci:1://file:///Users/pleiadian53/work/ehr-sequencing/examples/pretrain_finetune/benchmark_transfer_learning.py:72:0-180:35)
- Driver scripts just call one function

Now let me update the [__init__.py](cci:7://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/__init__.py:0:0-0:0) to export the new function:

Now let me update the benchmark script to use the clean new API:

Now let me update the main() function to use the clean API with proper arguments:







Perfect! ✅ The refactored code works. Let me commit and push the changes:







## ✅ Refactoring Complete - Clean Module Structure Implemented!

### What Was Done

**Created New Module:** [src/ehrsequencing/data/domain_shift.py](cci:7://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/data/domain_shift.py:0:0-0:0)

This centralizes all domain shift logic with a clean, reusable API. Driver scripts no longer need to worry about how synthetic data is generated or how domain shift is configured.

### Clean API Design

**Before (❌ Bad - Logic in Driver Script):**
```python
# Driver script had to manually modify DISEASE_PATTERNS
from ehrsequencing.data.realistic_synthetic import DISEASE_PATTERNS
original_patterns = {}
for disease_name, pattern in DISEASE_PATTERNS.items():
    original_patterns[disease_name] = {...}
    pattern.prevalence *= 0.6  # Manual modification
    pattern.age_range = (...)  # Manual modification
# ... 100+ lines of domain shift logic ...
```

**After (✅ Good - Clean API):**
```python
# Driver script just calls one function!
from ehrsequencing.data import generate_domain_shifted_datasets

source_data, target_data = generate_domain_shifted_datasets(
    source_patients=10000,
    target_patients=5000,
    scenario='general_to_elderly'  # Pre-configured
)
```

### Pre-configured Scenarios

1. **`general_to_elderly`** (default)
   - Source: Younger population (20-60 yrs), 40% lower disease rates
   - Target: Older population (50-90 yrs), 80% higher disease rates

2. **`hospital_a_to_b`**
   - Source: Urban hospital, diverse population
   - Target: Rural hospital, 30% higher disease rates

3. **`historical_to_recent`**
   - Source: 2010-2015 data, older treatment patterns
   - Target: 2016-2020 data, modern patterns, aging population

### Module Organization Clarified

```
src/ehrsequencing/
├── data/              # ✅ Main location for all data generation
│   ├── realistic_synthetic.py   # Base synthetic data with disease patterns
│   ├── domain_shift.py          # NEW: Domain shift scenarios
│   ├── demo_synthetic.py        # Demo data for quick tests
│   └── random_synthetic.py      # Random data (testing only)
├── synthetic/         # Separate purpose (survival analysis)
│   └── survival.py
```

### Benefits

- ✅ **Driver scripts are clean** - No domain shift logic
- ✅ **Reusable** - Any script can use domain-shifted datasets
- ✅ **Extensible** - Add new scenarios without touching driver code
- ✅ **Safe** - Automatic pattern restoration (no side effects)
- ✅ **Well-documented** - Clear docstrings and examples

### Testing

✅ Tested locally with small model - all 4 runs completed successfully

### Deploy on Pod

```bash
cd /workspace/ehr-sequencing
git pull origin main  # Get all fixes and refactoring

# Re-run transfer learning with clean API
cd examples/pretrain_finetune
nohup python -u benchmark_transfer_learning.py \
    --model-size large \
    --source-patients 10000 \
    --target-patients 5000 \
    --epochs 100 \
    --finetune-epochs 20 \
    --batch-size 128 \
    --output-dir /workspace/ehr-sequencing/experiments/transfer_learning \
    > /workspace/ehr-sequencing/experiments/sessions/transfer_learning_large.out 2>&1 &
```

The benchmark script is now much cleaner and the domain shift logic is properly encapsulated in a reusable module! 🚀

---



This design is okay but note that the package src/ehrsequencing/data/ was originally design to adapt to multiple EHR/EMR data sources such as Synthea, MIMIC, etc, so that they can be processed in a systematic principled way for EHR-seq workflow and for disease phenotyping applications. 

Various types of synthetic data generations should deserve their own subpackage. And I think that's why src/ehrsequencing/synthetic was created. Isn't this true? 

---

You're absolutely right - I misunderstood the original design intent. Let me check the original purpose of these directories:

You're absolutely correct! I apologize for the confusion. Let me check the original design:

You're absolutely right - I misunderstood the original architecture. Let me verify:

You're absolutely correct! I misunderstood the architecture. Let me fix this properly:

**Original Design Intent:**
- [src/ehrsequencing/data/](cci:9://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/data:0:0-0:0) → **Real EHR data adapters** (Synthea, MIMIC, etc.)
- [src/ehrsequencing/synthetic/](cci:9://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/synthetic:0:0-0:0) → **Synthetic data generation** (survival, medical LLM, etc.)

I incorrectly put synthetic data generation in [data/](cci:9://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/data:0:0-0:0). Let me move it to the correct location:













Now let me update the benchmark scripts to use the correct imports:

Now let me check if there are other files that need import updates:





Now let me test locally to ensure the reorganization works:



Perfect! ✅ The reorganization works. Now let me commit and push:







## ✅ Package Reorganization Complete!

You were absolutely right - I misunderstood the original architecture. Thank you for catching that!

### Corrected Package Structure

**Original Design Intent:**
- **[src/ehrsequencing/data/](cci:9://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/data:0:0-0:0)** → Real EHR data adapters (Synthea, MIMIC, etc.)
  - Systematic processing for various EHR/EMR data sources
  - Disease phenotyping applications
  - Production data pipelines

- **[src/ehrsequencing/synthetic/](cci:9://file:///Users/pleiadian53/work/ehr-sequencing/src/ehrsequencing/synthetic:0:0-0:0)** → All synthetic data generation
  - Survival analysis
  - Medical LLM training data
  - Transfer learning datasets

### What Was Fixed

**Moved 4 modules to correct location:**
```
src/ehrsequencing/data/realistic_synthetic.py → src/ehrsequencing/synthetic/realistic_synthetic.py
src/ehrsequencing/data/domain_shift.py       → src/ehrsequencing/synthetic/domain_shift.py
src/ehrsequencing/data/demo_synthetic.py     → src/ehrsequencing/synthetic/demo_synthetic.py
src/ehrsequencing/data/random_synthetic.py   → src/ehrsequencing/synthetic/random_synthetic.py
```

**Updated all imports:**
```python
# Correct ✅
from ehrsequencing.synthetic import generate_realistic_dataset
from ehrsequencing.synthetic import generate_domain_shifted_datasets

# Incorrect ❌ (old)
from ehrsequencing.data import generate_realistic_dataset
```

### Final Package Organization

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

### Testing

✅ Tested locally with small model - all benchmarks work correctly with new imports

### Deploy on Pod

```bash
cd /workspace/ehr-sequencing
git pull origin main  # Get corrected package structure

# Run transfer learning benchmark
cd examples/pretrain_finetune
nohup python -u benchmark_transfer_learning.py \
    --model-size large \
    --source-patients 10000 \
    --target-patients 5000 \
    --epochs 100 \
    --finetune-epochs 20 \
    --batch-size 128 \
    --output-dir /workspace/ehr-sequencing/experiments/transfer_learning \
    > /workspace/ehr-sequencing/experiments/sessions/transfer_learning_large.out 2>&1 &
```

The package structure now correctly reflects the original design intent! 🎯



---

Just a quick feedback, I think synthetic datasets for different purposes may deserve their own subpackages. I guess this depends on if a single module is sufficient for that topic (e.g. domain shift). 

Additionally, since you've restructured the module location for data generation, we should check all the downstream dependent modules and scripts to ensure that their import statements are consistent with the updates. 

For example,  

examples/pretrain_finetune
examples/benchmarking/ 

and, notebooks series: 
docs/notebooks/01_synthea_data_exploration/ 

Where else should we look into? 

---

![image-20260204160208721](/Users/pleiadian53/Library/Application Support/typora-user-images/image-20260204160208721.png)

---

