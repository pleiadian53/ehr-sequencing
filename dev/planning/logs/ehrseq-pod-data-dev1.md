Perfect! I've created a comprehensive data setup guide and updated the documentation. Here's what I documented:

## 📄 Created: `/Users/pleiadian53/work/ehr-sequencing/docs/notebooks/DATA_SETUP.md`

This guide explains:

### ✅ Where the Data Lives

**Current notebook setup:**
```python
data_path = Path.home() / 'work' / 'loinc-predictor' / 'data' / 'synthea' / 'all_cohorts'
```

The notebooks reference Synthea data from the `loinc-predictor` project (cross-project reference to avoid duplication).

### 🚀 Quick Commands for Your Pod

**Option 1: Upload the loinc-predictor data structure (Recommended)**

```bash
# From local machine
rsync -avzP --no-perms --no-owner --no-group \
  ~/work/loinc-predictor/data/synthea/ \
  runpod-main:/workspace/loinc-predictor/data/synthea/
```

Then the notebooks will work as-is on the pod since they reference the same path structure!

**Option 2: Copy to ehr-sequencing first, then upload**

```bash
# Step 1: Copy locally (one-time)
mkdir -p ~/work/ehr-sequencing/data/synthea/all_cohorts
cp -r ~/work/loinc-predictor/data/synthea/all_cohorts/* \
  ~/work/ehr-sequencing/data/synthea/all_cohorts/

# Step 2: Upload to pod
rsync -avzP --no-perms --no-owner --no-group \
  ~/work/ehr-sequencing/data/synthea/ \
  runpod-main:/workspace/ehr-sequencing/data/synthea/
```

Then update notebook paths from `loinc-predictor` to `ehr-sequencing`.

### 📚 What's Documented

The guide covers:
- ✅ **3 local setup options** (shared data, copy data, generate new)
- ✅ **2 pod deployment options** (rsync or generate on pod)
- ✅ **Data requirements** (required CSV files, minimum dataset sizes)
- ✅ **Path update strategies** (single notebook, bulk update, environment variables)
- ✅ **Troubleshooting** (common errors and solutions)
- ✅ **Quick reference commands** (copy-paste ready)

### 📍 Also Updated

`/Users/pleiadian53/work/ehr-sequencing/docs/notebooks/README.md` now prominently links to the new DATA_SETUP.md guide in the "Data Requirements" section.

---

**Next step**: Upload the loinc-predictor data to your pod using Option 1 above! 🎯