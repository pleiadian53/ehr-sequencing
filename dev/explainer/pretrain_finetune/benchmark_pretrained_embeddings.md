
```python
def compute_metrics(probs: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> Dict:
    """Compute performance metrics.
    
    Uses macro-average (equal weight per class) as primary metric for medical codes
    where rare classes are clinically important. Also computes micro-average for
    comparison with baselines.
    
    Filters to only classes present in the dataset to avoid sklearn warnings about
    one-class problems (common with 1000 vocab but small test sets).
    """
    # Convert to numpy
    probs_np = probs.numpy()
    labels_np = labels.numpy()
    
    # Get unique classes present in labels (avoids one-class warnings)
    present_classes = np.unique(labels_np)
    n_present = len(present_classes)
    
    # Create one-hot encoding for present classes only
    labels_onehot = np.zeros((len(labels_np), n_present))
    for i, cls in enumerate(present_classes):
        labels_onehot[labels_np == cls, i] = 1
    
    # Filter probabilities to present classes
    probs_filtered = probs_np[:, present_classes]
    
    # Compute ROC-AUC (macro and micro)
    try:
        roc_auc_macro = roc_auc_score(labels_onehot, probs_filtered, average='macro', multi_class='ovr')
    except:
        roc_auc_macro = 0.0

    ...
```


Looking at this code section, let me break down what's happening step by step:

## Code Walkthrough

### **Step 1: Convert to NumPy**
```python
probs_np = probs.numpy()  # Shape: (n_samples, vocab_size=1000)
labels_np = labels.numpy()  # Shape: (n_samples,) - single integer per sample
```

Converts PyTorch tensors to NumPy arrays for sklearn compatibility.

### **Step 2: Find Present Classes**
```python
present_classes = np.unique(labels_np)  # e.g., [5, 12, 23, 45, ..., 987]
n_present = len(present_classes)        # e.g., 847 out of 1000 possible codes
```

**Key insight:** Out of 1000 possible medical codes in your vocabulary, only ~847 actually appear in the test set. The other 153 codes have zero examples.

**Example:**
- Vocab has codes 0-999
- Test set only contains: [5, 12, 23, 45, 67, ..., 987] (847 unique codes)
- Missing codes like [0, 1, 2, 3, 4, 6, ...] never appear

### **Step 3: Create Filtered One-Hot Encoding**
```python
labels_onehot = np.zeros((len(labels_np), n_present))  # Shape: (n_samples, 847)
for i, cls in enumerate(present_classes):
    labels_onehot[labels_np == cls, i] = 1
```

**What this does:** Creates a one-hot matrix but **only for the 847 present classes**, not all 1000.

**Example with toy data:**
```python
# Say we have 5 samples and vocab_size=1000
labels_np = [5, 12, 5, 23, 12]  # Only 3 unique codes appear: [5, 12, 23]
present_classes = [5, 12, 23]   # n_present = 3

# OLD way (causes warnings):
labels_onehot_old = np.zeros((5, 1000))  # Most columns are all zeros!
labels_onehot_old[0, 5] = 1
labels_onehot_old[1, 12] = 1
# ... columns 0,1,2,3,4,6,7,...,999 are all zeros → sklearn warning

# NEW way (no warnings):
labels_onehot = np.zeros((5, 3))  # Only 3 columns for present classes
# Map: class 5→column 0, class 12→column 1, class 23→column 2
labels_onehot[0, 0] = 1  # Sample 0 has class 5 → column 0
labels_onehot[1, 1] = 1  # Sample 1 has class 12 → column 1
labels_onehot[2, 0] = 1  # Sample 2 has class 5 → column 0
labels_onehot[3, 2] = 1  # Sample 3 has class 23 → column 2
labels_onehot[4, 1] = 1  # Sample 4 has class 12 → column 1
```

### **Step 4: Filter Probabilities**
```python
probs_filtered = probs_np[:, present_classes]  # Shape: (n_samples, 847)
```

**What this does:** Extracts only the probability columns for the 847 present classes.

**Example:**
```python
# Original probabilities: (5 samples, 1000 classes)
probs_np = [[0.001, 0.002, ..., 0.05, ..., 0.01, ..., 0.03, ...],  # Sample 0
            [0.003, 0.001, ..., 0.02, ..., 0.04, ..., 0.01, ...],  # Sample 1
            ...]

# Filtered: (5 samples, 3 classes) - only columns [5, 12, 23]
probs_filtered = [[0.05, 0.01, 0.03],  # Sample 0: probs for classes [5, 12, 23]
                  [0.02, 0.04, 0.01],  # Sample 1: probs for classes [5, 12, 23]
                  ...]
```

### **Step 5: Compute ROC-AUC**
```python
roc_auc_macro = roc_auc_score(labels_onehot, probs_filtered, average='macro', multi_class='ovr')
```

Now sklearn computes ROC-AUC for **only the 847 present classes**, not all 1000. Each class has at least one positive example, so no warnings!

## Why This Matters

**Without filtering (old way):**
- sklearn tries to compute ROC-AUC for all 1000 classes
- 153 classes have zero examples in test set
- For each of those 153 classes: `y_true = [0, 0, 0, ..., 0]` (all negative)
- sklearn: "⚠️ Only one class present, ROC-AUC undefined" × 153 warnings

**With filtering (new way):**
- sklearn only computes ROC-AUC for 847 classes that actually appear
- Each class has at least one positive example
- No warnings! ✅

## Visual Analogy

Think of it like grading an exam:

**Old way:** 
- You have 1000 possible questions
- Students only answered 847 questions
- You try to grade all 1000 → 153 questions have no answers → error

**New way:**
- You only grade the 847 questions students actually answered
- Every question has at least one answer → no errors

The macro-average is then computed over the 847 present classes, which is more meaningful than trying to include 153 classes with no data.