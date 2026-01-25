# Tutorial 2a: Refining Comorbidity Burden Proxies

**Supplementary to:** [Tutorial 2: Synthetic Data Design](TUTORIAL_02_synthetic_data_design.md)  
**Part of:** Discrete-Time Survival Analysis for EHR Sequences  
**Audience:** Researchers building survival models for EHR data

---

## Table of Contents
1. [The Problem with Aggregate Code Counts](#the-problem-with-aggregate-code-counts)
2. [Medical Code Taxonomy](#medical-code-taxonomy)
3. [Refined Approaches](#refined-approaches)
4. [Implementation Strategies](#implementation-strategies)
5. [Clinical Validity Considerations](#clinical-validity-considerations)
6. [Recommended Approach](#recommended-approach)

---

## The Problem with Aggregate Code Counts

### Current Approach

The original tutorial uses **total code count per visit** as a proxy for comorbidity burden:

```python
# From TUTORIAL_02, lines 92, 135-145
risk_factors = {
    'comorbidity': avg_codes_per_visit,  # All codes treated equally
}

def compute_comorbidity(patient_sequence):
    codes_per_visit = [visit.num_codes() for visit in patient_sequence.visits]
    avg_codes = np.mean(codes_per_visit)
    return avg_codes
```

### The Limitation

**Key Issue:** Not all medical codes are equal indicators of disease burden.

Consider two patients with 10 codes per visit:

**Patient A (High actual burden):**
- 5 ICD codes: CHF, COPD, CKD, Diabetes, Hypertension
- 3 LOINC codes: BNP, Creatinine, HbA1c (monitoring)
- 2 CPT codes: Echocardiogram, Chest X-ray (diagnostics)

**Patient B (Lower actual burden):**
- 1 ICD code: Well-controlled Type 2 Diabetes
- 2 LOINC codes: HbA1c, Lipid panel (routine monitoring)
- 7 NDC codes: Metformin, Lisinopril, Atorvastatin, Aspirin, Vitamin D, Fish oil, Multivitamin

**Observation:** Patient B has similar code count but much lower disease severity.

### Specific Concerns by Code Type

#### 1. Drug Codes (NDC/RxNorm)
**Problem:** Medication count ≠ disease severity

- **Complex regimens for "small" diseases:** 
  - Migraine prophylaxis might involve 3-4 medications
  - Mild asthma might require 2-3 inhalers
- **Polypharmacy patterns:**
  - Elderly patients on 10+ medications (preventive + treatment)
  - Drug-drug interaction management adds more drugs
- **Supplements and OTC medications:**
  - Vitamins, supplements inflate counts
  - Not indicative of morbidity

#### 2. Lab/Test Codes (LOINC)
**Problem:** More tests ≠ sicker patient

- **Monitoring intensity:**
  - Well-controlled diabetes: frequent HbA1c checks
  - Stable warfarin patient: weekly INR monitoring
- **Diagnostic workup:**
  - Healthy patient with vague symptoms: extensive testing
  - Established diagnosis: fewer confirmatory tests
- **Screening protocols:**
  - Comprehensive metabolic panel = 1 order, 14 LOINC codes
  - Doesn't indicate disease burden

#### 3. Procedure Codes (CPT/HCPCS)
**Problem:** Procedures vary wildly in invasiveness

- **Minor vs. major:**
  - 5 outpatient minor procedures ≠ 1 open-heart surgery
  - Wound dressing changes (frequent) vs. transplant (rare)
- **Preventive vs. therapeutic:**
  - Colonoscopy (screening) vs. emergency surgery
  - Both count as "1 procedure code"

#### 4. Diagnosis Codes (ICD-9/10-CM)
**Most informative:** But still has nuances

- **Severity hierarchy:**
  - "Essential hypertension" vs. "Hypertensive crisis"
  - "Type 2 diabetes" vs. "Diabetes with complications"
- **Redundancy:**
  - Multiple codes for same condition (primary + manifestations)
  - Coding practices vary by institution

---

## Medical Code Taxonomy

### Code Type Hierarchy by Informative Value

For comorbidity burden estimation:

```
Most Informative → Least Informative

1. Diagnostic codes (ICD-9/10-CM)
   - Direct disease indicators
   - Severity stratification possible
   - Strong predictor of outcomes

2. Procedure codes (CPT/HCPCS)
   - Indicates interventions
   - Complexity varies (need weighting)
   - Major procedures = high burden

3. Lab/Test codes (LOINC)
   - Indirect indicators (monitoring/diagnosis)
   - High frequency ≠ high burden
   - Pattern matters more than count

4. Medication codes (NDC/RxNorm)
   - Treatment complexity
   - Confounded by polypharmacy
   - Class matters more than count
```

### Code Type Characteristics

| Code Type | Namespace | Count Meaning | Severity Proxy | Recommended Weight |
|-----------|-----------|---------------|----------------|-------------------|
| Diagnosis | ICD-9/10 | Number of conditions | Strong | 1.0 (baseline) |
| Procedure | CPT/HCPCS | Number of interventions | Moderate | 0.7-0.9 |
| Lab/Test | LOINC | Monitoring intensity | Weak | 0.3-0.5 |
| Medication | NDC/RxNorm | Treatment complexity | Weak | 0.2-0.4 |

---

## Refined Approaches

### Approach 1: Code-Type Weighted Counts

**Principle:** Weight codes by their informative value for disease burden.

#### Implementation

```python
def compute_weighted_comorbidity(patient_sequence, code_weights):
    """
    Compute comorbidity burden using code-type-specific weights.
    
    Args:
        patient_sequence: Patient visit history
        code_weights: Dict mapping code types to weights
    
    Returns:
        Weighted average code count per visit
    """
    weighted_codes_per_visit = []
    
    for visit in patient_sequence.visits:
        # Count codes by type
        icd_count = len(visit.get_codes_by_type('ICD'))
        cpt_count = len(visit.get_codes_by_type('CPT'))
        loinc_count = len(visit.get_codes_by_type('LOINC'))
        ndc_count = len(visit.get_codes_by_type('NDC'))
        
        # Apply weights
        weighted_count = (
            code_weights['ICD'] * icd_count +
            code_weights['CPT'] * cpt_count +
            code_weights['LOINC'] * loinc_count +
            code_weights['NDC'] * ndc_count
        )
        
        weighted_codes_per_visit.append(weighted_count)
    
    return np.mean(weighted_codes_per_visit)

# Recommended weights
code_weights = {
    'ICD': 1.0,    # Diagnoses: full weight
    'CPT': 0.8,    # Procedures: high weight
    'LOINC': 0.4,  # Lab tests: moderate weight
    'NDC': 0.3     # Medications: low weight
}

# Example comparison
# Patient with 5 ICD + 3 LOINC + 7 NDC codes
unweighted = 5 + 3 + 7 = 15 codes
weighted = 1.0*5 + 0.4*3 + 0.3*7 = 8.3 effective codes
# More accurately reflects disease burden
```

#### Rationale

- **Prioritizes diagnostic information:** ICD codes get full weight
- **Down-weights less informative codes:** Labs and meds contribute less
- **Clinically interpretable:** "Effective code count" approximates true burden
- **Preserves correlation:** High-burden patients still have higher scores

---

### Approach 2: Diagnosis-Only Focus

**Principle:** Use only diagnosis codes for comorbidity burden.

#### Implementation

```python
def compute_diagnosis_burden(patient_sequence):
    """
    Compute comorbidity burden using only diagnosis codes.
    """
    icd_per_visit = []
    
    for visit in patient_sequence.visits:
        icd_codes = visit.get_codes_by_type('ICD')
        icd_per_visit.append(len(icd_codes))
    
    return np.mean(icd_per_visit)

# Normalization
norm_comorbidity = avg_icd_per_visit / 10.0  # Typical max: 10 diagnoses
```

#### Advantages
- **Maximum specificity:** Direct measure of diagnosed conditions
- **Avoids confounders:** Not affected by medication polypharmacy
- **Clear interpretation:** "Average number of diagnoses per visit"

#### Disadvantages
- **Ignores treatment intensity:** Complex procedures not captured
- **Sensitive to coding practices:** Under-coding biases downward
- **May miss severity:** Well-managed conditions with few active diagnoses

---

### Approach 3: Stratified Complexity Scores

**Principle:** Create separate scores for each code type, then combine.

#### Implementation

```python
def compute_stratified_burden(patient_sequence):
    """
    Compute separate burden scores for each code type.
    """
    # Compute type-specific averages
    avg_icd = np.mean([len(v.get_codes_by_type('ICD')) for v in visits])
    avg_cpt = np.mean([len(v.get_codes_by_type('CPT')) for v in visits])
    avg_loinc = np.mean([len(v.get_codes_by_type('LOINC')) for v in visits])
    avg_ndc = np.mean([len(v.get_codes_by_type('NDC')) for v in visits])
    
    # Normalize each separately
    norm_icd = avg_icd / 10.0      # Typical max: 10 diagnoses
    norm_cpt = avg_cpt / 5.0       # Typical max: 5 procedures
    norm_loinc = avg_loinc / 15.0  # Typical max: 15 lab tests
    norm_ndc = avg_ndc / 10.0      # Typical max: 10 medications
    
    # Create composite score
    complexity_scores = {
        'diagnostic': norm_icd,
        'procedural': norm_cpt,
        'monitoring': norm_loinc,
        'therapeutic': norm_ndc
    }
    
    # Weighted combination
    burden_score = (
        0.50 * complexity_scores['diagnostic'] +
        0.25 * complexity_scores['procedural'] +
        0.15 * complexity_scores['monitoring'] +
        0.10 * complexity_scores['therapeutic']
    )
    
    return burden_score, complexity_scores

# Returns both composite and component scores
total_burden, components = compute_stratified_burden(patient_sequence)
```

#### Advantages
- **Multidimensional view:** Captures different aspects of complexity
- **Interpretable components:** Can analyze which dimension drives risk
- **Flexible weighting:** Easy to adjust based on outcome of interest

#### Use Case
```python
# Example: Different weights for different outcomes

# Mortality risk: emphasize diagnoses and procedures
mortality_weights = {'diagnostic': 0.5, 'procedural': 0.3, 
                     'monitoring': 0.1, 'therapeutic': 0.1}

# Readmission risk: emphasize therapeutic complexity
readmission_weights = {'diagnostic': 0.3, 'procedural': 0.2,
                       'monitoring': 0.2, 'therapeutic': 0.3}
```

---

### Approach 4: Clinical Condition Counting (Gold Standard)

**Principle:** Map ICD codes to distinct clinical conditions using hierarchical groupers.

#### Implementation

```python
def compute_condition_count(patient_sequence, grouper='CCS'):
    """
    Count distinct clinical conditions using hierarchical grouper.
    
    Groupers:
        - CCS (Clinical Classifications Software): ~300 categories
        - AHRQ Elixhauser: 31 comorbidity categories
        - Charlson: 17 weighted comorbidity categories
    """
    all_icd_codes = []
    for visit in patient_sequence.visits:
        all_icd_codes.extend(visit.get_codes_by_type('ICD'))
    
    # Map to condition categories
    conditions = set()
    for icd_code in all_icd_codes:
        category = grouper.map_to_category(icd_code)
        conditions.add(category)
    
    # Count distinct conditions
    num_conditions = len(conditions)
    
    return num_conditions

# Example using Elixhauser comorbidities
elixhauser_conditions = [
    'CHF', 'Arrhythmia', 'Valvular', 'PVD', 'HTN',
    'Paralysis', 'Neuro', 'Pulmonary', 'DM', 'Hypothyroid',
    'Renal', 'Liver', 'PUD', 'HIV', 'Lymphoma',
    'Mets', 'Tumor', 'Rheumatoid', 'Coagulopathy', 'Obesity',
    'WeightLoss', 'Fluid', 'Anemia', 'BloodLoss', 'Alcohol',
    'Drug', 'Psychoses', 'Depression'
]

# Patient has ICD codes mapping to:
icd_codes = ['I50.9', 'I48.91', 'E11.9', 'N18.3']
# Maps to:
conditions = ['CHF', 'Arrhythmia', 'DM', 'Renal']
# Condition count = 4 (not 4 ICD codes, but 4 distinct conditions)
```

#### Advantages
- **Clinically meaningful:** Aligns with medical understanding
- **Reduces redundancy:** Multiple ICD codes for same condition = 1
- **Standard metrics:** Can use validated comorbidity indices
- **Best reflects actual burden:** Gold standard in epidemiology

#### Disadvantages
- **Requires mapping:** Need grouper software/lookup tables
- **Implementation complexity:** More code infrastructure
- **Less granular:** Loses within-condition severity variation

#### Recommended Groupers

1. **CCS (Clinical Classifications Software)** - ICD-9/10-CM
   - ~300 categories (e.g., "Heart failure", "Diabetes")
   - Good for general comorbidity counting
   - Free from AHRQ

2. **Elixhauser Comorbidity Index**
   - 31 comorbidity categories
   - Validated for mortality/readmission prediction
   - Widely used in health services research

3. **Charlson Comorbidity Index**
   - 17 weighted comorbidity categories
   - Weights reflect mortality impact
   - Original index: sum weights (not just count)

---

## Implementation Strategies

### Strategy 1: Phased Refinement (Recommended)

**Phase 1: Baseline (Current)**
```python
# Start with aggregate count
comorbidity = avg_codes_per_visit
```

**Phase 2: Type Weighting**
```python
# Add code-type weights
comorbidity = weighted_avg_codes_per_visit
code_weights = {'ICD': 1.0, 'CPT': 0.8, 'LOINC': 0.4, 'NDC': 0.3}
```

**Phase 3: Diagnosis Focus**
```python
# Refine to diagnosis-heavy
comorbidity = 0.7 * avg_icd + 0.3 * weighted_other
```

**Phase 4: Condition Grouping**
```python
# Ultimate: distinct conditions
comorbidity = num_distinct_conditions / 10.0  # Normalize
```

### Strategy 2: Sensitivity Analysis

**Compare approaches on same data:**

```python
# Compute multiple comorbidity metrics
metrics = {
    'raw_count': compute_comorbidity(seq),
    'weighted': compute_weighted_comorbidity(seq, weights),
    'icd_only': compute_diagnosis_burden(seq),
    'condition_count': compute_condition_count(seq)
}

# Train models with each metric
for metric_name, comorbidity_values in metrics.items():
    outcomes = generate_outcomes(sequences, comorbidity=comorbidity_values)
    model = train_model(sequences, outcomes)
    performance[metric_name] = evaluate_model(model)

# Compare: which metric gives best risk stratification?
print(f"AUC by metric:\n{performance}")
```

### Strategy 3: Ensemble Risk Score

**Combine multiple metrics:**

```python
def compute_ensemble_comorbidity(patient_sequence):
    """
    Combine multiple comorbidity proxies.
    """
    # Compute variants
    raw = avg_codes_per_visit / 20.0
    weighted = weighted_avg / 15.0
    icd_only = avg_icd / 10.0
    
    # Ensemble with weights
    ensemble = (
        0.2 * raw +       # Keep some raw signal
        0.3 * weighted +  # Moderate weight
        0.5 * icd_only    # Emphasize diagnoses
    )
    
    return ensemble
```

---

## Clinical Validity Considerations

### 1. Face Validity

**Question:** Does the metric align with clinical intuition?

**Test cases:**
```python
# Patient A: 10 serious diagnoses (CHF, COPD, CKD, etc.)
# Patient B: 1 diagnosis + 20 vitamins/supplements

# Which has higher comorbidity?
# Answer: Patient A

# Check your metric:
assert comorbidity_A > comorbidity_B, "Metric fails face validity"
```

### 2. Predictive Validity

**Question:** Does the metric predict actual outcomes?

```python
# On real labeled data (if available)
correlation = np.corrcoef(comorbidity_scores, actual_mortality)[0,1]

# Strong correlation (|r| > 0.3) = good predictive validity
# Weak correlation (|r| < 0.15) = reconsider metric
```

### 3. Construct Validity

**Question:** Does the metric correlate with established indices?

```python
# Compare to Charlson or Elixhauser score
charlson_scores = compute_charlson_index(patients)
your_scores = compute_comorbidity(patients)

correlation = np.corrcoef(charlson_scores, your_scores)[0,1]

# High correlation (r > 0.6) = good construct validity
```

### 4. Discriminative Ability

**Question:** Does the metric separate high-risk from low-risk patients?

```python
# Compare score distributions
high_risk_patients = patients[actual_events == 1]
low_risk_patients = patients[actual_events == 0]

mean_high = comorbidity[high_risk_patients].mean()
mean_low = comorbidity[low_risk_patients].mean()

# Effect size (Cohen's d)
effect_size = (mean_high - mean_low) / pooled_std

# d > 0.5 = medium effect (adequate)
# d > 0.8 = large effect (excellent)
```

---

## Recommended Approach

### For Synthetic Data Generation

**Recommendation:** Use **Approach 1 (Code-Type Weighted Counts)** as starting point.

**Rationale:**
1. **Easy to implement:** Minor modification to existing code
2. **Clinically sound:** Prioritizes informative code types
3. **Preserves correlation:** Still captures high-risk patients
4. **Good compromise:** Between simplicity and accuracy

### Updated Risk Factor Computation

```python
def compute_risk_factors_refined(patient_sequence):
    """
    Compute risk factors with code-type-aware comorbidity.
    """
    # Code type weights
    code_weights = {
        'ICD': 1.0,    # Diagnoses
        'CPT': 0.8,    # Procedures
        'LOINC': 0.4,  # Lab tests
        'NDC': 0.3     # Medications
    }
    
    # 1. Weighted comorbidity burden
    weighted_codes = []
    for visit in patient_sequence.visits:
        visit_weighted = sum(
            code_weights.get(code.type, 0.5) * 1
            for code in visit.codes
        )
        weighted_codes.append(visit_weighted)
    
    avg_weighted_codes = np.mean(weighted_codes)
    norm_comorbidity = avg_weighted_codes / 15.0  # Normalized
    
    # 2. Visit frequency (unchanged)
    num_visits = len(patient_sequence.visits)
    time_span_years = (visits[-1].timestamp - visits[0].timestamp).days / 365.0
    frequency = num_visits / max(time_span_years, 0.1)
    norm_frequency = frequency / 5.0
    
    # 3. Diagnosis diversity (refined)
    # Use only ICD codes for diversity metric
    all_icd_codes = []
    for visit in patient_sequence.visits:
        all_icd_codes.extend([c for c in visit.codes if c.type == 'ICD'])
    
    if len(all_icd_codes) > 0:
        unique_icd = len(set(all_icd_codes))
        total_icd = len(all_icd_codes)
        icd_diversity = unique_icd / total_icd
    else:
        icd_diversity = 0.5  # Neutral if no diagnoses
    
    # Low diversity = repeated diagnoses = chronic = high risk
    risk_from_diversity = 1 - icd_diversity
    
    # 4. Combine into risk score
    risk_score = (
        0.4 * norm_comorbidity +      # Weighted code burden
        0.4 * norm_frequency +         # Visit frequency
        0.2 * risk_from_diversity      # Diagnosis diversity
    )
    
    risk_score = np.clip(risk_score, 0.1, 0.9)
    
    return risk_score
```

### Comparison: Old vs. New

```python
# Example patient: 
# - 5 ICD codes
# - 3 LOINC codes
# - 7 NDC codes
# Total = 15 codes

# OLD METHOD (unweighted)
comorbidity_old = 15 / 20.0 = 0.75

# NEW METHOD (weighted)
weighted = (1.0*5) + (0.4*3) + (0.3*7) = 5 + 1.2 + 2.1 = 8.3
comorbidity_new = 8.3 / 15.0 = 0.55

# NEW is lower → more accurate reflection of moderate (not high) burden
```

---

## Future Directions

### 1. Severity-Weighted ICD Codes

**Idea:** Not all diagnoses equal; weight by severity.

```python
# Example using ICD hierarchy
severity_weights = {
    'I50.9': 1.0,   # Heart failure (severe)
    'I10': 0.5,     # Essential HTN (moderate)
    'Z23': 0.1      # Vaccination encounter (minimal)
}

weighted_icd_burden = sum(severity_weights.get(code, 0.7) for code in icd_codes)
```

### 2. Temporal Comorbidity Evolution

**Idea:** Increasing comorbidity over time = higher risk.

```python
# Compare early vs. late visit comorbidity
early_burden = avg_comorbidity(visits[:3])
late_burden = avg_comorbidity(visits[-3:])

comorbidity_trend = late_burden - early_burden
# Positive trend = worsening = higher risk
```

### 3. Medication Class Analysis

**Idea:** Weight medications by therapeutic class.

```python
high_risk_meds = {
    'anticoagulants': 1.5,   # High risk
    'immunosuppressants': 1.3,
    'chemotherapy': 1.5,
    'insulin': 1.2,
    'opioids': 1.1,
    'vitamins': 0.1          # Low risk
}
```

### 4. Procedure Invasiveness Scoring

**Idea:** Major procedures indicate higher burden.

```python
# CPT-based invasiveness
invasiveness = {
    'major_surgery': 2.0,    # CABG, transplant
    'minor_surgery': 1.0,    # Excision, repair
    'diagnostic': 0.5,       # Endoscopy, imaging
    'minimal': 0.2           # Wound care, injections
}
```

---

## Summary

### Key Points

1. **Aggregate code counts are imperfect proxies** for comorbidity burden
2. **Different code types have different informative value:**
   - ICD (diagnoses) > CPT (procedures) > LOINC (labs) > NDC (meds)
3. **Code-type weighting is a practical refinement** (easy to implement)
4. **Condition counting is the gold standard** (requires grouper software)
5. **Clinical validation is essential** (test against known cases)

### Practical Recommendations

**For synthetic data generation:**
- Start with code-type weighted counts (Approach 1)
- Validate correlation with event times (should remain r < -0.5)
- Conduct sensitivity analysis (compare metrics)

**For real-world risk modeling:**
- Use established comorbidity indices (Charlson/Elixhauser)
- Map ICD codes to clinical conditions
- Weight by severity and prognostic impact

### Revised Risk Factor Equation

**Original (TUTORIAL_02):**
```python
comorbidity = avg_codes_per_visit / 20.0
```

**Refined (TUTORIAL_02a):**
```python
# Code-type weights
weights = {'ICD': 1.0, 'CPT': 0.8, 'LOINC': 0.4, 'NDC': 0.3}

# Weighted average
weighted_codes = sum(weights[code.type] for code in visit.codes)
comorbidity = avg_weighted_codes / 15.0
```

---

## References

1. **Elixhauser Comorbidity Index**  
   Elixhauser et al. (1998). *Medical Care*, 36(1), 8-27.

2. **Charlson Comorbidity Index**  
   Charlson et al. (1987). *Journal of Chronic Diseases*, 40(5), 373-383.

3. **Clinical Classifications Software (CCS)**  
   AHRQ Healthcare Cost and Utilization Project (HCUP).  
   https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp

4. **ICD Code Hierarchies**  
   WHO ICD-10-CM Official Guidelines for Coding and Reporting.

---

**Related Documents:**
- [Main Tutorial: Synthetic Data Design](TUTORIAL_02_synthetic_data_design.md)
- [Tutorial 1: Prediction Problem Definition](TUTORIAL_01_prediction_problem.md)
- [Tutorial 3: Loss Function Formulation](TUTORIAL_03_loss_function.md)
