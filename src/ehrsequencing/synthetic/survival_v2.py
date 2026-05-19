"""
Synthetic survival outcome generator v2 — hazard-process design.

Replaces the v1 deterministic outcome layer
(:func:`ehrsequencing.synthetic.survival.generate_survival_patient_sequences`)
with a discrete-time hazard process that depends on **what** codes appeared,
**when** they appeared, and **what followed** (interventions).

Design and rationale: see
``dev/planning/discrete_time_survival_analysis/synthetic_generator_v2.md``.

Output schema is a drop-in replacement for v1 — same
``{'visits': [...], 'outcome': {...}}`` keys consumed by
``HierarchicalSurvivalDataset`` and ``BEHRTSurvivalDataset`` — with three
optional new fields that downstream datasets ignore:

* ``state_traces`` — per-disease ordinal stage trajectories (CKD, HF, COPD)
* ``risk_trace`` — per-visit features, logit, and hazard for debugging
* ``phenotype_label`` — reserved for Phase 5 trajectory clustering

Public API
----------
:class:`HazardProcessConfig`
    Configuration dataclass with all model and noise hyperparameters.

:data:`DATA_PRESETS`
    Named presets — ``smoke`` (200 pts), ``local`` (2k), ``pod`` (20k+) —
    keyed by string. Benchmark scripts dispatch by name only so the same
    code path runs locally and on a pod.

:data:`STAGE_THRESHOLDS`
    Per-disease severity thresholds used to quantize the continuous severity
    score into clinical stages (CKD 1–5, NYHA I–IV, GOLD I–IV).

:func:`generate_hazard_process_sequences`
    Generate visit-grouped patient sequences with hazard-driven outcomes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .realistic_synthetic import DISEASE_PATTERNS, DiseasePattern, generate_realistic_dataset


# ---------------------------------------------------------------------------
# Stage scales (clinical convention)
# ---------------------------------------------------------------------------
# Thresholds are applied to the disease-specific continuous severity score.
# Stage 0 = not diagnosed; positive integers = clinical stages.
#
# CKD: 1, 2, 3a, 3b, 4, 5 (KDIGO eGFR-based) → 6 thresholds
# Heart failure: NYHA I, II, III, IV → 4 thresholds
# COPD: GOLD I, II, III, IV → 4 thresholds
STAGE_THRESHOLDS: dict[str, tuple[float, ...]] = {
    'ckd':           (0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
    'heart_failure': (0.0, 0.7, 1.4, 2.1),
    'copd':          (0.0, 0.7, 1.4, 2.1),
}


# ---------------------------------------------------------------------------
# Data-size presets (small / medium / large — same logic, single scale knob)
# ---------------------------------------------------------------------------
DATA_PRESETS: dict[str, dict[str, int]] = {
    'smoke': dict(num_patients=200,    max_visits=25, max_codes_per_visit=20, vocab_size=500),
    'local': dict(num_patients=2_000,  max_visits=50, max_codes_per_visit=30, vocab_size=1_000),
    'pod':   dict(num_patients=20_000, max_visits=80, max_codes_per_visit=40, vocab_size=2_000),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class HazardProcessConfig:
    """
    Hyperparameters for the discrete-time hazard process.

    Defaults are calibrated to give ~60–70% observed event rate at ``local``
    scale (2k patients), with a Cox-PH-on-hand-features floor in [0.62, 0.75]
    enforced by ``tests/test_survival_v2.py``.
    """

    # --- Linear predictor weights (η_t in the design doc) ---
    # Calibrated to give ~55% event rate at local scale with all coefficients
    # recoverable by time-varying Cox PH at p < 0.05.
    alpha:         float = -2.0    # baseline hazard logit (~12% baseline event rate)
    beta_chronic:  float =  0.45   # # of active chronic diseases
    beta_acute:    float =  0.60   # recency-weighted acute exposures
    beta_comorbid: float =  0.55   # weighted comorbid-pair interactions
    beta_treat:    float =  0.40   # cumulative treatment (subtracted from η)
    beta_age:      float =  0.20   # standardized age

    # --- Recency / noise ---
    tau_acute:     float =  3.0    # exp-decay constant for acute features (visits)
    sigma_frailty: float =  0.50   # per-patient frailty s.d. (added to η)

    # --- Censoring ---
    # Geometric dropout: at each visit, patient drops out with constant
    # probability `dropout_per_visit`, independent of features. This is the
    # standard non-informative-censoring assumption — dropout time is
    # marginally independent of features given (T_event, n_visits).
    #
    # E[censoring_time] = 1 / dropout_per_visit ≈ 25 visits at 0.04.
    dropout_per_visit: float = 0.04

    # --- Cohort enrichment ---
    # Real survival cohorts are selected populations (e.g., CKD-stage-2 registries),
    # not the general public. Multiplier on chronic-disease prevalences in the
    # underlying realistic generator. 1.0 = unenriched (most patients have no
    # chronic disease, Cox PH dominated by ties). 3.0 ≈ typical clinical cohort.
    enrichment_factor: float = 3.0

    # --- Disease groupings (override at construction time if needed) ---
    chronic_diseases: tuple[str, ...] = (
        'heart_failure', 'copd', 'ckd',
        'diabetes_type2', 'hypertension', 'arthritis',
    )
    acute_diseases: tuple[str, ...] = ('asthma', 'depression')
    comorbid_pairs: tuple[tuple[str, str, float], ...] = (
        ('heart_failure', 'ckd',          1.2),
        ('heart_failure', 'copd',         1.1),
        ('diabetes_type2', 'ckd',         1.0),
        ('diabetes_type2', 'hypertension', 0.6),
    )

    # --- Disease-specific staging (only diseases with a clinical stage scale) ---
    staged_diseases: tuple[str, ...] = ('ckd', 'heart_failure', 'copd')
    # Per-disease severity coefficients
    disease_progression_rate: dict[str, float] = field(default_factory=lambda: {
        'ckd':           0.08,  # slow chronic progression
        'heart_failure': 0.12,  # moderate progression
        'copd':          0.10,
    })
    disease_treatment_effect: dict[str, float] = field(default_factory=lambda: {
        'ckd':           0.04,
        'heart_failure': 0.06,
        'copd':          0.04,
    })
    disease_comorbid_coef: float = 0.30  # uniform across staged diseases

    seed: int = 42


# ---------------------------------------------------------------------------
# Code → disease lookup (built once at import time)
# ---------------------------------------------------------------------------
def _build_code_to_disease() -> dict[int, tuple[str, str]]:
    """Map every disease code to ``(disease_name, role)`` where
    ``role ∈ {'diagnosis', 'treatment', 'monitoring'}``."""
    mapping: dict[int, tuple[str, str]] = {}
    for dname, pattern in DISEASE_PATTERNS.items():
        for code in pattern.diagnosis_codes:
            mapping[int(code)] = (dname, 'diagnosis')
        for code in pattern.treatment_codes:
            mapping[int(code)] = (dname, 'treatment')
        for code in pattern.monitoring_codes:
            mapping[int(code)] = (dname, 'monitoring')
    return mapping


_CODE_TO_DISEASE: dict[int, tuple[str, str]] = _build_code_to_disease()


def _enriched_disease_patterns(
    enrichment_factor: float,
    chronic_diseases:  tuple[str, ...],
) -> dict[str, DiseasePattern]:
    """
    Return a deep copy of ``DISEASE_PATTERNS`` with chronic-disease prevalences
    multiplied by ``enrichment_factor`` (capped at 0.85 per disease).

    Mimics the selection bias of a real survival study cohort, where most
    enrolled patients have at least one chronic condition. Acute-disease
    prevalences are left unchanged.
    """
    if enrichment_factor == 1.0:
        return DISEASE_PATTERNS
    patterns = deepcopy(DISEASE_PATTERNS)
    for dname in chronic_diseases:
        if dname in patterns:
            p = patterns[dname]
            patterns[dname] = DiseasePattern(
                name=p.name,
                diagnosis_codes=p.diagnosis_codes,
                treatment_codes=p.treatment_codes,
                monitoring_codes=p.monitoring_codes,
                prevalence=min(p.prevalence * enrichment_factor, 0.85),
                age_range=p.age_range,
                progression_visits=p.progression_visits,
            )
    return patterns


# ---------------------------------------------------------------------------
# Per-patient hazard process
# ---------------------------------------------------------------------------
def _run_hazard_process(
    visits_codes: list[list[int]],
    visits_ages:  list[float],
    config:       HazardProcessConfig,
    rng:          np.random.Generator,
) -> dict:
    """
    Walk visits left-to-right; accumulate per-disease state; sample event.

    Returns a dict with ``event_time``, ``event_indicator``, ``state_traces``,
    and ``risk_trace``.
    """
    n_visits = len(visits_codes)

    # --- Per-disease tracking ---
    disease_first_dx: dict[str, int] = {}
    disease_cum_treat: dict[str, int] = {}
    acute_events: list[int] = []  # visit indices of acute-disease diagnoses

    # --- Per-visit feature accumulator: [chronic, acute, comorbid, treat, age] ---
    feats = np.zeros((n_visits, 5), dtype=np.float32)

    # --- Per-disease stage trace (0 = not dx'd; positive = stage) ---
    state_traces: dict[str, list[int]] = {d: [0] * n_visits for d in config.staged_diseases}

    frailty = float(rng.normal(0.0, config.sigma_frailty))

    for t, (v_codes, v_age) in enumerate(zip(visits_codes, visits_ages)):
        # --- Update disease tracking from codes seen at visit t ---
        for code in v_codes:
            entry = _CODE_TO_DISEASE.get(int(code))
            if entry is None:
                continue
            dname, role = entry
            if role == 'diagnosis':
                if dname not in disease_first_dx:
                    disease_first_dx[dname] = t
                    if dname in config.acute_diseases:
                        acute_events.append(t)
            elif role == 'treatment':
                disease_cum_treat[dname] = disease_cum_treat.get(dname, 0) + 1

        # --- Feature 1: chronic disease burden ---
        x_chronic = float(sum(1 for d in config.chronic_diseases if d in disease_first_dx))

        # --- Feature 2: recency-weighted acute exposure ---
        x_acute = float(sum(np.exp(-(t - t_e) / config.tau_acute) for t_e in acute_events))

        # --- Feature 3: comorbid-pair interactions ---
        x_comorbid = 0.0
        for d1, d2, weight in config.comorbid_pairs:
            if d1 in disease_first_dx and d2 in disease_first_dx:
                x_comorbid += weight

        # --- Feature 4: cumulative treatment for diagnosed diseases ---
        x_treat = float(
            sum(np.log1p(disease_cum_treat.get(d, 0)) for d in disease_first_dx)
        )

        # --- Feature 5: standardized age ---
        x_age = (v_age - 50.0) / 20.0

        feats[t] = (x_chronic, x_acute, x_comorbid, x_treat, x_age)

        # --- Per-disease stage update ---
        for d in config.staged_diseases:
            if d not in disease_first_dx:
                state_traces[d][t] = 0
                continue
            time_since_dx = t - disease_first_dx[d]
            d_comorbid = sum(
                w for d1, d2, w in config.comorbid_pairs
                if (d1 == d or d2 == d)
                and (d1 in disease_first_dx and d2 in disease_first_dx)
            )
            d_treat = float(np.log1p(disease_cum_treat.get(d, 0)))
            severity = (
                config.disease_progression_rate.get(d, 0.1) * time_since_dx
                + config.disease_comorbid_coef * d_comorbid
                - config.disease_treatment_effect.get(d, 0.05) * d_treat
            )
            thresholds = STAGE_THRESHOLDS[d]
            # searchsorted returns 0 for severity < thresholds[0], up to len for >= thresholds[-1]
            stage = int(np.searchsorted(thresholds, severity, side='right'))
            stage = max(1, min(stage, len(thresholds)))  # diagnosed → stage ≥ 1
            state_traces[d][t] = stage

    # --- Hazard process and event sampling ---
    eta = (
        config.alpha
        + config.beta_chronic  * feats[:, 0]
        + config.beta_acute    * feats[:, 1]
        + config.beta_comorbid * feats[:, 2]
        - config.beta_treat    * feats[:, 3]
        + config.beta_age      * feats[:, 4]
        + frailty
    )
    hazard = 1.0 / (1.0 + np.exp(-eta))

    draws = rng.random(n_visits) < hazard
    event_visits = np.flatnonzero(draws)
    if event_visits.size > 0:
        true_event_time = int(event_visits[0])
        has_event = True
    else:
        true_event_time = n_visits  # sentinel: no event observed
        has_event = False

    # --- Non-informative geometric dropout censoring ---
    # At each visit, the patient drops out with constant probability p. The
    # dropout time L is the first visit where dropout fires. L is bounded
    # above by n_visits - 1 (trajectory ends regardless). Crucially L is
    # sampled independently of features.
    dropout_draws = rng.random(n_visits) < config.dropout_per_visit
    dropout_visits = np.flatnonzero(dropout_draws)
    if dropout_visits.size > 0:
        L = int(dropout_visits[0])
    else:
        L = n_visits - 1

    if has_event and true_event_time <= L:
        event_time = true_event_time
        event_indicator = 1
    else:
        event_time = L
        event_indicator = 0

    event_time = int(np.clip(event_time, 0, n_visits - 1))

    return {
        'event_time': event_time,
        'event_indicator': event_indicator,
        'state_traces': state_traces,
        'risk_trace': {
            'features': feats,
            'hazard':   hazard.astype(np.float32),
            'logit':    eta.astype(np.float32),
        },
    }


# ---------------------------------------------------------------------------
# Helper: regroup flat tensor output of generate_realistic_dataset into
# per-patient visit-grouped lists.
# ---------------------------------------------------------------------------
def _to_visit_groups(
    codes_tensor, ages_tensor, visit_ids_tensor, attn_mask_tensor,
) -> list[tuple[list[list[int]], list[float]]]:
    """
    Convert the flat (N, L) tensors returned by generate_realistic_dataset
    into per-patient ``(visits_codes, visits_ages)`` tuples.
    """
    # generate_realistic_dataset returns torch tensors; convert without
    # requiring torch import here.
    codes_arr     = np.asarray(codes_tensor)
    ages_arr      = np.asarray(ages_tensor)
    visit_ids_arr = np.asarray(visit_ids_tensor)
    mask_arr      = np.asarray(attn_mask_tensor)

    num_patients = codes_arr.shape[0]
    out: list[tuple[list[list[int]], list[float]]] = []

    for i in range(num_patients):
        valid = mask_arr[i] == 1
        if not valid.any():
            out.append(([], []))
            continue
        v_codes = codes_arr[i][valid]
        v_ages  = ages_arr[i][valid]
        v_vids  = visit_ids_arr[i][valid]

        max_v = int(v_vids.max())
        visits_codes: list[list[int]] = []
        visits_ages:  list[float]     = []
        for v in range(max_v + 1):
            sel = v_vids == v
            if not sel.any():
                continue
            visits_codes.append([int(c) for c in v_codes[sel]])
            visits_ages.append(float(v_ages[sel][0]))
        out.append((visits_codes, visits_ages))

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_hazard_process_sequences(
    num_patients:        int = 2000,
    vocab_size:          int = 1000,
    max_visits:          int = 50,
    max_codes_per_visit: int = 30,
    config:              Optional[HazardProcessConfig] = None,
) -> list[dict]:
    """
    Generate visit-grouped patient sequences with hazard-process survival outcomes.

    Drop-in replacement for
    :func:`ehrsequencing.synthetic.survival.generate_survival_patient_sequences`.

    The underlying code sequences come from
    :func:`ehrsequencing.synthetic.realistic_synthetic.generate_realistic_dataset`.
    Outcomes are sampled from a discrete-time hazard process whose linear
    predictor depends on **chronic burden**, **recency-weighted acute
    exposure**, **comorbid-pair interactions**, **cumulative treatment**
    (subtracted), **age**, plus a per-patient frailty term. Censoring is
    independent of event time given covariates (non-informative).

    Args:
        num_patients: Number of synthetic patients to generate.
        vocab_size: Medical code vocabulary size.
        max_visits: Maximum visits per patient.
        max_codes_per_visit: Maximum codes per visit.
        config: Hyperparameters. Defaults to ``HazardProcessConfig()``.

    Returns:
        A list of patient dicts, each with::

            {
                'visits':   [{'codes': [...], 'age': float, 'time': float}, ...],
                'outcome':  {'event_time': int, 'event_indicator': int},
                'state_traces':    {'ckd': [...], 'heart_failure': [...], 'copd': [...]},
                'risk_trace':      {'features': np.ndarray, 'hazard': np.ndarray, 'logit': np.ndarray},
                'phenotype_label': None,
            }

    Example:
        >>> from ehrsequencing.synthetic import HazardProcessConfig, DATA_PRESETS
        >>> from ehrsequencing.synthetic import generate_hazard_process_sequences
        >>> sequences = generate_hazard_process_sequences(
        ...     **DATA_PRESETS['smoke'], config=HazardProcessConfig(seed=42),
        ... )
        >>> sequences[0]['outcome']
        {'event_time': ..., 'event_indicator': ...}
        >>> sequences[0]['state_traces']['ckd']
        [0, 0, 1, 1, 2, ...]
    """
    config = config or HazardProcessConfig()
    rng = np.random.default_rng(config.seed)

    enriched = _enriched_disease_patterns(
        config.enrichment_factor, config.chronic_diseases,
    )

    max_seq_length = max_visits * max_codes_per_visit
    codes, ages, visit_ids, attn_mask, _, _ = generate_realistic_dataset(
        num_patients=num_patients,
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        seed=config.seed,
        disease_patterns=enriched,
    )

    visit_groups = _to_visit_groups(codes, ages, visit_ids, attn_mask)

    output: list[dict] = []
    for visits_codes, visits_ages in visit_groups:
        if not visits_codes:
            # Empty patient — emit a minimal record so downstream code can
            # filter it. Mirrors v1 behaviour of skipping empties via mask.
            output.append({
                'visits':          [],
                'outcome':         {'event_time': 0, 'event_indicator': 0},
                'state_traces':    {d: [] for d in config.staged_diseases},
                'risk_trace':      {'features': np.zeros((0, 5), dtype=np.float32),
                                    'hazard':   np.zeros(0, dtype=np.float32),
                                    'logit':    np.zeros(0, dtype=np.float32)},
                'phenotype_label': None,
            })
            continue

        result = _run_hazard_process(visits_codes, visits_ages, config, rng)

        visits_out = [
            {'codes': vc, 'age': va, 'time': float(t)}
            for t, (vc, va) in enumerate(zip(visits_codes, visits_ages))
        ]

        output.append({
            'visits':          visits_out,
            'outcome':         {
                'event_time':      result['event_time'],
                'event_indicator': result['event_indicator'],
            },
            'state_traces':    result['state_traces'],
            'risk_trace':      result['risk_trace'],
            'phenotype_label': None,
        })

    return output


__all__ = [
    'HazardProcessConfig',
    'DATA_PRESETS',
    'STAGE_THRESHOLDS',
    'generate_hazard_process_sequences',
]
