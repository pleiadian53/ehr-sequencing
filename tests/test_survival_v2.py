"""
Sanity tests for the v2 hazard-process synthetic survival generator.

These tests verify that the generator produces data with the structural
properties claimed in the design doc
(``dev/planning/discrete_time_survival_analysis/synthetic_generator_v2.md``):

* Schema conformance — output dicts have all required keys.
* Event rate in target band — calibrated to ~50–65 % at local scale.
* **Signal exists**: stratified event rates rise monotonically with chronic
  disease burden; rate ratio (chronic ≥ 2) / (chronic = 0) ≥ 1.5×.
* **Time-varying Cox PH** recovers the linear-predictor coefficients with
  the correct sign and p < 0.05 — i.e. the features that drive the hazard
  process are statistically detectable as such.
* **Non-informative censoring**: a time-fixed Cox PH on shuffled event
  indicators gives a C-index near 0.5 (the n_visits-feature confound is
  controlled by geometric dropout).
* **Recency matters** — the acute-disease coefficient (recency-weighted)
  has a larger magnitude than chronic (cumulative).
* **Intervention reduces hazard** — treatment coefficient is negative.

The tests use ``lifelines`` for Cox PH fitting and concordance computation;
``ehrsequencing`` ships it as a dev dependency.

To run:
    pytest tests/test_survival_v2.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

# lifelines emits a lot of harmless warnings; silence them at module load
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from lifelines import CoxPHFitter, CoxTimeVaryingFitter  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402

from ehrsequencing.synthetic import (  # noqa: E402
    DATA_PRESETS,
    HazardProcessConfig,
    STAGE_THRESHOLDS,
    generate_hazard_process_sequences,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def local_sequences() -> list[dict]:
    """2k patients at the ``local`` scale — used by every signal-detection test."""
    return generate_hazard_process_sequences(
        **DATA_PRESETS["local"], config=HazardProcessConfig(seed=42),
    )


@pytest.fixture(scope="module")
def time_varying_panel(local_sequences) -> pd.DataFrame:
    """(start, stop, event, x_t) panel for time-varying Cox PH."""
    rows: list[dict] = []
    for pid, s in enumerate(local_sequences):
        if not s["visits"]:
            continue
        T = s["outcome"]["event_time"]
        E = s["outcome"]["event_indicator"]
        feats = s["risk_trace"]["features"]
        for t in range(min(T + 1, len(feats))):
            rows.append(dict(
                id=pid, start=t, stop=t + 1,
                event=int(E and t == T),
                chronic=float(feats[t, 0]), acute=float(feats[t, 1]),
                comorbid=float(feats[t, 2]), treat=float(feats[t, 3]),
                age=float(feats[t, 4]),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class TestSchema:
    def test_smoke_runs(self):
        seqs = generate_hazard_process_sequences(
            **DATA_PRESETS["smoke"], config=HazardProcessConfig(seed=0),
        )
        assert len(seqs) == DATA_PRESETS["smoke"]["num_patients"]

    def test_output_dict_keys(self, local_sequences):
        required = {"visits", "outcome", "state_traces", "risk_trace", "phenotype_label"}
        for s in local_sequences[:50]:
            assert required.issubset(s.keys()), f"missing keys: {required - s.keys()}"

    def test_outcome_format(self, local_sequences):
        for s in local_sequences:
            if not s["visits"]:
                continue
            assert "event_time" in s["outcome"]
            assert "event_indicator" in s["outcome"]
            assert s["outcome"]["event_indicator"] in (0, 1)
            assert 0 <= s["outcome"]["event_time"] < len(s["visits"])

    def test_state_traces_present(self, local_sequences):
        for s in local_sequences[:50]:
            traces = s["state_traces"]
            for disease in STAGE_THRESHOLDS:
                assert disease in traces, f"state trace missing for {disease}"
                if s["visits"]:
                    assert len(traces[disease]) == len(s["visits"])

    def test_state_trace_values_in_range(self, local_sequences):
        for s in local_sequences:
            for disease, thresholds in STAGE_THRESHOLDS.items():
                trace = s["state_traces"].get(disease, [])
                for stage in trace:
                    assert 0 <= stage <= len(thresholds), (
                        f"{disease} stage {stage} out of range "
                        f"[0, {len(thresholds)}]"
                    )

    def test_risk_trace_shape(self, local_sequences):
        for s in local_sequences[:50]:
            if not s["visits"]:
                continue
            rt = s["risk_trace"]
            n_visits = len(s["visits"])
            assert rt["features"].shape == (n_visits, 5)
            assert rt["hazard"].shape == (n_visits,)
            assert rt["logit"].shape == (n_visits,)
            assert np.all((rt["hazard"] >= 0) & (rt["hazard"] <= 1))


# ---------------------------------------------------------------------------
# Marginal statistics
# ---------------------------------------------------------------------------
class TestMarginalStatistics:
    def test_event_rate_in_band(self, local_sequences):
        """Calibrated default config targets ~50–65 % event rate."""
        er = np.mean([s["outcome"]["event_indicator"] for s in local_sequences])
        assert 0.45 < er < 0.70, f"event rate {er:.2%} outside [0.45, 0.70]"

    def test_event_time_distribution(self, local_sequences):
        """Events should not all happen at visit 0 or all at the last visit."""
        ets = np.array([
            s["outcome"]["event_time"] for s in local_sequences
            if s["outcome"]["event_indicator"] == 1
        ])
        assert ets.std() > 1.0, "event times have collapsed to a single value"
        assert np.percentile(ets, 5) >= 0
        # Median event time should be in the early-to-middle range
        assert 1 <= np.median(ets) <= 15


# ---------------------------------------------------------------------------
# Signal-detection tests — the core claim of v2 is "signal exists & is learnable"
# ---------------------------------------------------------------------------
class TestSignalExists:
    def test_stratified_event_rate_monotone(self, local_sequences):
        """Event rate should rise monotonically with chronic-disease burden."""
        chronic_max = np.array([
            int(s["risk_trace"]["features"][:, 0].max())
            if s["risk_trace"]["features"].shape[0] > 0 else 0
            for s in local_sequences
        ])
        E = np.array([s["outcome"]["event_indicator"] for s in local_sequences])
        rates = []
        for k in (0, 1, 2):
            sel = chronic_max == k
            if sel.sum() >= 30:  # enough samples
                rates.append(E[sel].mean())
        # At least the first three buckets should be monotone increasing
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1)), (
            f"event rate not monotone in chronic burden: {rates}"
        )

    def test_chronic_burden_rate_ratio(self, local_sequences):
        """High-burden (chronic ≥ 2) patients should have ≥ 1.5× event rate vs none."""
        chronic_max = np.array([
            int(s["risk_trace"]["features"][:, 0].max())
            if s["risk_trace"]["features"].shape[0] > 0 else 0
            for s in local_sequences
        ])
        E = np.array([s["outcome"]["event_indicator"] for s in local_sequences])
        high = E[chronic_max >= 2].mean()
        low = E[chronic_max == 0].mean()
        assert low > 0
        assert high / low >= 1.5, f"rate ratio {high / low:.2f}× < 1.5×"


# ---------------------------------------------------------------------------
# Time-varying Cox PH — recovers the linear predictor
# ---------------------------------------------------------------------------
class TestTimeVaryingCoxPH:
    @pytest.fixture(scope="class")
    def fitted_model(self, time_varying_panel) -> CoxTimeVaryingFitter:
        ctv = CoxTimeVaryingFitter(penalizer=0.01)
        ctv.fit(
            time_varying_panel,
            id_col="id", event_col="event",
            start_col="start", stop_col="stop",
        )
        return ctv

    def test_chronic_coef_positive_and_significant(self, fitted_model):
        summary = fitted_model.summary.loc["chronic"]
        assert summary["coef"] > 0, f"chronic coef should be positive, got {summary['coef']}"
        assert summary["p"] < 0.05, f"chronic p={summary['p']} not significant"

    def test_acute_coef_positive_and_significant(self, fitted_model):
        summary = fitted_model.summary.loc["acute"]
        assert summary["coef"] > 0
        assert summary["p"] < 0.05

    def test_treatment_coef_negative_and_significant(self, fitted_model):
        """Intervention should reduce hazard — coef must be negative."""
        summary = fitted_model.summary.loc["treat"]
        assert summary["coef"] < 0, f"treatment coef should be negative, got {summary['coef']}"
        assert summary["p"] < 0.05

    def test_acute_dominates_chronic(self, fitted_model):
        """Recency-weighted acute feature should have higher exp(coef) than the
        cumulative chronic feature — the recency design choice should be
        statistically visible."""
        chronic_hr = fitted_model.summary.loc["chronic", "exp(coef)"]
        acute_hr = fitted_model.summary.loc["acute", "exp(coef)"]
        assert acute_hr > chronic_hr, (
            f"recency design failed: chronic HR={chronic_hr:.2f}, "
            f"acute HR={acute_hr:.2f}"
        )


# ---------------------------------------------------------------------------
# Non-informative censoring — permutation null should approach 0.5
# ---------------------------------------------------------------------------
class TestPermutationNull:
    @staticmethod
    def _cox_c(seqs: list[dict], perm_seed: int | None) -> float:
        rows = []
        for s in seqs:
            if not s["visits"]:
                continue
            feats = s["risk_trace"]["features"]
            if feats.shape[0] == 0:
                continue
            rows.append(dict(
                chronic=float(feats[-1, 0]),
                acute=float(feats[-1, 1]),
                comorbid=float(feats[-1, 2]),
                treat=float(feats[-1, 3]),
                age=float(feats[-1, 4]),
                T=s["outcome"]["event_time"] + 1,
                E=s["outcome"]["event_indicator"],
            ))
        df = pd.DataFrame(rows)
        if perm_seed is not None:
            rng = np.random.default_rng(perm_seed)
            df["E"] = rng.permutation(df["E"].values)
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(df, duration_col="T", event_col="E")
        risk = cph.predict_partial_hazard(df).values
        return concordance_index(df["T"].values, -risk, df["E"].values)

    def test_permutation_null_near_half(self, local_sequences):
        """Shuffling event indicators (keeping T and features) should give
        Cox C-index near 0.5. Geometric dropout makes censoring independent
        of features, removing the n_visits confound. Some residual
        correlation is acceptable — band [0.45, 0.58]."""
        nulls = [self._cox_c(local_sequences, perm_seed=k) for k in range(5)]
        mean_null = np.mean(nulls)
        assert 0.45 <= mean_null <= 0.58, (
            f"permutation null {mean_null:.3f} outside [0.45, 0.58]"
        )
