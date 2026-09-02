"""Drift detection and the retraining decision.

Drift metrics are the kind of code that fails silently: a broken PSI still returns a
plausible number. So both directions are tested - it must read ~0 on identical data
*and* fire on a known shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipelines.monitor_drift import (
    MIN_PREDICTIONS_FOR_DRIFT,
    categorical_psi,
    compare_categorical,
    compare_numeric,
    psi,
    psi_band,
)
from pipelines.monitor_performance import (
    DRIFT_PSI_THRESHOLD,
    F1_FLOOR,
    MAX_MODEL_AGE_DAYS,
    evaluate_against_outcomes,
    should_retrain,
)


class TestPSI:
    def test_identical_distributions_score_zero(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(50, 10, 5000)
        assert psi(sample, sample) == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution_is_flagged(self):
        """The build notes' example: mean age 35 in training, 47 in production."""
        rng = np.random.default_rng(0)
        reference = rng.normal(35, 8, 5000)
        current = rng.normal(47, 8, 5000)
        assert psi(reference, current) > DRIFT_PSI_THRESHOLD

    def test_psi_grows_with_the_size_of_the_shift(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(50, 10, 5000)
        small = psi(reference, rng.normal(52, 10, 5000))
        large = psi(reference, rng.normal(70, 10, 5000))
        assert large > small

    def test_bands(self):
        assert psi_band(0.05) == "stable"
        assert psi_band(0.15) == "moderate"
        assert psi_band(0.40) == "significant"
        assert psi_band(float("nan")) == "unknown"

    def test_empty_input_is_not_a_crash(self):
        assert np.isnan(psi(np.array([]), np.array([1.0, 2.0])))

    def test_constant_column_does_not_divide_by_zero(self):
        constant = np.ones(100)
        assert psi(constant, constant) == 0.0

    def test_no_infinity_when_a_bucket_is_empty(self):
        """The epsilon floor exists for this case."""
        reference = np.arange(0, 100, dtype=float)
        current = np.full(100, 5.0)
        assert np.isfinite(psi(reference, current))


class TestCategoricalPSI:
    def test_identical_shares_score_zero(self):
        series = pd.Series(["A"] * 60 + ["B"] * 40)
        assert categorical_psi(series, series) == pytest.approx(0.0, abs=1e-9)

    def test_collapsed_category_is_flagged(self):
        reference = pd.Series(["Yes"] * 50 + ["No"] * 50)
        current = pd.Series(["Yes"] * 100)
        assert categorical_psi(reference, current) > DRIFT_PSI_THRESHOLD


class TestDriftReport:
    def test_self_comparison_reports_no_drift(self, employees):
        numeric = compare_numeric(employees, employees)
        categorical = compare_categorical(employees, employees)
        assert not numeric["drifted"].any()
        assert not categorical["drifted"].any()
        assert numeric["psi"].max() == pytest.approx(0.0, abs=1e-9)

    def test_shifted_data_is_detected(self, employees):
        current = employees.copy()
        current["Age"] = current["Age"] + 12

        result = compare_numeric(employees, current)
        age = result[result["feature"] == "Age"].iloc[0]

        assert age["drifted"]
        assert age["psi"] > DRIFT_PSI_THRESHOLD
        assert age["current_mean"] == pytest.approx(age["reference_mean"] + 12, abs=0.01)

    def test_drift_is_detected_in_either_direction(self, employees):
        """PSI is not symmetric, but a large shift must be caught whichever way round."""
        current = employees.copy()
        current["Age"] = current["Age"] + 12

        forward = compare_numeric(employees, current)
        backward = compare_numeric(current, employees)

        assert forward[forward["feature"] == "Age"]["drifted"].iloc[0]
        assert backward[backward["feature"] == "Age"]["drifted"].iloc[0]

    def test_unchanged_columns_stay_stable_when_one_shifts(self, employees):
        current = employees.copy()
        current["Age"] = current["Age"] + 12
        result = compare_numeric(employees, current)
        assert not result[result["feature"] == "MonthlyIncome"]["drifted"].iloc[0]

    def test_new_category_is_flagged(self, employees):
        current = employees.head(50).copy()
        current.loc[current.index[0], "Department"] = "Marketing"
        result = compare_categorical(employees, current)
        row = result[result["feature"] == "Department"].iloc[0]
        assert row["drifted"]
        assert "Marketing" in row["new_categories"]


class TestPerformanceEvaluation:
    def test_perfect_predictions_score_one(self):
        outcomes = np.array([0, 0, 1, 1])
        probabilities = np.array([0.01, 0.02, 0.99, 0.98])
        metrics = evaluate_against_outcomes(probabilities, outcomes, threshold=0.5)
        assert metrics["roc_auc"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 1.0

    def test_metrics_stay_in_range(self, employees):
        rng = np.random.default_rng(0)
        outcomes = employees["AttritionFlag"].astype(int).to_numpy()
        probabilities = rng.random(len(outcomes))
        metrics = evaluate_against_outcomes(probabilities, outcomes, threshold=0.5)
        for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "f2"]:
            assert 0.0 <= metrics[key] <= 1.0


class TestRetrainingDecision:
    def test_healthy_model_is_not_retrained(self):
        decision = should_retrain(live_metrics={"f1": 0.60}, max_psi=0.02)
        assert decision["retrain"] is False

    def test_significant_drift_triggers_retraining(self):
        decision = should_retrain(live_metrics={"f1": 0.60}, max_psi=0.90)
        assert decision["retrain"] is True
        assert any("drift" in r for r in decision["reasons"])

    def test_f1_below_the_floor_triggers_retraining(self):
        decision = should_retrain(live_metrics={"f1": F1_FLOOR - 0.05}, max_psi=0.01)
        assert decision["retrain"] is True
        assert any("floor" in r for r in decision["reasons"])

    def test_relative_f1_drop_triggers_retraining(self):
        """Catches decay from a high starting point the absolute floor would miss."""
        decision = should_retrain(live_metrics={"f1": 0.30}, max_psi=0.01)
        assert decision["retrain"] is True

    def test_decision_works_without_live_metrics(self):
        """Age alone can trigger a retrain; outcomes may not exist yet."""
        decision = should_retrain(live_metrics=None, max_psi=None)
        assert isinstance(decision["retrain"], bool)
        assert decision["model_age_days"] is not None

    def test_reasons_are_always_populated(self):
        decision = should_retrain(live_metrics={"f1": 0.60}, max_psi=0.01)
        assert decision["reasons"]

    def test_thresholds_are_sane(self):
        assert 0 < F1_FLOOR < 1
        assert DRIFT_PSI_THRESHOLD > 0
        assert MAX_MODEL_AGE_DAYS > 0
        assert MIN_PREDICTIONS_FOR_DRIFT > 1
