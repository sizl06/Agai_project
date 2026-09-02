"""Model behaviour: probabilities, risk banding, features, and explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.features import (
    ENGINEERED_COLUMNS,
    EXCLUDED_COLUMNS,
    check_leakage,
    engineer_features,
)
from app.utils.config import (
    DEFAULT_DECISION_THRESHOLD,
    RISK_BAND_MULTIPLIERS,
    risk_level,
)
from tests.conftest import requires_model


class TestRiskBanding:
    def test_bands_are_relative_to_the_threshold(self):
        threshold = 0.58
        assert risk_level(0.90, threshold) == "HIGH"
        assert risk_level(0.58, threshold) == "HIGH"
        assert risk_level(0.57, threshold) == "MEDIUM"
        assert risk_level(0.35, threshold) == "MEDIUM"  # 0.58 * 0.6 = 0.348
        assert risk_level(0.20, threshold) == "LOW"
        assert risk_level(0.00, threshold) == "LOW"

    def test_boundaries_are_inclusive_at_the_lower_edge(self):
        threshold = 0.5
        assert risk_level(threshold * RISK_BAND_MULTIPLIERS["HIGH"], threshold) == "HIGH"
        assert risk_level(threshold * RISK_BAND_MULTIPLIERS["MEDIUM"], threshold) == "MEDIUM"

    def test_bands_shift_with_a_different_threshold(self):
        """A probability can be HIGH under one model and LOW under another."""
        assert risk_level(0.30, 0.25) == "HIGH"
        assert risk_level(0.30, 0.80) == "LOW"

    def test_default_threshold_is_used_when_none_given(self):
        assert risk_level(DEFAULT_DECISION_THRESHOLD) == "HIGH"
        assert risk_level(0.01) == "LOW"


class TestFeatureEngineering:
    def test_all_engineered_columns_are_produced(self, employees):
        X = engineer_features(employees)
        for column in ENGINEERED_COLUMNS:
            assert column in X.columns

    def test_identifier_and_target_are_excluded(self, employees):
        X = engineer_features(employees)
        for column in EXCLUDED_COLUMNS:
            assert column not in X.columns, f"{column} must never be a feature"

    def test_input_frame_is_not_mutated(self, employees):
        before = employees.columns.tolist()
        engineer_features(employees)
        assert employees.columns.tolist() == before

    def test_zero_tenure_does_not_produce_infinity(self):
        """The +1 denominators exist for exactly this row."""
        row = pd.DataFrame([{
            "Age": 22, "MonthlyIncome": 3000, "YearsAtCompany": 0, "TotalWorkingYears": 0,
            "YearsSinceLastPromotion": 0, "YearsInCurrentRole": 0, "NumCompaniesWorked": 0,
            "JobLevel": 1, "JobSatisfaction": 3, "EnvironmentSatisfaction": 3,
            "RelationshipSatisfaction": 3, "JobInvolvement": 3, "YearsWithCurrManager": 0,
            "Department": "Sales", "JobRole": "Sales Executive", "OverTime": "No",
        }])
        X = engineer_features(row)
        assert not np.isinf(X[ENGINEERED_COLUMNS].to_numpy(dtype=float)).any()

    def test_income_per_year_uses_tenure_plus_one(self):
        row = pd.DataFrame([{
            "Age": 30, "MonthlyIncome": 6000, "YearsAtCompany": 5, "TotalWorkingYears": 10,
            "YearsSinceLastPromotion": 1, "YearsInCurrentRole": 3, "NumCompaniesWorked": 2,
            "JobLevel": 2, "JobSatisfaction": 4, "EnvironmentSatisfaction": 3,
            "RelationshipSatisfaction": 3, "JobInvolvement": 2, "YearsWithCurrManager": 3,
            "Department": "Sales", "JobRole": "Sales Executive", "OverTime": "No",
        }])
        X = engineer_features(row)
        assert X["IncomePerYearAtCompany"].iloc[0] == pytest.approx(6000 / 6)
        assert X["SatisfactionIndex"].iloc[0] == pytest.approx((4 + 3 + 3 + 2) / 4)

    def test_no_feature_leaks_the_target(self, employees):
        X = engineer_features(employees)
        leakage = check_leakage(X, employees["AttritionFlag"].astype(int))
        assert not leakage["suspicious"].any(), leakage[leakage["suspicious"]].to_dict("records")


@requires_model
class TestPrediction:
    def test_probability_is_in_range(self, employees):
        from app.ml.predictor import predict_probabilities

        probabilities = predict_probabilities(employees.head(50))
        assert len(probabilities) == 50
        assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()

    def test_batch_output_shape_and_columns(self, employees):
        from app.ml.predictor import predict_batch

        out = predict_batch(employees.head(30))
        assert len(out) == 30
        assert set(out.columns) == {"EmployeeID", "AttritionProbability",
                                    "RiskLevel", "ModelVersion"}
        assert out["RiskLevel"].isin(["HIGH", "MEDIUM", "LOW"]).all()

    def test_single_prediction_has_explanation(self, employees):
        from app.ml.predictor import predict_one

        result = predict_one(employees.head(1))
        assert 0.0 <= result["attrition_probability"] <= 1.0
        assert result["risk_level"] in {"HIGH", "MEDIUM", "LOW"}
        assert len(result["top_factors"]) == 3
        for factor in result["top_factors"]:
            assert {"feature", "label", "shap_value", "direction"} <= set(factor)

    def test_explanation_aggregates_dummies_to_one_entry(self, employees):
        """OverTime_Yes and OverTime_No must not both occupy a slot."""
        from app.ml.predictor import predict_one

        result = predict_one(employees.head(1))
        features = [f["feature"] for f in result["top_factors"]]
        assert len(features) == len(set(features))

    def test_prediction_is_deterministic(self, employees):
        from app.ml.predictor import predict_probabilities

        first = predict_probabilities(employees.head(10))
        second = predict_probabilities(employees.head(10))
        np.testing.assert_array_almost_equal(first, second)

    def test_risk_level_matches_the_probability(self, employees):
        from app.ml.model_loader import decision_threshold
        from app.ml.predictor import predict_batch

        out = predict_batch(employees.head(100))
        threshold = decision_threshold()
        for _, row in out.iterrows():
            assert row["RiskLevel"] == risk_level(row["AttritionProbability"], threshold)

    def test_pipeline_accepts_raw_records(self, employees):
        """The artifact must engineer features itself, not expect them pre-computed."""
        from app.ml.predictor import predict_probabilities

        raw = employees.drop(columns=["Attrition", "AttritionFlag"]).head(5)
        assert len(predict_probabilities(raw)) == 5
