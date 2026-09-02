"""Skill normalisation, recommendations, engagement, and prediction logging."""
from __future__ import annotations

import pandas as pd
import pytest

from app.services import engagement_service, recommendation_service
from app.services.skill_normalizer import (
    normalization_audit,
    normalize_series,
    normalize_skill,
)


class TestSkillNormalizer:
    @pytest.mark.parametrize("raw", [
        "AWS", "Amazon Web Services", "AWS Cloud", "aws",
        "Amazon Web Services AWS software", "  AWS  ",
    ])
    def test_aws_variants_collapse_to_one_name(self, raw):
        """The exact example the build notes call out."""
        assert normalize_skill(raw) == "Amazon Web Services (AWS)"

    @pytest.mark.parametrize("raw,expected", [
        ("SAP software", "SAP"),
        ("Salesforce software", "Salesforce"),
        ("Microsoft Office software", "Microsoft Office"),
        ("The MathWorks MATLAB", "MATLAB"),
        ("ESRI ArcGIS software", "ESRI ArcGIS"),
        ("excel", "Microsoft Excel"),
        ("MS Excel", "Microsoft Excel"),
        ("Applicant tracking software", "Applicant Tracking System (ATS)"),
    ])
    def test_known_aliases(self, raw, expected):
        assert normalize_skill(raw) == expected

    def test_upstream_typo_is_fixed(self):
        assert normalize_skill("Micosoft SQL Server").startswith("Microsoft")

    def test_case_and_punctuation_are_ignored(self):
        assert normalize_skill("a.w.s.") == normalize_skill("AWS")

    def test_unknown_skill_is_preserved(self):
        assert normalize_skill("Kubernetes") == "Kubernetes"

    def test_capitalisation_of_unknown_names_is_not_destroyed(self):
        assert normalize_skill("PyTorch Lightning") == "PyTorch Lightning"

    def test_empty_and_none_are_safe(self):
        assert normalize_skill("") == ""
        assert normalize_skill(None) == ""

    def test_normalisation_is_idempotent(self):
        once = normalize_skill("Amazon Web Services AWS software")
        assert normalize_skill(once) == once

    def test_series_helper(self):
        assert normalize_series(["AWS", "SAP software"]) == [
            "Amazon Web Services (AWS)", "SAP"]

    def test_audit_reports_merges(self):
        audit = normalization_audit(["AWS", "AWS Cloud", "Amazon Web Services", "Python"])
        assert "Amazon Web Services (AWS)" in audit
        assert len(audit["Amazon Web Services (AWS)"]) == 3
        assert "Python" not in audit  # only one spelling, so not a merge


class TestRecommendations:
    @pytest.fixture(scope="class")
    def catalogue(self):
        return recommendation_service.load_catalogue()

    def test_catalogue_loads_with_tags(self, catalogue):
        assert len(catalogue) > 0
        assert catalogue["SkillList"].apply(len).min() >= 1

    def test_tagged_skill_matches_by_tag(self, catalogue):
        match = recommendation_service.recommend_for_skill("Python", catalogue)
        assert match["MatchType"] == "tag"
        assert match["MatchScore"] == 1.0

    def test_every_required_skill_has_a_course(self, catalogue, requirements):
        index = recommendation_service.build_skill_index(catalogue)
        uncovered = set(requirements["SkillName"]) - set(index)
        assert not uncovered, f"skills with no course: {sorted(uncovered)}"

    def test_unknown_skill_falls_back_or_returns_none(self, catalogue):
        """Must not raise, whichever branch it takes."""
        result = recommendation_service.recommend_for_skill("Quantum Cryptography", catalogue)
        assert result is None or result["MatchType"] == "semantic"

    def test_priority_rises_with_attrition_risk(self):
        gaps = pd.DataFrame([
            {"EmployeeID": 1, "Department": "IT", "JobRole": "Analyst",
             "MissingSkill": "Python", "Importance": 4.0},
            {"EmployeeID": 2, "Department": "IT", "JobRole": "Analyst",
             "MissingSkill": "Python", "Importance": 4.0},
        ])
        org = pd.DataFrame([{"MissingSkill": "Python", "Severity": "HIGH"}])

        out = recommendation_service.build_recommendations(
            gaps, org, risk_by_employee={1: "HIGH", 2: "LOW"})

        high = out[out["EmployeeID"] == 1]["Priority"].iloc[0]
        low = out[out["EmployeeID"] == 2]["Priority"].iloc[0]
        assert high > low

    def test_recommendations_are_capped_per_employee(self, requirements):
        from app.services.skill_gap_service import compute_employee_gaps

        employees = pd.DataFrame([{"EmployeeID": 1, "Department": "IT",
                                   "JobRole": "Research Scientist"}])
        gaps = compute_employee_gaps(employees, requirements, pd.DataFrame(
            columns=["EmployeeID", "SkillName"]))
        org = pd.DataFrame({"MissingSkill": gaps["MissingSkill"].unique(),
                            "Severity": "HIGH"})

        out = recommendation_service.build_recommendations(
            gaps, org, per_employee=3)
        assert len(out) <= 3
        assert out["Rank"].max() <= 3

    def test_empty_gaps_produce_empty_recommendations(self):
        empty = pd.DataFrame(columns=["EmployeeID", "Department", "JobRole",
                                      "MissingSkill", "Importance"])
        out = recommendation_service.build_recommendations(
            empty, pd.DataFrame(columns=["MissingSkill", "Severity"]))
        assert out.empty


class TestEngagement:
    def test_bands(self):
        assert engagement_service.band(10) == "Low"
        assert engagement_service.band(55) == "Moderate"
        assert engagement_service.band(85) == "High"
        assert engagement_service.band(100) == "High"

    def test_summary_within_scale(self, engagement):
        summary = engagement_service.summary(engagement)
        assert 0 <= summary["average_engagement"] <= 100
        assert summary["employees"] == len(engagement)

    def test_department_breakdown_covers_everyone(self, engagement):
        by_department = engagement_service.by_department(engagement)
        assert by_department["headcount"].sum() == len(engagement)

    def test_lowest_engaged_is_sorted(self, engagement):
        lowest = engagement_service.lowest_engaged(10, engagement)
        assert len(lowest) == 10
        assert lowest["EngagementScore"].is_monotonic_increasing

    def test_band_distribution_sums_to_total(self, engagement):
        bands = engagement_service.band_distribution(engagement)
        assert bands["employees"].sum() == len(engagement)


class TestPredictionLog:
    def test_record_and_read_back(self, tmp_path, monkeypatch):
        from app.services import prediction_log

        monkeypatch.setattr(prediction_log, "PREDICTIONS_DIR", tmp_path)
        prediction_log.record(101, 0.82, "HIGH", "v1", source="test")
        prediction_log.record(102, 0.11, "LOW", "v1", source="test")

        logged = prediction_log.load_all()
        assert len(logged) == 2
        assert set(logged["employee_id"]) == {101, 102}

    def test_batch_logging(self, tmp_path, monkeypatch):
        from app.services import prediction_log

        monkeypatch.setattr(prediction_log, "PREDICTIONS_DIR", tmp_path)
        predictions = pd.DataFrame([
            {"EmployeeID": 1, "AttritionProbability": 0.5, "RiskLevel": "MEDIUM",
             "ModelVersion": "v1"},
            {"EmployeeID": 2, "AttritionProbability": 0.9, "RiskLevel": "HIGH",
             "ModelVersion": "v1"},
        ])
        assert prediction_log.record_batch(predictions) == 2
        assert prediction_log.summary()["total_predictions"] == 2

    def test_logging_failure_does_not_raise(self, monkeypatch, tmp_path):
        """A broken log must never cost the caller their prediction."""
        from app.services import prediction_log

        monkeypatch.setattr(prediction_log, "PREDICTIONS_DIR", tmp_path / "nope" / "deeper")
        prediction_log.record(1, 0.5, "MEDIUM", "v1")  # must not raise

    def test_summary_on_empty_log(self, tmp_path, monkeypatch):
        from app.services import prediction_log

        monkeypatch.setattr(prediction_log, "PREDICTIONS_DIR", tmp_path)
        assert prediction_log.summary()["total_predictions"] == 0
