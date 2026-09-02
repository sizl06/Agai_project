"""Validation rules - the checks the build notes name explicitly."""
from __future__ import annotations

import pandas as pd
import pytest

from app.validation.employee_schema import validate_employee_frame
from app.validation.engagement_schema import (
    validate_employee_skills_frame,
    validate_engagement_frame,
)
from app.validation.schema_checks import ValidationReport, check_referential_integrity


class TestEmployeeValidation:
    def test_clean_data_passes(self, employees):
        assert validate_employee_frame(employees).ok

    def test_missing_required_column_is_caught(self, employees):
        broken = employees.drop(columns=["JobSatisfaction"])
        report = validate_employee_frame(broken)
        assert not report.ok
        assert any("JobSatisfaction" in e for e in report.errors)

    def test_age_below_minimum_is_caught(self, employees):
        broken = employees.head(20).copy()
        broken.loc[broken.index[0], "Age"] = 7
        report = validate_employee_frame(broken)
        assert not report.ok
        assert any("Age" in e for e in report.errors)

    def test_age_above_maximum_is_caught(self, employees):
        broken = employees.head(20).copy()
        broken.loc[broken.index[0], "Age"] = 150
        assert not validate_employee_frame(broken).ok

    def test_duplicate_employee_id_is_caught(self, employees):
        broken = employees.head(20).copy()
        broken.loc[broken.index[1], "EmployeeID"] = broken["EmployeeID"].iloc[0]
        report = validate_employee_frame(broken)
        assert not report.ok
        assert any("unique" in e.lower() for e in report.errors)

    def test_unexpected_attrition_value_is_caught(self, employees):
        broken = employees.head(20).copy()
        broken.loc[broken.index[0], "Attrition"] = "Maybe"
        report = validate_employee_frame(broken)
        assert not report.ok
        assert any("Attrition" in e for e in report.errors)

    def test_null_in_required_column_is_caught(self, employees):
        broken = employees.head(20).copy()
        broken.loc[broken.index[0], "Department"] = None
        assert not validate_employee_frame(broken).ok

    def test_target_optional_at_inference_time(self, employees):
        inference = employees.drop(columns=["Attrition", "AttritionFlag"])
        assert validate_employee_frame(inference, require_target=False).ok

    def test_report_raises_on_failure(self, employees):
        broken = employees.head(5).drop(columns=["Age"])
        with pytest.raises(ValueError, match="Validation failed"):
            validate_employee_frame(broken).raise_if_failed()


class TestEngagementValidation:
    def test_clean_data_passes(self, engagement):
        assert validate_engagement_frame(engagement).ok

    def test_engagement_score_of_250_is_rejected(self, engagement):
        """The exact failure the build notes call out."""
        broken = engagement.head(20).copy()
        broken.loc[broken.index[0], "EngagementScore"] = 250.0
        report = validate_engagement_frame(broken)
        assert not report.ok
        assert any("EngagementScore" in e for e in report.errors)

    def test_negative_engagement_score_is_rejected(self, engagement):
        broken = engagement.head(20).copy()
        broken.loc[broken.index[0], "EngagementScore"] = -5.0
        assert not validate_engagement_frame(broken).ok

    def test_boundary_values_are_accepted(self, engagement):
        edge = engagement.head(20).copy()
        edge.loc[edge.index[0], "EngagementScore"] = 0.0
        edge.loc[edge.index[1], "EngagementScore"] = 100.0
        assert validate_engagement_frame(edge).ok

    def test_out_of_range_performance_rating_is_caught(self, engagement):
        broken = engagement.head(20).copy()
        broken.loc[broken.index[0], "PerformanceRating"] = 9
        assert not validate_engagement_frame(broken).ok


class TestSkillsValidation:
    def test_clean_data_passes(self, employee_skills, employees):
        report = validate_employee_skills_frame(
            employee_skills, valid_employee_ids=set(employees["EmployeeID"]))
        assert report.ok, report.errors

    def test_orphan_employee_is_caught(self, employee_skills, employees):
        broken = employee_skills.head(10).copy()
        broken.loc[broken.index[0], "EmployeeID"] = 999_999
        report = validate_employee_skills_frame(
            broken, valid_employee_ids=set(employees["EmployeeID"]))
        assert not report.ok

    def test_duplicate_employee_skill_pair_is_caught(self, employee_skills):
        broken = pd.concat([employee_skills.head(5), employee_skills.head(1)])
        report = validate_employee_skills_frame(broken)
        assert not report.ok
        assert any("duplicate" in e.lower() for e in report.errors)

    def test_proficiency_out_of_range_is_caught(self, employee_skills):
        broken = employee_skills.head(10).copy()
        broken.loc[broken.index[0], "ProficiencyLevel"] = 99
        assert not validate_employee_skills_frame(broken).ok


class TestReferentialIntegrity:
    def test_engagement_references_real_employees(self, employees, engagement):
        report = ValidationReport(dataset="refs", n_rows=len(engagement))
        check_referential_integrity(
            engagement, "EmployeeID", set(employees["EmployeeID"]), report, "engagement")
        assert report.ok

    def test_orphans_are_reported(self, employees):
        child = pd.DataFrame({"EmployeeID": [1, 2, 999_999]})
        report = ValidationReport(dataset="refs", n_rows=3)
        check_referential_integrity(
            child, "EmployeeID", set(employees["EmployeeID"]), report, "test")
        assert not report.ok
