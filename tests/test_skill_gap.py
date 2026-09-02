"""Skill gap engine: set subtraction, join safety, and severity banding."""
from __future__ import annotations

import pandas as pd

from app.services.skill_gap_service import (
    compute_employee_gaps,
    employee_gap_summary,
    organization_skill_gaps,
    required_skills_by_role,
    skills_by_employee,
)
from app.utils.config import gap_severity, gap_severity_relative


def _fixture_frames():
    """A tiny, fully hand-checkable dataset."""
    employees = pd.DataFrame([
        {"EmployeeID": 101, "Department": "IT", "JobRole": "ML Engineer"},
        {"EmployeeID": 102, "Department": "IT", "JobRole": "ML Engineer"},
        {"EmployeeID": 103, "Department": "IT", "JobRole": "Analyst"},
    ])
    requirements = pd.DataFrame([
        {"JobRole": "ML Engineer", "SkillName": "Python", "Importance": 5.0},
        {"JobRole": "ML Engineer", "SkillName": "SQL", "Importance": 4.0},
        {"JobRole": "ML Engineer", "SkillName": "MLOps", "Importance": 4.5},
        {"JobRole": "ML Engineer", "SkillName": "Docker", "Importance": 3.0},
        {"JobRole": "ML Engineer", "SkillName": "AWS", "Importance": 3.5},
        {"JobRole": "Analyst", "SkillName": "SQL", "Importance": 5.0},
        {"JobRole": "Analyst", "SkillName": "Excel", "Importance": 4.0},
    ])
    skills = pd.DataFrame([
        {"EmployeeID": 101, "SkillName": "Python", "ProficiencyLevel": 4},
        {"EmployeeID": 101, "SkillName": "SQL", "ProficiencyLevel": 3},
        {"EmployeeID": 101, "SkillName": "AWS", "ProficiencyLevel": 3},
        {"EmployeeID": 102, "SkillName": "Python", "ProficiencyLevel": 5},
        {"EmployeeID": 102, "SkillName": "SQL", "ProficiencyLevel": 4},
        {"EmployeeID": 102, "SkillName": "MLOps", "ProficiencyLevel": 3},
        {"EmployeeID": 102, "SkillName": "Docker", "ProficiencyLevel": 3},
        {"EmployeeID": 102, "SkillName": "AWS", "ProficiencyLevel": 4},
        # 103 deliberately has no skills recorded at all.
    ])
    return employees, requirements, skills


class TestSetSubtraction:
    def test_matches_the_worked_example_from_the_build_notes(self):
        """required - has == {'MLOps', 'Docker'} for employee 101."""
        employees, requirements, skills = _fixture_frames()
        gaps = compute_employee_gaps(employees, requirements, skills)

        gap_101 = set(gaps[gaps["EmployeeID"] == 101]["MissingSkill"])
        assert gap_101 == {"MLOps", "Docker"}

    def test_employee_with_every_skill_has_no_gap(self):
        employees, requirements, skills = _fixture_frames()
        gaps = compute_employee_gaps(employees, requirements, skills)
        assert gaps[gaps["EmployeeID"] == 102].empty

    def test_employee_with_no_skills_is_missing_everything(self):
        """The failure mode a plain inner join would hide."""
        employees, requirements, skills = _fixture_frames()
        gaps = compute_employee_gaps(employees, requirements, skills)

        gap_103 = set(gaps[gaps["EmployeeID"] == 103]["MissingSkill"])
        assert gap_103 == {"SQL", "Excel"}

    def test_importance_is_carried_through(self):
        employees, requirements, skills = _fixture_frames()
        gaps = compute_employee_gaps(employees, requirements, skills)
        mlops = gaps[(gaps["EmployeeID"] == 101) & (gaps["MissingSkill"] == "MLOps")]
        assert float(mlops["Importance"].iloc[0]) == 4.5


class TestJoinSafety:
    def test_summary_stays_one_row_per_employee(self):
        employees, requirements, skills = _fixture_frames()
        gaps = compute_employee_gaps(employees, requirements, skills)
        summary = employee_gap_summary(gaps, employees)

        assert len(summary) == len(employees)
        assert not summary["EmployeeID"].duplicated().any()

    def test_employee_with_no_gaps_survives_the_join(self):
        """102 has no gap rows, so a left join must still keep them, at zero."""
        employees, requirements, skills = _fixture_frames()
        gaps = compute_employee_gaps(employees, requirements, skills)
        summary = employee_gap_summary(gaps, employees)

        row = summary[summary["EmployeeID"] == 102].iloc[0]
        assert row["SkillGapCount"] == 0
        assert row["SkillGapScore"] == 0.0

    def test_real_data_does_not_explode(self, employees, requirements, employee_skills):
        gaps = compute_employee_gaps(employees, requirements, employee_skills)
        summary = employee_gap_summary(gaps, employees)
        assert len(summary) == len(employees)

    def test_helpers_deduplicate_correctly(self):
        _, requirements, skills = _fixture_frames()
        by_role = required_skills_by_role(requirements)
        by_employee = skills_by_employee(skills)

        assert set(by_role["ML Engineer"]) == {"Python", "SQL", "MLOps", "Docker", "AWS"}
        assert by_employee[101] == {"Python", "SQL", "AWS"}
        assert 103 not in by_employee


class TestSeverity:
    def test_absolute_bands_follow_the_build_notes(self):
        assert gap_severity(120) == "HIGH"
        assert gap_severity(100) == "HIGH"
        assert gap_severity(99) == "MEDIUM"
        assert gap_severity(50) == "MEDIUM"
        assert gap_severity(49) == "LOW"
        assert gap_severity(0) == "LOW"

    def test_relative_bands_scale_with_headcount(self):
        assert gap_severity_relative(500, 1000) == "HIGH"
        assert gap_severity_relative(250, 1000) == "MEDIUM"
        assert gap_severity_relative(100, 1000) == "LOW"

    def test_relative_bands_handle_zero_headcount(self):
        assert gap_severity_relative(10, 0) == "LOW"

    def test_rollup_counts_distinct_employees(self):
        employees, requirements, skills = _fixture_frames()
        gaps = compute_employee_gaps(employees, requirements, skills)
        org = organization_skill_gaps(gaps, total_employees=len(employees))

        sql = org[org["MissingSkill"] == "SQL"].iloc[0]
        assert sql["EmployeesMissing"] == 1  # only employee 103
        assert set(org.columns) >= {"Severity", "SeverityRelative", "pct_of_workforce"}

    def test_empty_gaps_produce_empty_rollup(self):
        empty = pd.DataFrame(columns=["EmployeeID", "Department", "JobRole",
                                      "MissingSkill", "Importance"])
        assert organization_skill_gaps(empty, total_employees=10).empty
