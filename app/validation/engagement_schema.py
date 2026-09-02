"""Engagement-table contracts.

The rule that earns its keep here is the 0-100 range on EngagementScore: an HR
export with a score of 250 is the exact failure the build notes call out, and it
would otherwise flow silently into department averages.
"""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.utils.config import ID_COL, VALIDATION_RULES
from app.validation.schema_checks import (
    ValidationReport,
    check_no_duplicate_rows,
    check_no_nulls,
    check_numeric,
    check_ranges,
    check_required_columns,
    check_unique,
)

REQUIRED_COLUMNS = [ID_COL, "Department", "EngagementScore", "PerformanceRating"]

SCORE_COLUMNS = [
    "EngagementScore",
    "JobSatisfactionScore",
    "JobInvolvementScore",
    "EnvironmentSatisfactionScore",
    "WorkLifeBalanceScore",
    "RelationshipSatisfactionScore",
]


def validate_engagement_frame(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport(dataset="hr_performance_engagement", n_rows=len(df))

    check_required_columns(df, REQUIRED_COLUMNS, report)
    check_numeric(df, [c for c in SCORE_COLUMNS if c in df.columns], report)

    # Every 0-100 score column gets the EngagementScore range rule.
    rules = {c: VALIDATION_RULES["EngagementScore"] for c in SCORE_COLUMNS if c in df.columns}
    check_ranges(df, rules, report)

    check_unique(df, ID_COL, report)
    check_no_nulls(df, [ID_COL, "EngagementScore"], report)
    check_no_duplicate_rows(df, report)

    if "PerformanceRating" in df.columns:
        report.checks_run += 1
        bad = df[~df["PerformanceRating"].between(1, 4)]
        if len(bad):
            report.add_error(f"PerformanceRating outside 1-4 for {len(bad)} row(s)")

    return report


class EngagementRecord(BaseModel):
    """One engagement row, for API ingestion."""

    model_config = ConfigDict(extra="forbid")

    EmployeeID: int = Field(..., ge=1)
    Department: str
    EngagementScore: float = Field(..., ge=0.0, le=100.0)
    PerformanceRating: int = Field(..., ge=1, le=4)


def validate_employee_skills_frame(df: pd.DataFrame, valid_employee_ids: set[int] | None = None) -> ValidationReport:
    """Long-format skills inventory: one row per employee/skill pair."""
    report = ValidationReport(dataset="employee_current_skills", n_rows=len(df))

    check_required_columns(df, [ID_COL, "SkillName", "ProficiencyLevel"], report)
    check_no_nulls(df, [ID_COL, "SkillName"], report)
    check_ranges(df, {"ProficiencyLevel": VALIDATION_RULES["ProficiencyLevel"]}, report)

    # The grain is one row per (employee, skill) - repeats mean a broken export.
    report.checks_run += 1
    if {ID_COL, "SkillName"}.issubset(df.columns):
        dupes = int(df.duplicated(subset=[ID_COL, "SkillName"]).sum())
        if dupes:
            report.add_error(f"{dupes} duplicate (EmployeeID, SkillName) pair(s)")

    if valid_employee_ids is not None and ID_COL in df.columns:
        report.checks_run += 1
        orphans = set(df[ID_COL].dropna().unique()) - valid_employee_ids
        if orphans:
            report.add_error(f"{len(orphans)} skill rows reference unknown employees")

    return report
