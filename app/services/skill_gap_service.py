"""Skill gap engine (steps 13 and 14).

The core operation is set subtraction:

    required = {"Python", "SQL", "MLOps", "Docker", "AWS"}   # what the role needs
    has      = {"Python", "SQL", "AWS"}                       # what the employee has
    gap      = required - has                                 # {"MLOps", "Docker"}

Everything else is bookkeeping: weighting each gap by how important the skill is to
the role, rolling gaps up across the organisation, and assigning a severity band.

Two failure modes this module is written to avoid:

  * **Row explosion.** Employees join to requirements many-to-many through JobRole.
    Requirements are collapsed to one set per role *before* comparing, so the output
    stays at one row per (employee, missing skill).

  * **Silently dropping employees.** Someone with no skills on file must come out as
    missing *every* required skill, not as having no gap. A plain inner join would
    drop them and quietly understate the shortage.
"""
from __future__ import annotations

import pandas as pd

from app.utils.config import ID_COL, PROCESSED_FILES, gap_severity, gap_severity_relative


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    employees = pd.read_csv(PROCESSED_FILES["attrition"])
    requirements = pd.read_csv(PROCESSED_FILES["role_requirements"])
    skills = pd.read_csv(PROCESSED_FILES["employee_skills"])
    return employees, requirements, skills


def required_skills_by_role(requirements: pd.DataFrame) -> dict[str, dict[str, float]]:
    """{JobRole: {SkillName: Importance}} - one entry per role, no duplication."""
    return {
        role: dict(zip(group["SkillName"], group["Importance"]))
        for role, group in requirements.groupby("JobRole")
    }


def skills_by_employee(skills: pd.DataFrame) -> dict[int, set[str]]:
    return {int(eid): set(group["SkillName"]) for eid, group in skills.groupby(ID_COL)}


def compute_employee_gaps(
    employees: pd.DataFrame | None = None,
    requirements: pd.DataFrame | None = None,
    skills: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (employee, missing skill), with the importance of that skill."""
    if employees is None or requirements is None or skills is None:
        employees, requirements, skills = load_inputs()

    role_requirements = required_skills_by_role(requirements)
    employee_skills = skills_by_employee(skills)

    records = []
    for _, employee in employees.iterrows():
        employee_id = int(employee[ID_COL])
        role = employee["JobRole"]
        required = role_requirements.get(role, {})

        # .get() default matters: an employee with nothing on file has an empty set,
        # so every required skill correctly falls into the gap.
        held = employee_skills.get(employee_id, set())
        missing = set(required) - held

        for skill in missing:
            records.append({
                ID_COL: employee_id,
                "Department": employee["Department"],
                "JobRole": role,
                "MissingSkill": skill,
                "Importance": round(float(required[skill]), 2),
            })

    gaps = pd.DataFrame(records, columns=[ID_COL, "Department", "JobRole",
                                          "MissingSkill", "Importance"])
    return gaps.sort_values([ID_COL, "Importance"], ascending=[True, False]).reset_index(drop=True)


def employee_gap_summary(gaps: pd.DataFrame, employees: pd.DataFrame) -> pd.DataFrame:
    """One row per employee: how many skills are missing and which matter most.

    Built from a left join on the full employee list so that an employee with a
    complete skill set appears with a gap count of zero rather than vanishing.
    """
    aggregated = (
        gaps.groupby(ID_COL)
        .agg(
            SkillGapCount=("MissingSkill", "count"),
            SkillGapScore=("Importance", "sum"),
            SkillGaps=("MissingSkill", lambda s: ", ".join(list(s)[:5])),
            TopMissingSkill=("MissingSkill", "first"),
        )
        .reset_index()
    )

    out = employees[[ID_COL, "Department", "JobRole"]].merge(aggregated, on=ID_COL, how="left")
    out["SkillGapCount"] = out["SkillGapCount"].fillna(0).astype(int)
    out["SkillGapScore"] = out["SkillGapScore"].fillna(0.0).round(2)
    out["SkillGaps"] = out["SkillGaps"].fillna("")
    out["TopMissingSkill"] = out["TopMissingSkill"].fillna("")
    return out


def organization_skill_gaps(gaps: pd.DataFrame, total_employees: int) -> pd.DataFrame:
    """Roll gaps up company-wide and assign a severity band.

    Severity thresholds are absolute headcounts (100+ HIGH, 50+ MEDIUM) exactly as
    the build notes specify. `pct_of_workforce` is reported alongside because an
    absolute cut means something different at 1,470 employees than at 10,000.
    """
    if gaps.empty:
        return pd.DataFrame(columns=["MissingSkill", "EmployeesMissing", "pct_of_workforce",
                                     "AvgImportance", "Severity"])

    out = (
        gaps.groupby("MissingSkill")
        .agg(
            EmployeesMissing=(ID_COL, "nunique"),
            AvgImportance=("Importance", "mean"),
            DepartmentsAffected=("Department", "nunique"),
            RolesAffected=("JobRole", "nunique"),
        )
        .reset_index()
        .sort_values("EmployeesMissing", ascending=False)
    )
    out["pct_of_workforce"] = (out["EmployeesMissing"] / total_employees * 100).round(1)
    out["AvgImportance"] = out["AvgImportance"].round(2)
    out["Severity"] = out["EmployeesMissing"].map(gap_severity)
    # Both bands are reported. `Severity` is the spec; `SeverityRelative` is the
    # size-independent alternative that actually discriminates at this scale.
    out["SeverityRelative"] = out["EmployeesMissing"].map(
        lambda n: gap_severity_relative(int(n), total_employees)
    )
    return out.reset_index(drop=True)


def gaps_by_department(gaps: pd.DataFrame) -> pd.DataFrame:
    if gaps.empty:
        return pd.DataFrame(columns=["Department", "TotalGaps", "DistinctSkills"])
    return (
        gaps.groupby("Department")
        .agg(TotalGaps=("MissingSkill", "count"),
             DistinctSkills=("MissingSkill", "nunique"),
             EmployeesAffected=(ID_COL, "nunique"))
        .reset_index()
        .sort_values("TotalGaps", ascending=False)
    )


def build_and_save() -> tuple[pd.DataFrame, pd.DataFrame]:
    employees, requirements, skills = load_inputs()
    gaps = compute_employee_gaps(employees, requirements, skills)
    org = organization_skill_gaps(gaps, total_employees=len(employees))

    gaps.to_csv(PROCESSED_FILES["skill_gaps"], index=False)
    org.to_csv(PROCESSED_FILES["org_skill_gaps"], index=False)
    return gaps, org
