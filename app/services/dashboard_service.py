"""Dashboard aggregations.

Reads the Employee Intelligence table rather than recomputing anything. That table
is built by `pipelines/build_intelligence.py`; the dashboard is a view onto it, so
the API and the Streamlit front end can never disagree about a number.

The table is cached in memory and reloaded when the file changes on disk, so a
pipeline rerun is picked up without restarting the service.
"""
from __future__ import annotations

import pandas as pd

from app.utils.config import PROCESSED_FILES
from app.utils.logger import get_logger

log = get_logger("dashboard")

_cache: dict[str, object] = {"mtime": None, "frame": None}


def load_intelligence(force: bool = False) -> pd.DataFrame:
    path = PROCESSED_FILES["intelligence"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run: python -m pipelines.build_intelligence"
        )

    mtime = path.stat().st_mtime
    if force or _cache["frame"] is None or _cache["mtime"] != mtime:
        _cache["frame"] = pd.read_csv(path)
        _cache["mtime"] = mtime
        log.info("Intelligence table loaded (%d rows)", len(_cache["frame"]))
    return _cache["frame"]  # type: ignore[return-value]


def summary() -> dict:
    """The KPI cards at the top of the dashboard."""
    df = load_intelligence()
    return {
        "total_employees": int(len(df)),
        "high_risk_employees": int((df["RiskLevel"] == "HIGH").sum()),
        "medium_risk_employees": int((df["RiskLevel"] == "MEDIUM").sum()),
        "low_risk_employees": int((df["RiskLevel"] == "LOW").sum()),
        "average_engagement": round(float(df["EngagementScore"].mean()), 1),
        "average_attrition_probability": round(float(df["AttritionProbability"].mean()), 4),
        "employees_with_skill_gaps": int((df["SkillGapCount"] > 0).sum()),
        "average_skill_gaps": round(float(df["SkillGapCount"].mean()), 1),
        "departments": int(df["Department"].nunique()),
        "job_roles": int(df["JobRole"].nunique()),
    }


def attrition_by_department() -> list[dict]:
    df = load_intelligence()
    out = (
        df.groupby("Department")
        .agg(
            headcount=("EmployeeID", "count"),
            high_risk=("RiskLevel", lambda s: int((s == "HIGH").sum())),
            avg_probability=("AttritionProbability", "mean"),
            avg_engagement=("EngagementScore", "mean"),
        )
        .reset_index()
    )
    out["high_risk_pct"] = (out["high_risk"] / out["headcount"] * 100).round(1)
    out["avg_probability"] = out["avg_probability"].round(4)
    out["avg_engagement"] = out["avg_engagement"].round(1)
    return out.sort_values("high_risk_pct", ascending=False).to_dict("records")


def attrition_by_role() -> list[dict]:
    df = load_intelligence()
    out = (
        df.groupby("JobRole")
        .agg(
            headcount=("EmployeeID", "count"),
            high_risk=("RiskLevel", lambda s: int((s == "HIGH").sum())),
            avg_probability=("AttritionProbability", "mean"),
        )
        .reset_index()
    )
    out["high_risk_pct"] = (out["high_risk"] / out["headcount"] * 100).round(1)
    out["avg_probability"] = out["avg_probability"].round(4)
    return out.sort_values("high_risk_pct", ascending=False).to_dict("records")


def risk_distribution() -> list[dict]:
    df = load_intelligence()
    counts = df["RiskLevel"].value_counts()
    return [
        {"risk_level": level,
         "employees": int(counts.get(level, 0)),
         "percent": round(float(counts.get(level, 0)) / len(df) * 100, 1)}
        for level in ["HIGH", "MEDIUM", "LOW"]
    ]


def skill_gaps(limit: int = 15) -> list[dict]:
    org = pd.read_csv(PROCESSED_FILES["org_skill_gaps"])
    return org.head(limit).to_dict("records")


def recommendations(limit: int = 25, risk_level: str | None = None) -> list[dict]:
    df = load_intelligence()
    filtered = df[df["Recommendation"] != "No action required"]
    if risk_level:
        filtered = filtered[filtered["RiskLevel"] == risk_level.upper()]

    columns = ["EmployeeID", "Department", "JobRole", "RiskLevel", "AttritionProbability",
               "TopMissingSkill", "Recommendation", "CourseTitle", "Provider", "DurationHours"]
    return (filtered.sort_values("AttritionProbability", ascending=False)
            .head(limit)[columns].to_dict("records"))


def employee_record(employee_id: int) -> dict | None:
    """The full intelligence record for one person - the drill-down view."""
    df = load_intelligence()
    match = df[df["EmployeeID"] == employee_id]
    if match.empty:
        return None

    record = match.iloc[0].to_dict()
    gaps = pd.read_csv(PROCESSED_FILES["skill_gaps"])
    record["skill_gap_detail"] = (
        gaps[gaps["EmployeeID"] == employee_id]
        .sort_values("Importance", ascending=False)[["MissingSkill", "Importance"]]
        .to_dict("records")
    )

    recs = pd.read_csv(PROCESSED_FILES["recommendations"])
    record["recommendations"] = (
        recs[recs["EmployeeID"] == employee_id]
        .sort_values("Rank")[["MissingSkill", "Recommendation", "CourseTitle",
                              "Provider", "DurationHours", "Priority"]]
        .to_dict("records")
    )
    return record


def department_list() -> list[str]:
    return sorted(load_intelligence()["Department"].unique().tolist())
