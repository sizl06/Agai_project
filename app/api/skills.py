"""Skill, employee-record and engagement endpoints."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.services import dashboard_service, engagement_service
from app.utils.config import PROCESSED_FILES

router = APIRouter(tags=["workforce"])


@router.get("/employees/{employee_id}")
def employee_record(employee_id: int) -> dict:
    """Full intelligence record for one person: risk, engagement, gaps, recommendations."""
    try:
        record = dashboard_service.employee_record(employee_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")
    return record


@router.get("/employees")
def list_employees(
    department: str | None = None,
    risk_level: str | None = Query(None, pattern="^(HIGH|MEDIUM|LOW)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    try:
        df = dashboard_service.load_intelligence()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if department:
        df = df[df["Department"] == department]
    if risk_level:
        df = df[df["RiskLevel"] == risk_level.upper()]

    columns = ["EmployeeID", "Department", "JobRole", "AttritionProbability", "RiskLevel",
               "EngagementScore", "SkillGapCount", "Recommendation"]
    page = df.sort_values("AttritionProbability", ascending=False).iloc[offset: offset + limit]
    return {"total": int(len(df)), "offset": offset, "limit": limit,
            "employees": page[columns].to_dict("records")}


@router.get("/skills/role/{job_role}")
def role_requirements(job_role: str) -> dict:
    """What a role requires, per O*NET."""
    requirements = pd.read_csv(PROCESSED_FILES["role_requirements"])
    match = requirements[requirements["JobRole"].str.lower() == job_role.lower()]
    if match.empty:
        available = sorted(requirements["JobRole"].unique().tolist())
        raise HTTPException(
            status_code=404, detail=f"Unknown job role '{job_role}'. Available: {available}")

    return {
        "job_role": match["JobRole"].iloc[0],
        "onet_soc_code": match["ONET_SOC_Code"].iloc[0],
        "required_skills": match.sort_values(["SkillType", "RankInRole"])[
            ["SkillName", "SkillType", "Importance"]].to_dict("records"),
    }


@router.get("/skills/employee/{employee_id}")
def employee_skills(employee_id: int) -> dict:
    """What one employee holds, and what they are missing."""
    skills = pd.read_csv(PROCESSED_FILES["employee_skills"])
    gaps = pd.read_csv(PROCESSED_FILES["skill_gaps"])

    held = skills[skills["EmployeeID"] == employee_id]
    missing = gaps[gaps["EmployeeID"] == employee_id]
    if held.empty and missing.empty:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")

    return {
        "employee_id": employee_id,
        "skills_held": held[["SkillName", "SkillType", "ProficiencyLevel"]].to_dict("records"),
        "skills_missing": missing.sort_values("Importance", ascending=False)[
            ["MissingSkill", "Importance"]].to_dict("records"),
    }


@router.get("/engagement/summary")
def engagement_summary() -> dict:
    return engagement_service.summary()


@router.get("/engagement/by-department")
def engagement_by_department() -> list[dict]:
    return engagement_service.by_department().to_dict("records")


@router.get("/engagement/lowest")
def lowest_engaged(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    return engagement_service.lowest_engaged(limit).to_dict("records")
