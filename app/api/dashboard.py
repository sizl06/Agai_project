"""Dashboard endpoints - everything the Streamlit front end renders."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services import dashboard_service

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _guard(fn, *args, **kwargs):
    """Turn a missing intelligence table into a clear 503 rather than a stack trace."""
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/summary")
def summary() -> dict:
    """KPI cards: headcount, high-risk count, average engagement."""
    return _guard(dashboard_service.summary)


@router.get("/attrition-by-department")
def attrition_by_department() -> list[dict]:
    return _guard(dashboard_service.attrition_by_department)


@router.get("/attrition-by-role")
def attrition_by_role() -> list[dict]:
    return _guard(dashboard_service.attrition_by_role)


@router.get("/risk-distribution")
def risk_distribution() -> list[dict]:
    return _guard(dashboard_service.risk_distribution)


@router.get("/skill-gaps")
def skill_gaps(limit: int = Query(15, ge=1, le=100)) -> list[dict]:
    """Organisation-wide skill shortages, most widespread first."""
    return _guard(dashboard_service.skill_gaps, limit=limit)


@router.get("/recommendations")
def recommendations(
    limit: int = Query(25, ge=1, le=500),
    risk_level: str | None = Query(None, pattern="^(HIGH|MEDIUM|LOW)$"),
) -> list[dict]:
    return _guard(dashboard_service.recommendations, limit=limit, risk_level=risk_level)


@router.get("/departments")
def departments() -> list[str]:
    return _guard(dashboard_service.department_list)
