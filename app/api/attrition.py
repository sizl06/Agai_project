"""Attrition prediction endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services import attrition_service
from app.utils.logger import get_logger
from app.validation.employee_schema import AttritionPrediction, EmployeeFeatures

log = get_logger("api.attrition")

router = APIRouter(prefix="/predict", tags=["attrition"])


@router.post("/attrition", response_model=AttritionPrediction)
def predict_attrition(
    employee: EmployeeFeatures,
    version: str | None = Query(None, description="Model version, e.g. v1. Defaults to latest."),
) -> AttritionPrediction:
    """Score a single employee.

    The request body is validated by Pydantic before anything reaches the model, so
    an out-of-range value returns 422 rather than a confident-looking prediction
    built on nonsense.
    """
    try:
        result = attrition_service.predict_for_employee(employee, version=version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Prediction failed for employee %s", employee.EmployeeID)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return AttritionPrediction(**result)


@router.get("/attrition/{employee_id}", response_model=AttritionPrediction)
def predict_known_employee(employee_id: int, version: str | None = None) -> AttritionPrediction:
    """Score an employee already present in the processed dataset."""
    result = attrition_service.predict_for_known_employee(employee_id, version=version)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")
    return AttritionPrediction(**result)


@router.get("/model/info", tags=["model"])
def model_info(version: str | None = None) -> dict:
    """Which model is serving, and how it scored when it was trained."""
    try:
        return attrition_service.model_info(version=version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
