"""Attrition prediction, as the API consumes it.

Sits between the HTTP layer and the model: validate, predict, log, return. Keeping
this separate from `app/api/` means the same logic is callable from a notebook or a
batch job without going through HTTP.
"""
from __future__ import annotations

import pandas as pd

from app.ml import predictor
from app.ml.model_loader import active_version, decision_threshold, load_metadata
from app.services import prediction_log
from app.utils.config import ID_COL, PROCESSED_FILES
from app.utils.logger import get_logger
from app.validation.employee_schema import EmployeeFeatures

log = get_logger("attrition_service")


def predict_for_employee(features: EmployeeFeatures, version: str | None = None) -> dict:
    """Score one validated employee record and log the result."""
    log.info("Prediction request received for employee %s", features.EmployeeID)

    result = predictor.predict_one(features.to_frame(), version=version)

    prediction_log.record(
        employee_id=result["employee_id"],
        probability=result["attrition_probability"],
        risk_level=result["risk_level"],
        model_version=result["model_version"],
        source="api",
    )
    log.info(
        "Prediction completed for employee %s: %.4f (%s) using model %s",
        result["employee_id"], result["attrition_probability"],
        result["risk_level"], result["model_version"],
    )
    return result


def predict_for_known_employee(employee_id: int, version: str | None = None) -> dict | None:
    """Score an employee already in the processed dataset."""
    employees = pd.read_csv(PROCESSED_FILES["attrition"])
    record = employees[employees[ID_COL] == employee_id]
    if record.empty:
        return None

    result = predictor.predict_one(record, version=version)
    prediction_log.record(
        employee_id=result["employee_id"],
        probability=result["attrition_probability"],
        risk_level=result["risk_level"],
        model_version=result["model_version"],
        source="api-known",
    )
    return result


def score_population(version: str | None = None, log_predictions: bool = True) -> pd.DataFrame:
    employees = pd.read_csv(PROCESSED_FILES["attrition"])
    predictions = predictor.predict_batch(employees, version=version)
    if log_predictions:
        prediction_log.record_batch(predictions)
    return predictions


def model_info(version: str | None = None) -> dict:
    version = version or active_version()
    metadata = load_metadata(version)
    return {
        "active_version": version,
        "algorithm": metadata.get("algorithm"),
        "trained": metadata.get("training_date"),
        "decision_threshold": decision_threshold(version),
        "roc_auc": metadata.get("roc_auc"),
        "pr_auc": metadata.get("pr_auc"),
        "recall": metadata.get("recall"),
        "precision": metadata.get("precision"),
        "f1": metadata.get("f1"),
    }
