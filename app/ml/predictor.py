"""Turn employee records into attrition predictions.

The saved pipeline handles engineering and preprocessing itself, so this module is
thin on purpose: validate, predict, band the probability, attach an explanation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.explainer import top_factors
from app.ml.model_loader import (
    active_version,
    decision_threshold,
    load_background,
    load_pipeline,
)
from app.utils.config import ID_COL, risk_level
from app.utils.logger import get_logger

log = get_logger("predictor")

# Columns the pipeline never uses; harmless if present, dropped for clarity.
NON_FEATURE_COLUMNS = ["Attrition", "AttritionFlag"]


def _clean_input(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])


def predict_probabilities(df: pd.DataFrame, version: str | None = None) -> np.ndarray:
    pipeline = load_pipeline(version)
    return pipeline.predict_proba(_clean_input(df))[:, 1]


def predict_one(record: pd.DataFrame, version: str | None = None,
                explain: bool = True, n_factors: int = 3) -> dict:
    """Predict for a single employee, with the reasons behind it."""
    version = version or active_version()
    threshold = decision_threshold(version)

    cleaned = _clean_input(record)
    probability = float(load_pipeline(version).predict_proba(cleaned)[0, 1])

    factors: list[dict] = []
    if explain:
        try:
            factors = top_factors(load_pipeline(version), cleaned,
                                  background=_clean_input(load_background()), n=n_factors)
        except Exception as exc:  # noqa: BLE001
            # An explanation failure must never cost the caller their prediction.
            log.warning("Explanation unavailable: %s: %s", type(exc).__name__, exc)

    employee_id = int(record[ID_COL].iloc[0]) if ID_COL in record.columns else -1
    return {
        "employee_id": employee_id,
        "attrition_probability": round(probability, 4),
        "risk_level": risk_level(probability, threshold),
        "model_version": version,
        "top_factors": factors,
    }


def predict_batch(df: pd.DataFrame, version: str | None = None) -> pd.DataFrame:
    """Score a whole population. No SHAP - explanations are per-request."""
    version = version or active_version()
    threshold = decision_threshold(version)

    probabilities = predict_probabilities(df, version)
    out = pd.DataFrame({
        ID_COL: df[ID_COL].astype(int).values,
        "AttritionProbability": probabilities.round(4),
        "RiskLevel": [risk_level(p, threshold) for p in probabilities],
    })
    out["ModelVersion"] = version
    log.info("Scored %d employees with model %s (threshold %.2f)", len(out), version, threshold)
    return out
