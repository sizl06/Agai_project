"""Prediction logging (step 21).

Kept deliberately separate from the application log. Application logs answer "what
did the service do"; this answers "what did the model say, and when". Mixing them
would mean grepping startup noise out of a drift analysis every time.

One CSV per day under data/predictions/, appended to. That keeps files small enough
to read, makes retention a matter of deleting old files, and means a day's
predictions can be loaded without parsing everything ever recorded.

The distribution of these probabilities over time is the earliest available warning
of model drift - it moves before ground-truth attrition outcomes are known. See
`pipelines/monitor_drift.py`.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.utils.config import PREDICTIONS_DIR
from app.utils.logger import get_logger

log = get_logger("prediction_log")

FIELDNAMES = [
    "timestamp",
    "employee_id",
    "model_version",
    "attrition_probability",
    "risk_level",
    "source",
]


def _today_file() -> Path:
    return PREDICTIONS_DIR / f"predictions_{datetime.now(timezone.utc):%Y-%m-%d}.csv"


def record(employee_id: int, probability: float, risk_level: str,
           model_version: str, source: str = "api") -> None:
    """Append one prediction. Never raises - logging must not break serving."""
    path = _today_file()
    try:
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            if is_new:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "employee_id": employee_id,
                "model_version": model_version,
                "attrition_probability": round(float(probability), 4),
                "risk_level": risk_level,
                "source": source,
            })
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not write prediction log: %s: %s", type(exc).__name__, exc)


def record_batch(predictions: pd.DataFrame, source: str = "batch") -> int:
    """Log a scored population in one pass."""
    rows = [
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "employee_id": int(row["EmployeeID"]),
            "model_version": row.get("ModelVersion", "unknown"),
            "attrition_probability": round(float(row["AttritionProbability"]), 4),
            "risk_level": row["RiskLevel"],
            "source": source,
        }
        for _, row in predictions.iterrows()
    ]
    if not rows:
        return 0

    path = _today_file()
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)

    log.info("Logged %d predictions to %s", len(rows), path.name)
    return len(rows)


def load_all() -> pd.DataFrame:
    """Every logged prediction, for drift analysis."""
    files = sorted(PREDICTIONS_DIR.glob("predictions_*.csv"))
    if not files:
        return pd.DataFrame(columns=FIELDNAMES)
    frames = [pd.read_csv(f) for f in files]
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], format="mixed", utc=True)
    return out


def summary() -> dict:
    df = load_all()
    if df.empty:
        return {"total_predictions": 0, "days_logged": 0}
    return {
        "total_predictions": int(len(df)),
        "days_logged": int(df["timestamp"].dt.date.nunique()),
        "distinct_employees": int(df["employee_id"].nunique()),
        "model_versions": sorted(df["model_version"].astype(str).unique().tolist()),
        "mean_probability": round(float(df["attrition_probability"].mean()), 4),
        "high_risk_share": round(float((df["risk_level"] == "HIGH").mean()), 4),
        "first_logged": str(df["timestamp"].min()),
        "last_logged": str(df["timestamp"].max()),
    }
