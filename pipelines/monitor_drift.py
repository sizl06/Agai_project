"""Data drift monitoring (step 25).

Compares the distribution of production data against what the model was trained on.
The example from the build notes: if the average employee age in training was 35 but
production is averaging 47, that is worth investigating before trusting predictions.

Plain pandas and scipy, as the notes prescribe - Evidently AI is the upgrade once
this proves useful, not the starting point.

Two measures, because they catch different things:

  * **PSI (Population Stability Index)** - the industry-standard bucketed comparison.
    Under 0.10 is stable, 0.10-0.25 is moderate drift, above 0.25 is significant.
    Interpretable and threshold-able, but insensitive to shifts within a bucket.

  * **KS statistic** with a p-value - detects distribution shape changes PSI can miss.
    On large samples it flags differences too small to matter, so it is read
    alongside PSI rather than on its own.

Prediction drift is tracked separately. It is the earliest available signal, because
it moves before anyone knows who actually left.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from app.services import prediction_log
from app.utils.config import PROCESSED_FILES, REPORTS_DIR
from app.utils.logger import get_logger

log = get_logger("drift")

# The columns the build notes name, plus the engineered ones most likely to move.
MONITORED_COLUMNS = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "JobSatisfaction",
    "TotalWorkingYears",
    "DistanceFromHome",
    "JobLevel",
    "WorkLifeBalance",
]

CATEGORICAL_COLUMNS = ["Department", "JobRole", "OverTime", "BusinessTravel", "MaritalStatus"]

PSI_BANDS = {"stable": 0.10, "moderate": 0.25}

# PSI on a handful of predictions is noise. Below this, report "not enough data".
MIN_PREDICTIONS_FOR_DRIFT = 100


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

    Buckets come from the *reference* quantiles, so the comparison asks "where do
    today's values fall in yesterday's distribution" rather than re-cutting both.
    """
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:  # near-constant column; PSI is not meaningful
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=edges)[0] / len(actual)

    # Floor at a small epsilon so an empty bucket does not produce infinity.
    epsilon = 1e-6
    expected_pct = np.clip(expected_pct, epsilon, None)
    actual_pct = np.clip(actual_pct, epsilon, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def psi_band(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value < PSI_BANDS["stable"]:
        return "stable"
    if value < PSI_BANDS["moderate"]:
        return "moderate"
    return "significant"


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """PSI over category shares."""
    categories = sorted(set(expected.dropna()) | set(actual.dropna()))
    if not categories:
        return float("nan")
    epsilon = 1e-6
    expected_pct = np.clip(
        expected.value_counts(normalize=True).reindex(categories).fillna(0).to_numpy(),
        epsilon, None)
    actual_pct = np.clip(
        actual.value_counts(normalize=True).reindex(categories).fillna(0).to_numpy(),
        epsilon, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def compare_numeric(reference: pd.DataFrame, current: pd.DataFrame,
                    columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or MONITORED_COLUMNS
    rows = []
    for column in columns:
        if column not in reference.columns or column not in current.columns:
            continue
        ref, cur = reference[column].dropna(), current[column].dropna()
        psi_value = psi(ref.to_numpy(), cur.to_numpy())
        ks_stat, p_value = stats.ks_2samp(ref, cur)
        rows.append({
            "feature": column,
            "type": "numeric",
            "reference_mean": round(float(ref.mean()), 3),
            "current_mean": round(float(cur.mean()), 3),
            "mean_shift_%": round(float((cur.mean() - ref.mean()) / abs(ref.mean()) * 100), 2)
            if ref.mean() else np.nan,
            "psi": round(psi_value, 4),
            "psi_band": psi_band(psi_value),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(p_value), 4),
            "drifted": psi_band(psi_value) != "stable",
        })
    return pd.DataFrame(rows)


def compare_categorical(reference: pd.DataFrame, current: pd.DataFrame,
                        columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or CATEGORICAL_COLUMNS
    rows = []
    for column in columns:
        if column not in reference.columns or column not in current.columns:
            continue
        psi_value = categorical_psi(reference[column], current[column])
        new_categories = sorted(set(current[column].dropna()) - set(reference[column].dropna()))
        rows.append({
            "feature": column,
            "type": "categorical",
            "psi": round(psi_value, 4),
            "psi_band": psi_band(psi_value),
            "new_categories": ", ".join(map(str, new_categories)) or "-",
            "drifted": psi_band(psi_value) != "stable" or bool(new_categories),
        })
    return pd.DataFrame(rows)


def prediction_drift(reference_probabilities: np.ndarray | None = None) -> dict:
    """Compare logged production predictions against the training-time distribution.

    This is the earliest warning available: prediction distribution moves before
    ground-truth outcomes are known.

    PSI is only computed once enough predictions have accumulated. On a handful of
    rows it is wildly unstable - a dozen requests that happen to be high-risk
    employees will report "significant drift" every time. An alerting system that
    cries wolf on its first day gets ignored, so below the minimum the report says
    plainly that there is not enough data yet.
    """
    logged = prediction_log.load_all()
    if logged.empty:
        return {"status": "no predictions logged"}

    current = logged["attrition_probability"].to_numpy()
    result = {
        "predictions_logged": int(len(logged)),
        "current_mean_probability": round(float(current.mean()), 4),
        "current_high_risk_share": round(float((logged["risk_level"] == "HIGH").mean()), 4),
    }

    if len(current) < MIN_PREDICTIONS_FOR_DRIFT:
        result["status"] = (
            f"insufficient data - {len(current)} predictions logged, "
            f"{MIN_PREDICTIONS_FOR_DRIFT} needed before PSI is meaningful"
        )
        return result

    if reference_probabilities is not None and len(reference_probabilities):
        value = psi(reference_probabilities, current)
        result.update({
            "reference_mean_probability": round(float(np.mean(reference_probabilities)), 4),
            "prediction_psi": round(value, 4),
            "prediction_psi_band": psi_band(value),
        })
    return result


def run(current: pd.DataFrame | None = None, save: bool = True) -> dict:
    """Full drift report.

    With no `current` frame this compares the training data against itself, which is
    a self-test rather than a finding: PSI must come out at ~0 across the board. That
    is worth running - a non-zero result here would mean the metric is broken.
    """
    reference = pd.read_csv(PROCESSED_FILES["attrition"])
    self_test = current is None
    current = reference.copy() if self_test else current

    numeric = compare_numeric(reference, current)
    categorical = compare_categorical(reference, current)
    combined = pd.concat([numeric, categorical], ignore_index=True)

    from app.ml.predictor import predict_probabilities
    reference_probabilities = predict_probabilities(reference)

    report = {
        "generated": date.today().isoformat(),
        "mode": "self-test (no production data yet)" if self_test else "production comparison",
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features_checked": int(len(combined)),
        "features_drifted": int(combined["drifted"].sum()),
        "max_psi": round(float(combined["psi"].max()), 4),
        "prediction_drift": prediction_drift(reference_probabilities),
        "features": combined.to_dict("records"),
    }

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"drift_report_{date.today():%Y%m%d}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        combined.to_csv(REPORTS_DIR / "drift_features_latest.csv", index=False)
        log.info("Drift report written to %s", path.name)

    return report


def main() -> dict:
    report = run()
    print(f"Mode              : {report['mode']}")
    print(f"Features checked  : {report['features_checked']}")
    print(f"Features drifted  : {report['features_drifted']}")
    print(f"Max PSI           : {report['max_psi']}")
    print("\nPer-feature:")
    print(pd.DataFrame(report["features"])[
        ["feature", "type", "psi", "psi_band", "drifted"]].to_string(index=False))
    print(f"\nPrediction drift: {report['prediction_drift']}")
    return report


if __name__ == "__main__":
    main()
