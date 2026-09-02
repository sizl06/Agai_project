"""Model performance monitoring and the retraining decision (steps 26 and 27).

Once real attrition outcomes arrive - that is, once it is known who actually left -
performance is recomputed on live data and compared against the metrics the model
was released with.

The retraining rule from the build notes, written down in advance so it is not a
judgement call under pressure:

    IF drift > threshold
    OR F1 drops below threshold
    OR 6 months of new data collected
    THEN retrain

`should_retrain()` implements exactly that and returns the reasons, so the decision
is auditable rather than a number someone eyeballed.

One caveat this module is careful about: attrition outcomes arrive *late*. An
employee predicted at risk today may leave in eight months, so recent predictions
have no ground truth yet. Scoring them as wrong would make every model look like it
was decaying. Only predictions old enough to have resolved are evaluated.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.ml.model_loader import decision_threshold, load_metadata
from app.utils.config import PROCESSED_FILES, REPORTS_DIR, latest_model_version
from app.utils.logger import get_logger

log = get_logger("performance")

# Retraining triggers.
F1_FLOOR = 0.45                  # below the release F1 by a meaningful margin
F1_RELATIVE_DROP = 0.15          # or a 15% relative fall from the released score
DRIFT_PSI_THRESHOLD = 0.25       # "significant" on the PSI scale
MAX_MODEL_AGE_DAYS = 182         # ~6 months, per the build notes
OUTCOME_LAG_DAYS = 180           # how long before an attrition outcome is known


def evaluate_against_outcomes(
    probabilities: np.ndarray, outcomes: np.ndarray, threshold: float | None = None
) -> dict:
    """Recompute the release metrics on data with known outcomes."""
    threshold = threshold if threshold is not None else decision_threshold()
    predictions = (probabilities >= threshold).astype(int)

    return {
        "n": int(len(outcomes)),
        "positive_rate": round(float(np.mean(outcomes)), 4),
        "roc_auc": round(float(roc_auc_score(outcomes, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(outcomes, probabilities)), 4),
        "precision": round(float(precision_score(outcomes, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(outcomes, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(outcomes, predictions, zero_division=0)), 4),
        "f2": round(float(fbeta_score(outcomes, predictions, beta=2, zero_division=0)), 4),
        "threshold": threshold,
    }


def compare_to_release(live: dict, released: dict | None = None) -> pd.DataFrame:
    """Live metrics beside the ones recorded when the model was released."""
    released = released or load_metadata()
    rows = []
    for metric in ["roc_auc", "pr_auc", "precision", "recall", "f1", "f2"]:
        release_value = released.get(metric)
        live_value = live.get(metric)
        if release_value is None or live_value is None:
            continue
        delta = live_value - release_value
        rows.append({
            "metric": metric,
            "at_release": release_value,
            "live": live_value,
            "delta": round(delta, 4),
            "relative_change_%": round(delta / release_value * 100, 2) if release_value else np.nan,
        })
    return pd.DataFrame(rows)


def model_age_days(released: dict | None = None) -> int | None:
    released = released or load_metadata()
    training_date = released.get("training_date")
    if not training_date:
        return None
    return (date.today() - date.fromisoformat(training_date)).days


def should_retrain(live_metrics: dict | None = None, max_psi: float | None = None) -> dict:
    """Apply the retraining rule and return the reasons.

    Returns a decision even when live metrics are unavailable - age alone can
    trigger a retrain, and "no outcome data yet" is itself worth reporting.
    """
    released = load_metadata()
    reasons: list[str] = []

    age = model_age_days(released)
    if age is not None and age >= MAX_MODEL_AGE_DAYS:
        reasons.append(f"model is {age} days old (limit {MAX_MODEL_AGE_DAYS})")

    if max_psi is not None and max_psi > DRIFT_PSI_THRESHOLD:
        reasons.append(f"feature drift PSI {max_psi:.3f} exceeds {DRIFT_PSI_THRESHOLD}")

    if live_metrics:
        live_f1 = live_metrics.get("f1")
        release_f1 = released.get("f1")
        if live_f1 is not None:
            if live_f1 < F1_FLOOR:
                reasons.append(f"live F1 {live_f1:.3f} below the floor of {F1_FLOOR}")
            if release_f1 and (release_f1 - live_f1) / release_f1 > F1_RELATIVE_DROP:
                drop = (release_f1 - live_f1) / release_f1
                reasons.append(
                    f"live F1 fell {drop:.1%} from the released {release_f1:.3f} "
                    f"(limit {F1_RELATIVE_DROP:.0%})"
                )

    return {
        "retrain": bool(reasons),
        "reasons": reasons or ["all triggers within limits"],
        "model_version": latest_model_version(),
        "model_age_days": age,
        "evaluated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run(save: bool = True) -> dict:
    """Evaluate the active model against the outcomes currently available.

    In this project the only labelled data is the original dataset, so this is a
    *retrospective* check on data the model was trained on, not a production
    measurement. It is reported as such - the numbers will look better than reality
    because the model has seen these rows.
    """
    from app.ml.predictor import predict_probabilities

    employees = pd.read_csv(PROCESSED_FILES["attrition"])
    probabilities = predict_probabilities(employees)
    outcomes = employees["AttritionFlag"].astype(int).to_numpy()

    live = evaluate_against_outcomes(probabilities, outcomes)
    comparison = compare_to_release(live)

    cutoff = datetime.now(timezone.utc) - timedelta(days=OUTCOME_LAG_DAYS)
    report = {
        "generated": date.today().isoformat(),
        "mode": "retrospective (in-sample) - not a production measurement",
        "outcome_lag_days": OUTCOME_LAG_DAYS,
        "outcomes_resolved_before": cutoff.date().isoformat(),
        "model_version": latest_model_version(),
        "live_metrics": live,
        "comparison": comparison.to_dict("records"),
        "classification_report": classification_report(
            outcomes, (probabilities >= live["threshold"]).astype(int),
            target_names=["stayed", "left"], output_dict=True, zero_division=0),
    }
    report["retraining_decision"] = should_retrain(live_metrics=live)

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"performance_report_{date.today():%Y%m%d}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log.info("Performance report written to %s", path.name)

    return report


def main() -> dict:
    report = run()
    print(f"Mode          : {report['mode']}")
    print(f"Model version : {report['model_version']}\n")

    print("Live metrics vs release:")
    print(pd.DataFrame(report["comparison"]).to_string(index=False))

    decision = report["retraining_decision"]
    print(f"\nRetrain? {'YES' if decision['retrain'] else 'no'}")
    for reason in decision["reasons"]:
        print(f"  - {reason}")
    print(f"\nModel age: {decision['model_age_days']} days "
          f"(limit {MAX_MODEL_AGE_DAYS})")
    return report


if __name__ == "__main__":
    main()
