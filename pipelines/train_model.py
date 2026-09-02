"""Day 2 - train, compare and version the attrition model (steps 06, 07, 09).

Three models on one identical split and one identical preprocessor, so the
comparison measures algorithms rather than accidental differences in preparation.

Two decisions worth calling out:

1. **Never accuracy.** The target is ~16% positive, so predicting "stays" for
   everybody scores 84%. The model is selected on ROC-AUC and PR-AUC, which measure
   how well it *orders* employees by risk and do not depend on where the cut falls.

2. **The decision threshold is tuned, not left at 0.5.** The build notes say missing a
   genuinely high-risk employee is the expensive error. With an imbalanced target a
   0.5 cut buys precision nobody asked for at the cost of recall that matters, so the
   threshold is chosen on the validation folds to maximise F2 (recall weighted twice
   as heavily as precision) and stored in the model metadata.
"""
from __future__ import annotations

import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from app.ml.features import build_preprocessor, engineer_features
from app.utils.config import (
    MODELS_DIR,
    PROCESSED_FILES,
    RANDOM_STATE,
    TEST_SIZE,
    available_model_versions,
)
from app.utils.logger import get_logger

log = get_logger("train")

# Recall weighted twice as heavily as precision - the cost asymmetry above.
FBETA = 2.0


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED_FILES["attrition"])
    y = df["AttritionFlag"].astype(int)
    return df, y


def build_candidates(y_train: pd.Series) -> dict[str, object]:
    """Three candidates, each told about the class imbalance in its own idiom."""
    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def make_pipeline(estimator, X_sample: pd.DataFrame) -> Pipeline:
    """Engineering + preprocessing + model as one artifact.

    Because engineering is a pipeline step, the saved .joblib takes a raw employee
    record and does everything itself - the API cannot drift from training.
    """
    engineered = engineer_features(X_sample)
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_features, validate=False)),
        ("preprocess", build_preprocessor(engineered)),
        ("model", estimator),
    ])


def tune_threshold(y_true: np.ndarray, probabilities: np.ndarray, beta: float = FBETA) -> tuple[float, float]:
    """Pick the probability cut that maximises F-beta on out-of-fold predictions."""
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.05, 0.95, 0.01):
        score = fbeta_score(y_true, (probabilities >= threshold).astype(int), beta=beta, zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return round(best_threshold, 2), round(best_score, 4)


def evaluate(y_true, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true, predictions, beta=FBETA, zero_division=0)), 4),
        "threshold": threshold,
    }


def compare_models(X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict, dict]:
    """Cross-validate each candidate, tune its threshold, then score it on the test set."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted, thresholds = [], {}, {}

    for name, estimator in build_candidates(y_train).items():
        pipeline = make_pipeline(estimator, X_train)

        # Out-of-fold probabilities: the threshold is chosen without ever seeing the
        # test set, so the reported test metrics stay honest.
        oof = cross_val_predict(pipeline, X_train, y_train, cv=cv,
                                method="predict_proba", n_jobs=1)[:, 1]
        threshold, cv_f2 = tune_threshold(y_train.to_numpy(), oof)
        cv_metrics = evaluate(y_train.to_numpy(), oof, threshold)

        pipeline.fit(X_train, y_train)
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test.to_numpy(), test_probabilities, threshold)

        fitted[name] = pipeline
        thresholds[name] = threshold
        results.append({
            "model": name,
            "cv_roc_auc": cv_metrics["roc_auc"],
            "cv_f2": cv_f2,
            "threshold": threshold,
            **{f"test_{k}": v for k, v in test_metrics.items() if k != "threshold"},
        })
        log.info("%s: cv_auc=%.4f test_auc=%.4f test_recall=%.4f",
                 name, cv_metrics["roc_auc"], test_metrics["roc_auc"], test_metrics["recall"])

    return pd.DataFrame(results).sort_values("test_roc_auc", ascending=False), fitted, thresholds


def select_winner(comparison: pd.DataFrame) -> str:
    """Select on threshold-independent ranking quality: ROC-AUC and PR-AUC.

    Choosing a model and choosing an operating point are two different decisions and
    are kept separate here. Ranking F2 across models would conflate them - each model
    carries its own tuned threshold, so a model can post a high F2 purely by cutting
    low enough to flag most of the workforce. ROC-AUC and PR-AUC ask the question that
    actually matters first: which model orders employees by risk best?

    PR-AUC gets equal weight because on a 16%-positive target it is far more
    informative than ROC-AUC about performance on the minority class.

    Once a winner is chosen, its tuned threshold sets where the risk bands fall.
    """
    scored = comparison.assign(
        combined=lambda d: 0.5 * d["test_roc_auc"] + 0.5 * d["test_pr_auc"]
    ).sort_values("combined", ascending=False)
    return str(scored.iloc[0]["model"])


def next_version() -> str:
    existing = available_model_versions()
    return f"v{len(existing) + 1}"


def save_model(pipeline: Pipeline, name: str, metrics: dict, threshold: float,
               n_train: int, n_test: int, feature_count: int) -> str:
    version = next_version()
    folder = MODELS_DIR / version
    folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, folder / "attrition_pipeline.joblib")

    metadata = {
        "model_name": "Attrition Prediction Model",
        "version": f"{version}.0",
        "version_folder": version,
        "algorithm": name,
        "training_date": date.today().isoformat(),
        "decision_threshold": threshold,
        "selection_metric": "0.5 * ROC-AUC + 0.5 * PR-AUC (threshold-independent)",
        "threshold_metric": f"F{FBETA:.0f} maximised on out-of-fold predictions",
        "training_rows": n_train,
        "test_rows": n_test,
        "engineered_feature_count": feature_count,
        "random_state": RANDOM_STATE,
        **metrics,
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("saved %s (%s) to %s", name, version, folder)
    return version


def main() -> dict:
    df, y = load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info("train=%d test=%d positive_rate=%.4f", len(X_train), len(X_test), y.mean())

    comparison, fitted, thresholds = compare_models(X_train, y_train, X_test, y_test)
    print("\n=== Model comparison ===")
    print(comparison.to_string(index=False))

    winner = select_winner(comparison)
    print(f"\nSelected: {winner}")

    row = comparison[comparison["model"] == winner].iloc[0]
    metrics = {k.replace("test_", ""): float(row[k]) for k in row.index if k.startswith("test_")}
    feature_count = engineer_features(X_train).shape[1]

    version = save_model(
        fitted[winner], winner, metrics, thresholds[winner],
        len(X_train), len(X_test), feature_count,
    )

    comparison.to_csv(PROCESSED_FILES["attrition"].parent / "model_comparison.csv", index=False)
    return {"version": version, "winner": winner, "comparison": comparison,
            "pipelines": fitted, "split": (X_train, X_test, y_train, y_test)}


if __name__ == "__main__":
    main()
