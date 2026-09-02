"""SHAP explanations for attrition predictions.

A prediction like "Employee 101 - 82% attrition risk" is useless to an HR manager
without an answer to "why". SHAP provides both levels the build notes ask for:

  global - what drives attrition across the whole company
  local  - why this specific employee was flagged

The explainer is chosen from the fitted estimator rather than hard-coded, because
the winning algorithm is decided by notebook 07 and can change on a retrain. The
build notes assume `shap.TreeExplainer`; the model that actually won is linear, so
committing to TreeExplainer would break the moment it was run.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def split_pipeline(pipeline: Pipeline):
    """Separate the transform stages from the final estimator.

    SHAP explains the estimator, so it needs the design matrix the estimator
    actually sees - not the raw employee record.
    """
    transform = Pipeline(pipeline.steps[:-1])
    estimator = pipeline.steps[-1][1]
    return transform, estimator


def feature_names(pipeline: Pipeline) -> np.ndarray:
    return pipeline.named_steps["preprocess"].get_feature_names_out()


def transform(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    transform_stage, _ = split_pipeline(pipeline)
    matrix = transform_stage.transform(df)
    return pd.DataFrame(matrix, columns=feature_names(pipeline), index=df.index)


def build_explainer(pipeline: Pipeline, background: pd.DataFrame):
    """Pick the explainer that matches the fitted estimator."""
    _, estimator = split_pipeline(pipeline)
    background_matrix = transform(pipeline, background)

    name = type(estimator).__name__
    if name in {"RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier",
                "DecisionTreeClassifier", "LGBMClassifier"}:
        return shap.TreeExplainer(estimator), background_matrix
    if name in {"LogisticRegression", "LinearSVC", "RidgeClassifier"}:
        return shap.LinearExplainer(estimator, background_matrix), background_matrix

    # Anything else: the model-agnostic sampler. Slower, always correct.
    return shap.Explainer(estimator.predict_proba, background_matrix), background_matrix


def _positive_class(values) -> np.ndarray:
    """Normalise SHAP output to the positive class.

    Tree explainers on a binary classifier may return a single array, a two-element
    list, or a 3-D array with one slice per class.
    """
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def _expected_value(explainer) -> float:
    """The explainer's base value, as a scalar for the positive class."""
    base = getattr(explainer, "expected_value", 0.0)
    if isinstance(base, (list, tuple, np.ndarray)):
        base = np.asarray(base).ravel()
        base = base[1] if base.size == 2 else base[0]
    return float(base)


def shap_values(pipeline: Pipeline, df: pd.DataFrame, background: pd.DataFrame | None = None) -> tuple:
    """Return (shap_values_for_positive_class, transformed_frame)."""
    background = df if background is None else background
    explainer, _ = build_explainer(pipeline, background)
    matrix = transform(pipeline, df)
    return _positive_class(explainer.shap_values(matrix)), matrix


def explanation(pipeline: Pipeline, df: pd.DataFrame,
                background: pd.DataFrame | None = None) -> "shap.Explanation":
    """A full shap.Explanation, carrying the base value the waterfall plot needs."""
    background = df if background is None else background
    explainer, _ = build_explainer(pipeline, background)
    matrix = transform(pipeline, df)

    return shap.Explanation(
        values=_positive_class(explainer.shap_values(matrix)),
        base_values=_expected_value(explainer),
        data=matrix.values,
        feature_names=list(matrix.columns),
    )


def _split_camel(name: str) -> str:
    """YearsSinceLastPromotion -> Years since last promotion."""
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    return spaced[0].upper() + spaced[1:].lower()


def humanize_feature(feature: str, raw_row: pd.DataFrame | None = None) -> str:
    """Turn an encoded column name into something an HR manager can read.

    One-hot encoding produces both `OverTime_Yes` and `OverTime_No`, and a negated
    dummy can carry a positive contribution for an employee who *does* work overtime
    (see notebook 08). Echoing the encoded name onto a dashboard invites exactly the
    wrong reading, so the underlying question and the employee's actual answer are
    rendered instead.
    """
    if raw_row is not None and "_" in feature:
        prefix = feature.split("_", 1)[0]
        if prefix in raw_row.columns:
            return f"{_split_camel(prefix)}: {raw_row[prefix].iloc[0]}"

    if raw_row is not None and feature in raw_row.columns:
        return f"{_split_camel(feature)}: {raw_row[feature].iloc[0]}"

    return _split_camel(feature.replace("_", " ")) if "_" not in feature else feature


def source_column(feature: str, raw_columns) -> str:
    """Map an encoded feature back to the raw column it came from.

    `OverTime_Yes` and `OverTime_No` both originate in `OverTime`.
    """
    if feature in raw_columns:
        return feature
    if "_" in feature:
        prefix = feature.split("_", 1)[0]
        if prefix in raw_columns:
            return prefix
    return feature


def top_factors(pipeline: Pipeline, row: pd.DataFrame, background: pd.DataFrame,
                n: int = 3) -> list[dict]:
    """The n drivers pushing this single prediction hardest, with direction.

    Contributions are aggregated back to the **raw column** before ranking. One-hot
    encoding splits one question across several columns - `OverTime_Yes` and
    `OverTime_No` are the same fact stated twice - and ranking the encoded columns
    directly lets a single variable occupy two of the three slots an HR manager sees.
    Summing them is also the honest total: SHAP is additive, so the contribution of
    "overtime" is the sum of its dummies.
    """
    values, matrix = shap_values(pipeline, row, background=background)
    contributions = pd.Series(values[0], index=matrix.columns)

    grouped = contributions.groupby(
        [source_column(f, row.columns) for f in contributions.index]
    ).sum()
    ranked = grouped.reindex(grouped.abs().sort_values(ascending=False).index).head(n)

    return [
        {
            "feature": str(name),
            "label": humanize_feature(str(name), row),
            "shap_value": round(float(value), 4),
            "direction": "increases risk" if value > 0 else "reduces risk",
        }
        for name, value in ranked.items()
    ]


def global_importance(pipeline: Pipeline, df: pd.DataFrame,
                      background: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean absolute SHAP value per feature - the company-wide view."""
    values, matrix = shap_values(pipeline, df, background=background)
    return (
        pd.DataFrame({
            "feature": matrix.columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
