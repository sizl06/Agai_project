"""Feature engineering for the attrition model.

Every engineered feature below has a stated reason. Anything that seemed
interesting but had no argument behind it was left out - unmotivated features are
noise the model has to learn to ignore.

This module is imported by both training and serving, and the engineering step is
embedded *inside* the sklearn pipeline as a FunctionTransformer. That is deliberate:
if engineering lived only in the training script, the API could silently compute a
feature differently and every prediction would be subtly wrong with nothing failing
loudly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.utils.config import ID_COL, TARGET_COL

# Never features. EmployeeID is an arbitrary identifier - a tree will happily carve it
# into "risky" ranges that mean nothing. The target columns are the answer.
EXCLUDED_COLUMNS = [ID_COL, TARGET_COL, "AttritionFlag"]

ENGINEERED_COLUMNS = [
    "IncomePerYearAtCompany",
    "TenureRatio",
    "PromotionStagnation",
    "AvgYearsPerCompany",
    "IncomePerJobLevel",
    "SatisfactionIndex",
    "ManagerStability",
]

# Why each engineered feature exists:
#
#   IncomePerYearAtCompany  pay relative to tenure. Someone paid little after years
#                           of service is a classic flight risk; raw income alone
#                           does not express that.
#   TenureRatio             share of a whole career spent here. A low ratio means a
#                           mover; a high one means someone invested in the company.
#   PromotionStagnation     time since promotion relative to time in role. Being
#                           "due" is what matters, not the absolute year count.
#   AvgYearsPerCompany      average stint length across the career - a direct,
#                           behavioural measure of job-hopping.
#   IncomePerJobLevel       pay fairness within a grade. Under-payment relative to
#                           peers at the same level drives exits.
#   SatisfactionIndex       one combined signal from four correlated Likert items,
#                           which is more stable than any single question.
#   ManagerStability        years with the current manager relative to tenure. A
#                           recent manager change is a well-known attrition trigger.


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns and drop identifiers/targets. Input is not modified."""
    X = df.copy()

    # +1 denominators keep a zero-tenure employee from producing an infinity.
    X["IncomePerYearAtCompany"] = X["MonthlyIncome"] / (X["YearsAtCompany"] + 1)
    X["TenureRatio"] = X["YearsAtCompany"] / (X["TotalWorkingYears"] + 1)
    X["PromotionStagnation"] = X["YearsSinceLastPromotion"] / (X["YearsInCurrentRole"] + 1)
    X["AvgYearsPerCompany"] = X["TotalWorkingYears"] / (X["NumCompaniesWorked"] + 1)
    X["IncomePerJobLevel"] = X["MonthlyIncome"] / X["JobLevel"].clip(lower=1)
    X["SatisfactionIndex"] = X[
        ["JobSatisfaction", "EnvironmentSatisfaction",
         "RelationshipSatisfaction", "JobInvolvement"]
    ].mean(axis=1)
    X["ManagerStability"] = X["YearsWithCurrManager"] / (X["YearsAtCompany"] + 1)

    X = X.drop(columns=[c for c in EXCLUDED_COLUMNS if c in X.columns])
    return X.replace([np.inf, -np.inf], np.nan)


def split_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical


def build_preprocessor(X_engineered: pd.DataFrame) -> ColumnTransformer:
    """Scale numerics, one-hot the categoricals.

    Scaling is only strictly needed by Logistic Regression, but keeping one shared
    preprocessor means all three models in notebook 07 are compared on identical
    inputs - otherwise the comparison measures preprocessing, not algorithms.
    """
    numeric, categorical = split_column_types(X_engineered)

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Unknown categories at serving time must not crash a live prediction.
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def check_leakage(df: pd.DataFrame, target: pd.Series, threshold: float = 0.95) -> pd.DataFrame:
    """Flag any single feature that predicts the target almost perfectly.

    A feature scoring above the threshold is nearly always a proxy for the answer
    that would not actually be available at prediction time.

    Both metrics are expressed as "how well could you predict the target knowing only
    this column", so they are comparable to each other and to the base rate. For a
    numeric column that is |correlation|; for a categorical one it is the accuracy of
    always guessing the majority class within each category, rescaled so that
    predicting no better than the base rate reads as 0.
    """
    base_rate = max(target.mean(), 1 - target.mean())
    rows = []

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            value = abs(s.corr(target))
            value = 0.0 if pd.isna(value) else float(value)
            metric = "abs_correlation"
        else:
            grouped = target.groupby(s, observed=True)
            purity = grouped.apply(lambda t: max(t.mean(), 1 - t.mean()))
            counts = grouped.size()
            accuracy = float((purity * counts).sum() / len(target))
            # Rescale: base rate -> 0, perfect separation -> 1.
            value = (accuracy - base_rate) / max(1 - base_rate, 1e-9)
            metric = "lift_over_base_rate"

        rows.append({
            "feature": col,
            "metric": metric,
            "value": round(value, 4),
            "suspicious": bool(value >= threshold),
        })

    return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
