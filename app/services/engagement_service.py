"""Engagement analytics (step 10).

No ML here, and none is needed. These are aggregations - averages, breakdowns and a
ranked list of the least engaged people. The build notes are right that a model
would add nothing until there is a target worth predicting.

One constraint carried forward from Day 1: `EngagementScore` is a deterministic
function of five columns the attrition model also uses. Correlating engagement with
attrition *risk* is therefore not independent evidence, and is reported here with
that caveat attached rather than presented as a discovery.
"""
from __future__ import annotations

import pandas as pd

from app.utils.config import ID_COL, PROCESSED_FILES

# Bands for reporting. 0-100 scale, so thirds with a slightly generous "high".
ENGAGEMENT_BANDS = [(0, 40, "Low"), (40, 70, "Moderate"), (70, 101, "High")]


def load_engagement() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_FILES["engagement"])


def band(score: float) -> str:
    for low, high, name in ENGAGEMENT_BANDS:
        if low <= score < high:
            return name
    return "Unknown"


def summary(engagement: pd.DataFrame | None = None) -> dict:
    """Headline engagement figures for the dashboard."""
    df = load_engagement() if engagement is None else engagement
    scores = df["EngagementScore"]
    return {
        "employees": int(len(df)),
        "average_engagement": round(float(scores.mean()), 1),
        "median_engagement": round(float(scores.median()), 1),
        "std_engagement": round(float(scores.std()), 1),
        "lowest": round(float(scores.min()), 1),
        "highest": round(float(scores.max()), 1),
        "pct_low_engagement": round(float((scores < 40).mean() * 100), 1),
        "pct_high_engagement": round(float((scores >= 70).mean() * 100), 1),
    }


def by_department(engagement: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_engagement() if engagement is None else engagement
    out = (
        df.groupby("Department")
        .agg(
            headcount=(ID_COL, "count"),
            avg_engagement=("EngagementScore", "mean"),
            median_engagement=("EngagementScore", "median"),
            avg_performance=("PerformanceRating", "mean"),
            pct_low=("EngagementScore", lambda s: (s < 40).mean() * 100),
        )
        .round(2)
        .sort_values("avg_engagement")
    )
    return out.reset_index()


def by_role(engagement: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_engagement() if engagement is None else engagement
    return (
        df.groupby("JobRole")
        .agg(headcount=(ID_COL, "count"), avg_engagement=("EngagementScore", "mean"))
        .round(2)
        .sort_values("avg_engagement")
        .reset_index()
    )


def lowest_engaged(n: int = 20, engagement: pd.DataFrame | None = None) -> pd.DataFrame:
    """The people HR should look at directly."""
    df = load_engagement() if engagement is None else engagement
    columns = [ID_COL, "Department", "JobRole", "EngagementScore",
               "JobSatisfactionScore", "WorkLifeBalanceScore", "PerformanceRating"]
    out = df.nsmallest(n, "EngagementScore")[columns].copy()
    out["EngagementBand"] = out["EngagementScore"].map(band)
    return out.reset_index(drop=True)


def band_distribution(engagement: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_engagement() if engagement is None else engagement
    bands = df["EngagementScore"].map(band)
    out = bands.value_counts().rename_axis("band").reset_index(name="employees")
    out["percent"] = (out["employees"] / len(df) * 100).round(1)
    order = {"Low": 0, "Moderate": 1, "High": 2}
    return out.sort_values("band", key=lambda s: s.map(order)).reset_index(drop=True)


def high_performer_low_engagement(engagement: pd.DataFrame | None = None,
                                  engagement_cutoff: float = 40.0) -> pd.DataFrame:
    """Strong performers who are disengaged - the most expensive group to lose."""
    df = load_engagement() if engagement is None else engagement
    mask = (df["PerformanceRating"] >= 4) & (df["EngagementScore"] < engagement_cutoff)
    columns = [ID_COL, "Department", "JobRole", "EngagementScore", "PerformanceRating"]
    return df.loc[mask, columns].sort_values("EngagementScore").reset_index(drop=True)
