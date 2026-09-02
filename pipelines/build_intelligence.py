"""Day 3, step 16 - the Employee Intelligence table.

Everything from Days 1-3 converges here: one row per employee carrying attrition
risk, engagement, role, skill gaps and a recommendation. This table is the actual
business output of the project; the dashboard is a view onto it.

The join discipline established in notebook 04 is what keeps it correct. The
employee table is the spine, and every other source is aggregated to one row per
employee *before* being joined. Skill gaps arrive at ~15,500 rows and
recommendations at ~4,400 - joining either directly would silently multiply the
spine and inflate every count on the dashboard.
"""
from __future__ import annotations

import pandas as pd

from app.ml.predictor import predict_batch
from app.services import engagement_service, recommendation_service, skill_gap_service
from app.utils.config import ID_COL, PROCESSED_FILES
from app.utils.logger import get_logger

log = get_logger("intelligence")


def build() -> pd.DataFrame:
    employees = pd.read_csv(PROCESSED_FILES["attrition"])
    engagement = pd.read_csv(PROCESSED_FILES["engagement"])
    log.info("spine: %d employees", len(employees))

    # --- attrition risk -----------------------------------------------------
    predictions = predict_batch(employees)

    # --- skill gaps ---------------------------------------------------------
    gaps, org_gaps = skill_gap_service.build_and_save()
    gap_summary = skill_gap_service.employee_gap_summary(gaps, employees)
    log.info("skill gaps: %d rows across %d distinct skills",
             len(gaps), gaps["MissingSkill"].nunique())

    # --- recommendations ----------------------------------------------------
    risk_by_employee = dict(zip(predictions[ID_COL], predictions["RiskLevel"]))
    recommendations = recommendation_service.build_recommendations(
        gaps, org_gaps, risk_by_employee=risk_by_employee)
    recommendation_service.save(recommendations)
    top_recommendation = recommendation_service.top_recommendation_per_employee(recommendations)
    log.info("recommendations: %d rows, %d employees covered",
             len(recommendations), recommendations[ID_COL].nunique() if len(recommendations) else 0)

    # --- assemble -----------------------------------------------------------
    intelligence = (
        employees[[ID_COL, "Department", "JobRole", "Age", "MonthlyIncome",
                   "YearsAtCompany", "OverTime", "Attrition"]]
        .merge(predictions, on=ID_COL, how="left")
        .merge(
            engagement[[ID_COL, "EngagementScore", "PerformanceRating"]],
            on=ID_COL, how="left",
        )
        .merge(
            gap_summary[[ID_COL, "SkillGapCount", "SkillGapScore", "SkillGaps", "TopMissingSkill"]],
            on=ID_COL, how="left",
        )
        .merge(
            top_recommendation[[ID_COL, "Recommendation", "CourseTitle", "Provider",
                                "DurationHours", "Priority"]],
            on=ID_COL, how="left",
        )
    )

    # An employee with no gaps has no recommendation - that is a real state, not a bug.
    intelligence["Recommendation"] = intelligence["Recommendation"].fillna("No action required")
    intelligence["CourseTitle"] = intelligence["CourseTitle"].fillna("")
    intelligence["Provider"] = intelligence["Provider"].fillna("")
    intelligence["SkillGaps"] = intelligence["SkillGaps"].fillna("")
    intelligence["TopMissingSkill"] = intelligence["TopMissingSkill"].fillna("")
    intelligence["SkillGapCount"] = intelligence["SkillGapCount"].fillna(0).astype(int)
    intelligence["EngagementBand"] = intelligence["EngagementScore"].map(engagement_service.band)

    # The spine must not have grown.
    if len(intelligence) != len(employees):
        raise ValueError(
            f"intelligence table has {len(intelligence)} rows but there are "
            f"{len(employees)} employees - a join multiplied the spine"
        )
    if intelligence[ID_COL].duplicated().any():
        raise ValueError("duplicate EmployeeID in the intelligence table")

    intelligence.to_csv(PROCESSED_FILES["intelligence"], index=False)
    log.info("intelligence table: %s -> %s", intelligence.shape,
             PROCESSED_FILES["intelligence"].name)
    return intelligence


def main() -> pd.DataFrame:
    intelligence = build()
    print(f"\nEmployee Intelligence table: {intelligence.shape[0]:,} rows x "
          f"{intelligence.shape[1]} columns")
    print(f"\nRisk distribution:\n{intelligence['RiskLevel'].value_counts().to_string()}")
    print(f"\nAverage engagement: {intelligence['EngagementScore'].mean():.1f}")
    print(f"Employees with at least one skill gap: "
          f"{int((intelligence['SkillGapCount'] > 0).sum()):,}")
    return intelligence


if __name__ == "__main__":
    main()
