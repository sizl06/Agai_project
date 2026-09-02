"""
Build the two source tables that could not be downloaded from a public origin.

PROVENANCE - read this before trusting these files
--------------------------------------------------
Three of the five raw datasets in the build notes are genuine public data:
    employee_attrition.csv  <- IBM/employee-attrition-aif360 (IBM HR Analytics, 1,470 employees)
    occupation_data.csv     <- O*NET database 31.0
    essential_skills.csv    <- O*NET database 31.0
    software_skills.csv     <- O*NET database 31.0

Two could not be obtained:
    hr_performance_engagement.csv  - the Kaggle source needs API credentials.
    employee_current_skills.csv    - no public dataset records per-employee skill inventories.

Both are DERIVED here rather than downloaded, for a specific reason: the build notes
specify that engagement joins to attrition 1:1 on the same employees. An unrelated
Kaggle HR file describes a different population, so that join would be meaningless -
matching column names are not a matching key. Deriving from the IBM population keeps
the join real.

hr_performance_engagement.csv is a TRANSFORMATION of real IBM columns (the five Likert
satisfaction/involvement items plus PerformanceRating). No values are invented.
Because it is a deterministic function of columns that are also model features, it is
used for ANALYTICS ONLY and never fed back into the attrition model - that would be
circular. See docs/data_relationships.md.

employee_current_skills.csv is SYNTHESISED, which the build notes explicitly sanction
("If it doesn't exist -> build a controlled table for the MVP"). Skill vocabulary is
real O*NET data; which employee holds which skill is modelled from real employee
attributes (Education, TotalWorkingYears, JobLevel, TrainingTimesLastYear) under a
fixed seed. Treat organisation-wide gap counts as illustrative of the mechanism, not
as measurements of a real workforce.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOCS = ROOT / "docs"
SEED = 42

# IBM JobRole -> O*NET occupation. Each IBM role maps to a distinct SOC code so that
# role-level skill requirements do not collide.
ROLE_TO_SOC = {
    "Sales Executive": (
        "41-4012.00",
        "Sales Representatives, Wholesale and Manufacturing, Except Technical and Scientific Products",
    ),
    "Sales Representative": ("41-3091.00", "Sales Representatives of Services"),
    "Healthcare Representative": (
        "41-4011.00",
        "Sales Representatives, Wholesale and Manufacturing, Technical and Scientific Products",
    ),
    "Manager": ("11-1021.00", "General and Operations Managers"),
    "Human Resources": ("13-1071.00", "Human Resources Specialists"),
    "Research Scientist": ("19-1042.00", "Medical Scientists, Except Epidemiologists"),
    "Research Director": ("11-9121.00", "Natural Sciences Managers"),
    "Laboratory Technician": ("29-2012.00", "Medical and Clinical Laboratory Technicians"),
    "Manufacturing Director": ("11-3051.00", "Industrial Production Managers"),
}

# Engagement weights. Job satisfaction and involvement dominate; work-life balance and
# environment contribute; relationship satisfaction is the smallest single driver.
ENGAGEMENT_WEIGHTS = {
    "JobSatisfaction": 0.30,
    "JobInvolvement": 0.25,
    "EnvironmentSatisfaction": 0.20,
    "WorkLifeBalance": 0.15,
    "RelationshipSatisfaction": 0.10,
}


def load_attrition() -> pd.DataFrame:
    df = pd.read_csv(RAW / "employee_attrition.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    # The build notes assume EmployeeID; the real IBM file ships EmployeeNumber.
    return df.rename(columns={"EmployeeNumber": "EmployeeID"})


def build_engagement(emp: pd.DataFrame) -> pd.DataFrame:
    """Rescale the real 1-4 Likert items to 0-100 and combine into one score."""

    def to_100(s: pd.Series) -> pd.Series:
        return ((s - 1) / 3.0 * 100).round(1)

    out = pd.DataFrame(
        {
            "EmployeeID": emp["EmployeeID"],
            "Department": emp["Department"],
            "JobRole": emp["JobRole"],
            "JobSatisfactionScore": to_100(emp["JobSatisfaction"]),
            "JobInvolvementScore": to_100(emp["JobInvolvement"]),
            "EnvironmentSatisfactionScore": to_100(emp["EnvironmentSatisfaction"]),
            "WorkLifeBalanceScore": to_100(emp["WorkLifeBalance"]),
            "RelationshipSatisfactionScore": to_100(emp["RelationshipSatisfaction"]),
            "PerformanceRating": emp["PerformanceRating"],
            "TrainingTimesLastYear": emp["TrainingTimesLastYear"],
        }
    )
    score = sum(out[f"{k}Score"] * w for k, w in ENGAGEMENT_WEIGHTS.items())
    out["EngagementScore"] = score.round(1)
    out["ReviewPeriod"] = "2026-H1"
    cols = ["EmployeeID", "Department", "JobRole", "ReviewPeriod", "EngagementScore"]
    return out[cols + [c for c in out.columns if c not in cols]]


def role_requirements(top_skills: int = 8, top_software: int = 10) -> pd.DataFrame:
    """Required skills per mapped role, from O*NET importance ratings.

    Software requirements use the concrete tool in `Workplace Example` ("Microsoft
    Excel", "Amazon Web Services AWS software") rather than the broad O*NET category
    ("Spreadsheet software"), because a training recommendation has to name a real
    thing to learn. The raw names are inconsistent - that is cleaned downstream by
    app.services.skill_normalizer, not here.
    """
    ess = pd.read_csv(RAW / "essential_skills.csv")
    ess = ess[(ess["Scale ID"] == "IM") & (ess["Recommend Suppress"] != "Y")]
    soft = pd.read_csv(RAW / "software_skills.csv")

    rows = []
    for role, (soc, _title) in ROLE_TO_SOC.items():
        e = (
            ess[ess["O*NET-SOC Code"] == soc]
            .sort_values("Data Value", ascending=False)
            .head(top_skills)
        )
        for rank, (_, r) in enumerate(e.iterrows(), start=1):
            rows.append(
                (role, soc, r["Element Name"], "Essential Skill", r["Element Name"],
                 round(float(r["Data Value"]), 2), rank)
            )

        s = soft[soft["O*NET-SOC Code"] == soc].copy()
        # Prefer tools O*NET flags as in-demand or hot; they are the ones worth training.
        s["prio"] = (s["In Demand"] == "Y").astype(int) * 2 + (s["Hot Technology"] == "Y").astype(int)
        s = (
            s.sort_values("prio", ascending=False)
            .drop_duplicates("Workplace Example")
            .head(top_software)
        )
        for rank, (_, r) in enumerate(s.iterrows(), start=1):
            # Importance proxy on the same 1-5 scale as essential skills.
            imp = 3.0 + 0.5 * int(r["prio"])
            rows.append(
                (role, soc, r["Workplace Example"], "Software Skill", r["Element Name"], imp, rank)
            )

    return pd.DataFrame(
        rows,
        columns=[
            "JobRole", "ONET_SOC_Code", "SkillName", "SkillType",
            "SkillCategory", "Importance", "RankInRole",
        ],
    )


def build_employee_skills(emp: pd.DataFrame, req: pd.DataFrame) -> pd.DataFrame:
    """Assign each employee a subset of their role's required skills.

    Propensity to hold a skill rises with education, tenure, seniority and training,
    and falls as the skill gets less central to the role. Seeded, so reruns match.
    """
    rng = np.random.default_rng(SEED)
    by_role = {r: g.sort_values("RankInRole") for r, g in req.groupby("JobRole")}
    all_skills = req[["SkillName", "SkillType"]].drop_duplicates()

    records = []
    for _, e in emp.iterrows():
        # 0..1 capability signal built from real attributes.
        propensity = (
            0.30 * (e["Education"] - 1) / 4.0
            + 0.30 * min(e["TotalWorkingYears"], 25) / 25.0
            + 0.25 * (e["JobLevel"] - 1) / 4.0
            + 0.15 * min(e["TrainingTimesLastYear"], 6) / 6.0
        )
        role_req = by_role[e["JobRole"]]
        held = []
        for i, (_, s) in enumerate(role_req.iterrows()):
            # Central skills (low rank) are held more often than peripheral ones.
            centrality = 1.0 - 0.045 * i
            p = float(np.clip(0.25 + 0.65 * propensity * centrality, 0.05, 0.95))
            if rng.random() < p:
                held.append((s["SkillName"], s["SkillType"], "Role"))

        # A few transferable skills picked up outside the current role.
        n_extra = int(rng.integers(0, 3))
        if n_extra:
            pool = all_skills[~all_skills["SkillName"].isin(role_req["SkillName"])]
            for idx in rng.choice(len(pool), size=min(n_extra, len(pool)), replace=False):
                row = pool.iloc[int(idx)]
                held.append((row["SkillName"], row["SkillType"], "Transferable"))

        for name, stype, source in held:
            # Proficiency 1-5, anchored on the same propensity.
            prof = int(np.clip(round(2 + 3 * propensity + rng.normal(0, 0.6)), 1, 5))
            records.append((e["EmployeeID"], name, stype, source, prof))

    return (
        pd.DataFrame(
            records,
            columns=["EmployeeID", "SkillName", "SkillType", "AcquiredVia", "ProficiencyLevel"],
        )
        .drop_duplicates(subset=["EmployeeID", "SkillName"])
        .sort_values(["EmployeeID", "SkillName"])
        .reset_index(drop=True)
    )


def main() -> None:
    emp = load_attrition()

    eng = build_engagement(emp)
    eng.to_csv(RAW / "hr_performance_engagement.csv", index=False)
    print(f"hr_performance_engagement.csv  {eng.shape[0]:>6,} rows x {eng.shape[1]} cols")

    req = role_requirements()
    req.to_csv(RAW / "role_skill_requirements.csv", index=False)
    print(
        f"role_skill_requirements.csv    {req.shape[0]:>6,} rows x {req.shape[1]} cols "
        f"({req.JobRole.nunique()} roles, {req.SkillName.nunique()} distinct skills)"
    )

    skills = build_employee_skills(emp, req)
    skills.to_csv(RAW / "employee_current_skills.csv", index=False)
    print(
        f"employee_current_skills.csv    {skills.shape[0]:>6,} rows x {skills.shape[1]} cols "
        f"({skills.EmployeeID.nunique():,} employees, "
        f"avg {skills.groupby('EmployeeID').size().mean():.1f} skills each)"
    )

    DOCS.mkdir(exist_ok=True)
    pd.DataFrame(
        [(r, soc, t) for r, (soc, t) in ROLE_TO_SOC.items()],
        columns=["JobRole", "ONET_SOC_Code", "ONET_Title"],
    ).to_csv(DOCS / "role_to_onet_mapping.csv", index=False)
    print("docs/role_to_onet_mapping.csv  written")


if __name__ == "__main__":
    main()
