"""Upskilling recommendation engine (step 15).

Two layers, in the order the build notes prescribe.

**v1 - explicit tags.** Every course in the catalogue declares the skills it covers,
so a missing skill maps to a course by lookup. Deliberately dumb, completely
predictable, and correct whenever the catalogue is tagged. This is what serves the
recommendations.

**v2 - semantic fallback.** For a missing skill with no tagged course, the engine
compares the skill against course titles and descriptions using TF-IDF cosine
similarity over word and character n-grams.

> **Honest limitation.** The build notes describe v2 as sentence-transformer
> embeddings, so that "MLOps" matches a course called "Deploying and Monitoring
> Machine Learning Systems" despite sharing no words. TF-IDF **cannot** do that - it
> matches surface forms, so it succeeds on "Excel" vs "Advanced Spreadsheet
> Modelling" only through partial character overlap and fails outright on true
> synonyms. Notebook 15 demonstrates the failure rather than hiding it. Swapping in
> sentence-transformers means replacing `_semantic_scores` alone; the interface and
> everything downstream stay as they are.

Prioritisation combines three signals, because an employee with eight gaps needs to
know which two to start with:

    priority = skill importance to the role
             x organisational severity of that gap
             x the employee's own attrition risk
"""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.config import EXTERNAL_DIR, ID_COL, PROCESSED_FILES

CATALOGUE_PATH = EXTERNAL_DIR / "course_catalogue.csv"

# An organisation-wide shortage is worth more training budget than an isolated one.
SEVERITY_WEIGHT = {"HIGH": 1.5, "MEDIUM": 1.2, "LOW": 1.0}

# A flight risk who is also under-skilled is the most urgent case.
RISK_WEIGHT = {"HIGH": 1.4, "MEDIUM": 1.15, "LOW": 1.0}

MIN_SEMANTIC_SIMILARITY = 0.20


def load_catalogue() -> pd.DataFrame:
    catalogue = pd.read_csv(CATALOGUE_PATH)
    catalogue["SkillList"] = catalogue["CoversSkills"].fillna("").str.split("|")
    return catalogue


def build_skill_index(catalogue: pd.DataFrame) -> dict[str, list[dict]]:
    """{skill: [course, ...]} from the explicit tags - the v1 lookup."""
    index: dict[str, list[dict]] = {}
    for _, course in catalogue.iterrows():
        for skill in course["SkillList"]:
            skill = skill.strip()
            if skill:
                index.setdefault(skill, []).append({
                    "CourseID": course["CourseID"],
                    "Title": course["Title"],
                    "Provider": course["Provider"],
                    "Level": course["Level"],
                    "DurationHours": int(course["DurationHours"]),
                })
    return index


def _semantic_scores(skill: str, catalogue: pd.DataFrame) -> pd.Series:
    """Cosine similarity between a skill name and each course's text.

    Word n-grams catch shared terminology; character n-grams catch partial matches
    such as "Excel" inside "Spreadsheet". Neither catches genuine synonyms - see the
    module docstring.
    """
    corpus = (catalogue["Title"] + " " + catalogue["Description"] + " "
              + catalogue["CoversSkills"].str.replace("|", " ", regex=False)).tolist()

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus + [skill])
    return pd.Series(cosine_similarity(matrix[-1], matrix[:-1]).ravel(), index=catalogue.index)


def recommend_for_skill(skill: str, catalogue: pd.DataFrame,
                        index: dict[str, list[dict]] | None = None) -> dict | None:
    """Best course for one missing skill: tag lookup first, semantic fallback second."""
    index = build_skill_index(catalogue) if index is None else index

    if skill in index:
        course = min(index[skill], key=lambda c: c["DurationHours"])
        return {**course, "MatchType": "tag", "MatchScore": 1.0}

    scores = _semantic_scores(skill, catalogue)
    best = scores.idxmax()
    if scores[best] < MIN_SEMANTIC_SIMILARITY:
        return None

    course = catalogue.loc[best]
    return {
        "CourseID": course["CourseID"],
        "Title": course["Title"],
        "Provider": course["Provider"],
        "Level": course["Level"],
        "DurationHours": int(course["DurationHours"]),
        "MatchType": "semantic",
        "MatchScore": round(float(scores[best]), 3),
    }


def build_recommendations(
    gaps: pd.DataFrame,
    org_gaps: pd.DataFrame,
    risk_by_employee: dict[int, str] | None = None,
    per_employee: int = 3,
) -> pd.DataFrame:
    """One row per recommended course per employee, ranked by priority."""
    if gaps.empty:
        return pd.DataFrame(columns=[ID_COL, "MissingSkill", "Recommendation", "Priority"])

    catalogue = load_catalogue()
    index = build_skill_index(catalogue)
    severity = dict(zip(org_gaps["MissingSkill"], org_gaps["Severity"]))
    risk_by_employee = risk_by_employee or {}

    # Resolve each distinct skill once rather than once per employee - the semantic
    # fallback refits a vectoriser per call and is far too slow to run 12,000 times.
    resolved = {s: recommend_for_skill(s, catalogue, index) for s in gaps["MissingSkill"].unique()}

    records = []
    for _, gap in gaps.iterrows():
        course = resolved.get(gap["MissingSkill"])
        if course is None:
            continue

        employee_id = int(gap[ID_COL])
        gap_severity = severity.get(gap["MissingSkill"], "LOW")
        risk = risk_by_employee.get(employee_id, "LOW")
        priority = (float(gap["Importance"])
                    * SEVERITY_WEIGHT.get(gap_severity, 1.0)
                    * RISK_WEIGHT.get(risk, 1.0))

        records.append({
            ID_COL: employee_id,
            "Department": gap["Department"],
            "JobRole": gap["JobRole"],
            "MissingSkill": gap["MissingSkill"],
            "Recommendation": f"Learn {gap['MissingSkill']}",
            "CourseID": course["CourseID"],
            "CourseTitle": course["Title"],
            "Provider": course["Provider"],
            "Level": course["Level"],
            "DurationHours": course["DurationHours"],
            "MatchType": course["MatchType"],
            "SkillImportance": round(float(gap["Importance"]), 2),
            "OrgSeverity": gap_severity,
            "AttritionRisk": risk,
            "Priority": round(priority, 3),
        })

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out = out.sort_values([ID_COL, "Priority"], ascending=[True, False])
    out["Rank"] = out.groupby(ID_COL).cumcount() + 1
    return out[out["Rank"] <= per_employee].reset_index(drop=True)


def top_recommendation_per_employee(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per employee for the intelligence table."""
    if recommendations.empty:
        return pd.DataFrame(columns=[ID_COL, "Recommendation", "CourseTitle"])
    top = recommendations[recommendations["Rank"] == 1]
    return top[[ID_COL, "MissingSkill", "Recommendation", "CourseTitle",
                "Provider", "DurationHours", "Priority"]].reset_index(drop=True)


def organisation_training_plan(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate demand per course - what to actually buy seats for."""
    if recommendations.empty:
        return pd.DataFrame(columns=["CourseTitle", "EmployeesNeeding"])
    return (
        recommendations.groupby(["CourseID", "CourseTitle", "Provider", "DurationHours"])
        .agg(EmployeesNeeding=(ID_COL, "nunique"),
             AvgPriority=("Priority", "mean"),
             SkillsCovered=("MissingSkill", lambda s: ", ".join(sorted(set(s)))))
        .reset_index()
        .assign(TotalTrainingHours=lambda d: d["EmployeesNeeding"] * d["DurationHours"],
                AvgPriority=lambda d: d["AvgPriority"].round(3))
        .sort_values("EmployeesNeeding", ascending=False)
        .reset_index(drop=True)
    )


def save(recommendations: pd.DataFrame) -> None:
    recommendations.to_csv(PROCESSED_FILES["recommendations"], index=False)
