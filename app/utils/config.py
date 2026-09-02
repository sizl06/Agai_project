"""Central configuration: paths, column contracts, and business thresholds.

Everything that another module might otherwise hard-code lives here, so that a
threshold or a filename is changed in exactly one place.
"""
from __future__ import annotations

import json
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

for _d in (PROCESSED_DIR, EXTERNAL_DIR, PREDICTIONS_DIR, MODELS_DIR, DOCS_DIR, REPORTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- files
RAW_FILES = {
    "attrition": RAW_DIR / "employee_attrition.csv",
    "engagement": RAW_DIR / "hr_performance_engagement.csv",
    "occupation": RAW_DIR / "occupation_data.csv",
    "essential_skills": RAW_DIR / "essential_skills.csv",
    "software_skills": RAW_DIR / "software_skills.csv",
    "role_requirements": RAW_DIR / "role_skill_requirements.csv",
    "employee_skills": RAW_DIR / "employee_current_skills.csv",
}

PROCESSED_FILES = {
    "attrition": PROCESSED_DIR / "employee_attrition_processed.csv",
    "engagement": PROCESSED_DIR / "engagement_processed.csv",
    "occupation": PROCESSED_DIR / "occupation_master.csv",
    "essential_skills": PROCESSED_DIR / "essential_skills_processed.csv",
    "software_skills": PROCESSED_DIR / "software_skills_processed.csv",
    "role_requirements": PROCESSED_DIR / "role_skill_requirements_processed.csv",
    "employee_skills": PROCESSED_DIR / "employee_skills_processed.csv",
    "features": PROCESSED_DIR / "attrition_features.csv",
    "skill_gaps": PROCESSED_DIR / "employee_skill_gaps.csv",
    "org_skill_gaps": PROCESSED_DIR / "organization_skill_gaps.csv",
    "recommendations": PROCESSED_DIR / "recommendations.csv",
    "intelligence": PROCESSED_DIR / "employee_intelligence.csv",
}

# --------------------------------------------------------------------------- columns
ID_COL = "EmployeeID"
TARGET_COL = "Attrition"

# Zero-variance in the IBM extract: they carry no signal and only add noise to encoders.
CONSTANT_COLS = ["EmployeeCount", "Over18", "StandardHours"]

# Engagement is a deterministic function of these five Likert items. Feeding it back
# into the attrition model would be circular, so engagement stays analytics-only and
# these columns are the model's view of satisfaction instead.
ENGAGEMENT_SOURCE_COLS = [
    "JobSatisfaction",
    "JobInvolvement",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "RelationshipSatisfaction",
]

# --------------------------------------------------------------------------- validation rules
VALIDATION_RULES = {
    "Age": (18, 100),
    "EngagementScore": (0.0, 100.0),
    "MonthlyIncome": (0, 1_000_000),
    "DistanceFromHome": (0, 200),
    "TotalWorkingYears": (0, 60),
    "YearsAtCompany": (0, 60),
    "PercentSalaryHike": (0, 100),
    "ProficiencyLevel": (1, 5),
}

CATEGORICAL_DOMAINS = {
    "Attrition": {"Yes", "No"},
    "OverTime": {"Yes", "No"},
    "Gender": {"Male", "Female"},
    "BusinessTravel": {"Travel_Rarely", "Travel_Frequently", "Non-Travel"},
}

# --------------------------------------------------------------------------- business thresholds
# Attrition risk bands, expressed as multiples of the model's own tuned decision
# threshold rather than as fixed probabilities.
#
# Fixed cuts would be wrong here. The models are trained with class weighting, so their
# output is calibrated to a balanced world, not to the true ~16% base rate - a literal
# 0.50 means "more likely than not" only under that reweighting. Tying the bands to the
# threshold the model was actually tuned at keeps HIGH meaning "at or past the point
# this model calls a leaver", and keeps the bands correct if a future model is
# calibrated differently.
RISK_BAND_MULTIPLIERS = {"HIGH": 1.0, "MEDIUM": 0.6}
DEFAULT_DECISION_THRESHOLD = 0.5

# Organisation-wide skill-gap severity, in absolute headcount missing a skill,
# exactly as specified in the build notes.
SEVERITY_BANDS = {"HIGH": 100, "MEDIUM": 50}

# The absolute bands above were written with a ~2,500-employee organisation in mind.
# At 1,470 employees they stop discriminating: 36 of 52 skills land in HIGH, which
# tells a training manager nothing about where to start. These proportional bands are
# the alternative, kept alongside rather than replacing the spec - see notebook 14.
SEVERITY_BANDS_RELATIVE = {"HIGH": 0.40, "MEDIUM": 0.20}

RANDOM_STATE = 42
TEST_SIZE = 0.2


def risk_level(probability: float, threshold: float | None = None) -> str:
    """Map an attrition probability onto a risk band.

    `threshold` is the model's tuned decision threshold, read from its metadata.
    """
    threshold = threshold if threshold is not None else DEFAULT_DECISION_THRESHOLD
    if probability >= threshold * RISK_BAND_MULTIPLIERS["HIGH"]:
        return "HIGH"
    if probability >= threshold * RISK_BAND_MULTIPLIERS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def gap_severity(employees_missing: int) -> str:
    """Map a headcount missing a skill onto a severity band, per the build notes."""
    if employees_missing >= SEVERITY_BANDS["HIGH"]:
        return "HIGH"
    if employees_missing >= SEVERITY_BANDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def gap_severity_relative(employees_missing: int, total_employees: int) -> str:
    """Severity as a share of the workforce, for organisations of any size."""
    if total_employees <= 0:
        return "LOW"
    share = employees_missing / total_employees
    if share >= SEVERITY_BANDS_RELATIVE["HIGH"]:
        return "HIGH"
    if share >= SEVERITY_BANDS_RELATIVE["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------- model registry
def available_model_versions() -> list[str]:
    """Version folders under models/, newest last (v1, v2, ... v10 sorts correctly)."""
    if not MODELS_DIR.exists():
        return []
    versions = [p.name for p in MODELS_DIR.iterdir() if p.is_dir() and p.name.startswith("v")]
    return sorted(versions, key=lambda v: int(v.lstrip("v")) if v.lstrip("v").isdigit() else 0)


def latest_model_version() -> str | None:
    versions = available_model_versions()
    return versions[-1] if versions else None


def model_path(version: str | None = None) -> Path:
    version = version or latest_model_version()
    if version is None:
        raise FileNotFoundError(
            "No trained model found under models/. Run pipelines/train_model.py first."
        )
    return MODELS_DIR / version / "attrition_pipeline.joblib"


def model_metadata(version: str | None = None) -> dict:
    version = version or latest_model_version()
    if version is None:
        return {}
    meta_file = MODELS_DIR / version / "metadata.json"
    if not meta_file.exists():
        return {}
    return json.loads(meta_file.read_text(encoding="utf-8"))
