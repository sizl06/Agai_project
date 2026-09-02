"""Shared fixtures.

Session-scoped where loading is expensive, so the whole suite loads the model and
the processed data once rather than per test.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning)

from app.utils.config import PROCESSED_FILES, latest_model_version  # noqa: E402

# Skip anything needing a trained model or built tables, with a message that says
# how to produce them, rather than failing with an opaque FileNotFoundError.
requires_model = pytest.mark.skipif(
    latest_model_version() is None,
    reason="no trained model - run: python -m pipelines.train_model",
)
requires_intelligence = pytest.mark.skipif(
    not PROCESSED_FILES["intelligence"].exists(),
    reason="no intelligence table - run: python -m pipelines.build_intelligence",
)


@pytest.fixture(scope="session")
def employees() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_FILES["attrition"])


@pytest.fixture(scope="session")
def engagement() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_FILES["engagement"])


@pytest.fixture(scope="session")
def requirements() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_FILES["role_requirements"])


@pytest.fixture(scope="session")
def employee_skills() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_FILES["employee_skills"])


@pytest.fixture(scope="session")
def api_client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def valid_payload() -> dict:
    """A well-formed prediction request."""
    return {
        "EmployeeID": 90001,
        "Age": 34,
        "Department": "Sales",
        "DistanceFromHome": 8,
        "JobRole": "Sales Executive",
        "JobSatisfaction": 3,
        "MonthlyIncome": 5800,
        "OverTime": "No",
        "TotalWorkingYears": 10,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
    }
