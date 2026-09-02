"""Employee-level contracts: the batch dataframe rules and the API request model."""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.utils.config import CATEGORICAL_DOMAINS, ID_COL, TARGET_COL, VALIDATION_RULES
from app.validation.schema_checks import (
    ValidationReport,
    check_categories,
    check_no_duplicate_rows,
    check_no_nulls,
    check_numeric,
    check_ranges,
    check_required_columns,
    check_unique,
)

REQUIRED_COLUMNS = [
    ID_COL,
    "Age",
    "Department",
    "JobRole",
    "MonthlyIncome",
    "OverTime",
    "JobSatisfaction",
    "YearsAtCompany",
    "TotalWorkingYears",
    "WorkLifeBalance",
]

NUMERIC_COLUMNS = [
    "Age",
    "MonthlyIncome",
    "DistanceFromHome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsSinceLastPromotion",
    "JobSatisfaction",
    "JobLevel",
]


def validate_employee_frame(df: pd.DataFrame, *, require_target: bool = True) -> ValidationReport:
    """Validate an employee table. `require_target` is False for inference-time data."""
    report = ValidationReport(dataset="employee_attrition", n_rows=len(df))

    required = list(REQUIRED_COLUMNS)
    if require_target:
        required.append(TARGET_COL)

    check_required_columns(df, required, report)
    check_numeric(df, NUMERIC_COLUMNS, report)
    check_ranges(df, VALIDATION_RULES, report)
    check_unique(df, ID_COL, report)
    check_no_nulls(df, [ID_COL, "Age", "Department", "JobRole"], report)
    check_no_duplicate_rows(df, report)

    domains = dict(CATEGORICAL_DOMAINS)
    if not require_target:
        domains.pop(TARGET_COL, None)
    check_categories(df, domains, report)

    return report


class EmployeeFeatures(BaseModel):
    """One employee, as accepted by POST /predict/attrition.

    Pydantic rejects out-of-range input with a 422 before anything reaches the
    model, so a nonsense payload can never produce a confident-looking prediction.
    """

    model_config = ConfigDict(extra="forbid")

    EmployeeID: int = Field(..., ge=1, description="Unique employee identifier")
    Age: int = Field(..., ge=18, le=100)
    BusinessTravel: Literal["Travel_Rarely", "Travel_Frequently", "Non-Travel"] = "Travel_Rarely"
    DailyRate: int = Field(800, ge=0, le=10_000)
    Department: Literal["Sales", "Research & Development", "Human Resources"]
    DistanceFromHome: int = Field(..., ge=0, le=200)
    Education: int = Field(3, ge=1, le=5)
    EducationField: str = "Life Sciences"
    EnvironmentSatisfaction: int = Field(3, ge=1, le=4)
    Gender: Literal["Male", "Female"] = "Male"
    HourlyRate: int = Field(65, ge=0, le=1_000)
    JobInvolvement: int = Field(3, ge=1, le=4)
    JobLevel: int = Field(2, ge=1, le=5)
    JobRole: str
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MaritalStatus: Literal["Single", "Married", "Divorced"] = "Married"
    MonthlyIncome: int = Field(..., ge=0, le=1_000_000)
    MonthlyRate: int = Field(14_000, ge=0, le=100_000)
    NumCompaniesWorked: int = Field(2, ge=0, le=20)
    OverTime: Literal["Yes", "No"]
    PercentSalaryHike: int = Field(14, ge=0, le=100)
    PerformanceRating: int = Field(3, ge=1, le=4)
    RelationshipSatisfaction: int = Field(3, ge=1, le=4)
    StockOptionLevel: int = Field(1, ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0, le=60)
    TrainingTimesLastYear: int = Field(3, ge=0, le=10)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0, le=60)
    YearsInCurrentRole: int = Field(3, ge=0, le=60)
    YearsSinceLastPromotion: int = Field(1, ge=0, le=60)
    YearsWithCurrManager: int = Field(3, ge=0, le=60)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])


class AttritionPrediction(BaseModel):
    """Response for a single attrition prediction."""

    employee_id: int
    attrition_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]
    model_version: str
    top_factors: list[dict] = Field(default_factory=list)
