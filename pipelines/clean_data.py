"""Day 1, step 03 - cleaning.

Reads data/raw/, writes data/processed/. The raw files are never modified.

What gets fixed, and why:
  * EmployeeNumber -> EmployeeID          the build notes assume EmployeeID; the real
                                          IBM extract ships EmployeeNumber.
  * EmployeeCount / Over18 / StandardHours dropped - zero variance, so they carry no
                                          signal and only widen the encoder.
  * Attrition                             kept as Yes/No for reporting, plus a 0/1
                                          AttritionFlag for modelling.
  * skill names                            canonicalised via skill_normalizer, then
                                          de-duplicated, because normalisation can
                                          merge two raw rows into one.
  * outliers                               reported, NOT removed. See note in
                                          report_outliers().
"""
from __future__ import annotations

import pandas as pd

from app.services.skill_normalizer import normalize_series
from app.utils.config import CONSTANT_COLS, ID_COL, PROCESSED_FILES, RAW_FILES, TARGET_COL
from app.utils.logger import get_logger

log = get_logger("clean")


def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def clean_attrition() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES["attrition"], encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"EmployeeNumber": ID_COL})

    before = df.shape
    dropped = [c for c in CONSTANT_COLS if c in df.columns]
    df = df.drop(columns=dropped)
    df = df.drop_duplicates()
    df = _strip_object_columns(df)

    # Numeric 0/1 target for modelling; the Yes/No column stays for reporting.
    df["AttritionFlag"] = (df[TARGET_COL] == "Yes").astype(int)

    df[ID_COL] = df[ID_COL].astype(int)
    log.info(
        "attrition cleaned: %s -> %s (dropped constant columns %s)", before, df.shape, dropped
    )
    return df


def clean_engagement() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES["engagement"])
    df = _strip_object_columns(df).drop_duplicates()
    df[ID_COL] = df[ID_COL].astype(int)

    score_cols = [c for c in df.columns if c.endswith("Score")]
    df[score_cols] = df[score_cols].round(1)
    log.info("engagement cleaned: %s", df.shape)
    return df


def clean_occupation() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES["occupation"])
    df = df.rename(columns={
        "O*NET-SOC Code": "ONET_SOC_Code",
        "Title": "ONET_Title",
        "Description": "ONET_Description",
    })
    df = _strip_object_columns(df).drop_duplicates(subset=["ONET_SOC_Code"])
    log.info("occupation master cleaned: %s", df.shape)
    return df


def clean_essential_skills() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES["essential_skills"])
    # Importance ratings only; Level ratings answer a different question, and rows
    # O*NET recommends suppressing are statistically unreliable.
    df = df[(df["Scale ID"] == "IM") & (df["Recommend Suppress"] != "Y")]
    df = df.rename(columns={
        "O*NET-SOC Code": "ONET_SOC_Code",
        "Title": "ONET_Title",
        "Element Name": "SkillName",
        "Data Value": "Importance",
    })[["ONET_SOC_Code", "ONET_Title", "SkillName", "Importance"]]
    df["SkillName"] = normalize_series(df["SkillName"])
    df = df.drop_duplicates(subset=["ONET_SOC_Code", "SkillName"])
    log.info("essential skills cleaned: %s", df.shape)
    return df


def clean_software_skills() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES["software_skills"])
    df = df.rename(columns={
        "O*NET-SOC Code": "ONET_SOC_Code",
        "Title": "ONET_Title",
        "Workplace Example": "SkillName",
        "Element Name": "SkillCategory",
        "Hot Technology": "HotTechnology",
        "In Demand": "InDemand",
    })[["ONET_SOC_Code", "ONET_Title", "SkillName", "SkillCategory", "HotTechnology", "InDemand"]]
    df["SkillName"] = normalize_series(df["SkillName"])
    df = df.drop_duplicates(subset=["ONET_SOC_Code", "SkillName"])
    log.info("software skills cleaned: %s", df.shape)
    return df


def clean_role_requirements() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES["role_requirements"])
    df["SkillName"] = normalize_series(df["SkillName"])
    before = len(df)
    df = df.drop_duplicates(subset=["JobRole", "SkillName"])
    if len(df) < before:
        log.info("role requirements: %d row(s) merged by name normalisation", before - len(df))
    log.info("role requirements cleaned: %s", df.shape)
    return df


def clean_employee_skills() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES["employee_skills"])
    df["SkillName"] = normalize_series(df["SkillName"])
    before = len(df)
    # Normalisation can collapse two spellings held by the same employee; keep the
    # higher proficiency rather than an arbitrary one.
    df = (
        df.sort_values("ProficiencyLevel", ascending=False)
        .drop_duplicates(subset=[ID_COL, "SkillName"])
        .sort_values([ID_COL, "SkillName"])
        .reset_index(drop=True)
    )
    if len(df) < before:
        log.info("employee skills: %d duplicate pair(s) merged after normalisation", before - len(df))
    df[ID_COL] = df[ID_COL].astype(int)
    log.info("employee skills cleaned: %s", df.shape)
    return df


def report_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Flag IQR outliers without changing them.

    MonthlyIncome and TotalWorkingYears are genuinely right-skewed - a director on
    ~20k is a real employee, not a data error. Clipping them would erase exactly the
    senior population attrition cost is highest for, so this reports and moves on.
    """
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({
            "column": col,
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "lower_fence": round(low, 2),
            "upper_fence": round(high, 2),
            "n_outliers": n,
            "pct": round(100 * n / len(df), 2),
        })
    return pd.DataFrame(rows)


def main() -> dict[str, pd.DataFrame]:
    cleaned = {
        "attrition": clean_attrition(),
        "engagement": clean_engagement(),
        "occupation": clean_occupation(),
        "essential_skills": clean_essential_skills(),
        "software_skills": clean_software_skills(),
        "role_requirements": clean_role_requirements(),
        "employee_skills": clean_employee_skills(),
    }
    for key, df in cleaned.items():
        df.to_csv(PROCESSED_FILES[key], index=False)
        print(f"{PROCESSED_FILES[key].name:45s} {df.shape[0]:>7,} rows x {df.shape[1]:>2} cols")
    return cleaned


if __name__ == "__main__":
    main()
