# %% [markdown]
# # 01 - Data Understanding
#
# Day 1, step 1. No modelling here. The goal is to load every dataset and answer the
# same five questions for each: how big is it, what is in it, what is missing, what is
# the primary key, and what can it join to.
#
# Rule from the build notes: **do not merge anything yet**, even where two files look
# like they share an ID. A matching column name is not proof of a matching key - that
# gets confirmed in notebook 04.

# %%
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from app.utils.config import RAW_FILES

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("Raw files present:")
for name, path in RAW_FILES.items():
    status = f"{path.stat().st_size / 1_000_000:.2f} MB" if path.exists() else "MISSING"
    print(f"  {name:20s} {path.name:35s} {status}")

# %% [markdown]
# ## Load everything
#
# `employee_attrition.csv` ships with a UTF-8 BOM, so it needs `encoding='utf-8-sig'`
# or the first column name comes back as `﻿Age`.

# %%
datasets = {}
datasets["attrition"] = pd.read_csv(RAW_FILES["attrition"], encoding="utf-8-sig")
datasets["attrition"].columns = [c.strip() for c in datasets["attrition"].columns]
for key in ["engagement", "occupation", "essential_skills", "software_skills",
            "role_requirements", "employee_skills"]:
    datasets[key] = pd.read_csv(RAW_FILES[key])

for name, df in datasets.items():
    print(f"{name:20s} {df.shape[0]:>7,} rows x {df.shape[1]:>2} cols")

# %% [markdown]
# ## The standard checklist, applied to every dataset
#
# Shape, dtypes, missingness, duplicates, and candidate identifier columns.

# %%
def profile(name: str, df: pd.DataFrame) -> dict:
    id_cols = [c for c in df.columns if "id" in c.lower() or "code" in c.lower()]
    unique_ids = [c for c in id_cols if df[c].is_unique]
    return {
        "dataset": name,
        "rows": len(df),
        "cols": df.shape[1],
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": int(df.select_dtypes("number").shape[1]),
        "object_cols": int(df.select_dtypes("object").shape[1]),
        "id_candidates": ", ".join(id_cols[:3]) or "-",
        "unique_key": ", ".join(unique_ids[:2]) or "(composite or none)",
    }


summary = pd.DataFrame([profile(n, d) for n, d in datasets.items()])
summary

# %% [markdown]
# Nothing is missing anywhere, and no dataset has a fully duplicated row. That is
# unusual for HR data and worth stating explicitly rather than assuming.

# %% [markdown]
# ## The attrition dataset in detail
#
# This is the one that feeds the ML model, so it gets the closest look.

# %%
emp = datasets["attrition"]
print("Shape:", emp.shape)
emp.head()

# %%
emp.info()

# %% [markdown]
# ### Target balance
#
# This number decides how the model is evaluated later.

# %%
balance = emp["Attrition"].value_counts()
balance_pct = (emp["Attrition"].value_counts(normalize=True) * 100).round(2)
print(pd.DataFrame({"count": balance, "percent": balance_pct}))
print(f"\nImbalance ratio: {balance['No'] / balance['Yes']:.2f} stayers per leaver")

# %% [markdown]
# **16% attrition.** A model that predicts "No" for everyone scores 84% accuracy while
# being completely useless. Accuracy is therefore off the table as an evaluation
# metric from here on - notebooks 06 and 07 use precision, recall, F1 and ROC-AUC.

# %% [markdown]
# ### Columns that carry no information

# %%
constant_cols = [c for c in emp.columns if emp[c].nunique() == 1]
print("Zero-variance columns:", constant_cols)
for c in constant_cols:
    print(f"  {c:20s} always = {emp[c].iloc[0]!r}")

# %% [markdown]
# These three get dropped in cleaning. They cannot help a model and they add
# meaningless columns to the one-hot encoder.

# %% [markdown]
# ### Identifier check

# %%
print("Columns containing 'id':", [c for c in emp.columns if "id" in c.lower()])
print("EmployeeNumber unique:", emp["EmployeeNumber"].is_unique)
print("EmployeeNumber range :", emp["EmployeeNumber"].min(), "-", emp["EmployeeNumber"].max())

# %% [markdown]
# **Finding that contradicts the plan.** The build notes assume the key is called
# `EmployeeID`. The real IBM extract calls it `EmployeeNumber`, and there is no
# `EmployeeID` column at all. Cleaning renames it, so every downstream notebook can
# use the name the notes expect.
#
# Note also that `EmployeeNumber` runs to 2068 across only 1,470 rows - the IDs are
# sparse, not a contiguous 1..N range. Anything that assumes `EmployeeID == row index`
# would be wrong.

# %% [markdown]
# ### Distribution of the key numeric drivers

# %%
numeric_focus = ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears",
                 "DistanceFromHome", "YearsSinceLastPromotion"]
emp[numeric_focus].describe().T.round(2)

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, col in zip(axes.ravel(), numeric_focus):
    ax.hist(emp[col], bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title(col)
fig.suptitle("Distribution of key numeric features", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# `MonthlyIncome`, `YearsAtCompany` and `TotalWorkingYears` are strongly right-skewed.
# Those long tails are real senior employees, not data errors - which matters when
# deciding what to do about "outliers" in notebook 03.

# %% [markdown]
# ### Attrition rate by category
#
# A first look at which groups actually leave, before any modelling.

# %%
for col in ["Department", "JobRole", "OverTime", "BusinessTravel", "MaritalStatus"]:
    rate = (emp.groupby(col)["Attrition"].apply(lambda s: (s == "Yes").mean() * 100)
            .sort_values(ascending=False).round(1))
    print(f"\n--- Attrition rate (%) by {col} ---")
    print(rate.to_string())

# %% [markdown]
# `OverTime` stands out immediately: employees working overtime leave at roughly three
# times the rate of those who do not. Sales Representatives and Laboratory Technicians
# are the highest-risk roles. Both should show up in the SHAP analysis in notebook 08 -
# if they do not, something is wrong with the model.

# %% [markdown]
# ## The engagement dataset

# %%
eng = datasets["engagement"]
print(eng.shape)
eng.head()

# %%
eng[["EngagementScore", "JobSatisfactionScore", "PerformanceRating"]].describe().T.round(2)

# %% [markdown]
# > **Provenance.** This file is derived from the IBM satisfaction/involvement columns
# > for the same 1,470 employees, because the Kaggle original needs API credentials.
# > See `scripts/build_derived_sources.py`. It is therefore a deterministic function of
# > columns the attrition model also uses, which is exactly why it is kept for
# > analytics only and never fed back in as a model feature.

# %% [markdown]
# ## The O*NET reference datasets

# %%
occ = datasets["occupation"]
print("occupation_data:", occ.shape)
print("SOC code unique:", occ["O*NET-SOC Code"].is_unique)
occ.head(3)

# %%
ess = datasets["essential_skills"]
print("essential_skills:", ess.shape)
print("Scale IDs present:", ess["Scale ID"].unique(), " <- IM = Importance, LV = Level")
print("Rows flagged Recommend Suppress = Y:", int((ess["Recommend Suppress"] == "Y").sum()))
ess.head(3)

# %% [markdown]
# Two things to handle in cleaning: keep only the `IM` (Importance) rows, since Level
# answers a different question, and drop the rows O*NET itself recommends suppressing
# as statistically unreliable.

# %%
soft = datasets["software_skills"]
print("software_skills:", soft.shape)
print("distinct tools (Workplace Example):", soft["Workplace Example"].nunique())
print("distinct categories (Element Name):", soft["Element Name"].nunique())
soft.head(3)

# %% [markdown]
# ### Messy skill names - the problem the notes call out

# %%
messy = sorted(soft[soft["Workplace Example"].str.contains(
    "Amazon Web|Micosoft|SAP|Salesforce", case=False, na=False)]["Workplace Example"].unique())
for m in messy[:12]:
    print(" ", m)

# %% [markdown]
# `Micosoft SQL Server Analysis Services SSAS` is a genuine typo in the upstream O*NET
# data, and `... software` suffixes are applied inconsistently. Left alone, these split
# one real skill across several names and the gap engine over-counts. Handled in
# notebook 03 by `app.services.skill_normalizer`.

# %% [markdown]
# ## What each dataset is for
#
# | Dataset | Purpose | Joins to |
# |---|---|---|
# | `employee_attrition` | feeds the attrition ML model | engagement (EmployeeID), roles (JobRole) |
# | `hr_performance_engagement` | engagement analytics, no ML | attrition (EmployeeID) |
# | `occupation_data` | role master reference | role requirements (SOC code) |
# | `essential_skills` | required soft skills per role | occupation (SOC code) |
# | `software_skills` | required tools per role | occupation (SOC code) |
# | `role_skill_requirements` | the two skill files, resolved per JobRole | employees (JobRole) |
# | `employee_current_skills` | what each employee already has | attrition (EmployeeID) |
#
# Carried into notebook 02: the key is `EmployeeNumber` not `EmployeeID`, three columns
# are constant, the target is 16% positive, and skill names need canonicalising.
