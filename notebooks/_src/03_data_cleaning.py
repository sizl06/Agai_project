# %% [markdown]
# # 03 - Data Cleaning
#
# Day 1, step 3. Notebook 01 found the problems; this fixes them and writes a clean
# copy of every dataset to `data/processed/`. The raw files are never overwritten.
#
# The transformations live in `pipelines/clean_data.py` so the API and the batch jobs
# clean data identically. This notebook runs that pipeline and inspects what it did.

# %%
import pandas as pd

from app.services.skill_normalizer import normalization_audit, normalize_skill
from app.utils.config import PROCESSED_FILES, RAW_FILES
from pipelines import clean_data

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

raw_emp = pd.read_csv(RAW_FILES["attrition"], encoding="utf-8-sig")
raw_emp.columns = [c.strip() for c in raw_emp.columns]
print("raw attrition shape:", raw_emp.shape)

# %% [markdown]
# ## Run the cleaning pipeline

# %%
cleaned = clean_data.main()

# %% [markdown]
# ## What changed in the employee table

# %%
emp = cleaned["attrition"]
removed = sorted(set(raw_emp.columns) - set(emp.columns))
added = sorted(set(emp.columns) - set(raw_emp.columns))

print(f"columns: {raw_emp.shape[1]} -> {emp.shape[1]}")
print(f"  removed: {removed}")
print(f"  added  : {added}")
print(f"rows   : {len(raw_emp)} -> {len(emp)}")

# %% [markdown]
# `EmployeeNumber` is renamed to `EmployeeID`, so it appears in both lists. The three
# genuinely dropped columns are the zero-variance ones found in notebook 01, and
# `AttritionFlag` is the 0/1 encoding of the target for modelling - the original
# `Attrition` Yes/No column is kept for reporting.

# %%
print("Attrition vs AttritionFlag agree:",
      bool((emp["AttritionFlag"] == (emp["Attrition"] == "Yes").astype(int)).all()))
print(emp[["EmployeeID", "Attrition", "AttritionFlag"]].head())

# %% [markdown]
# ## Outliers: reported, not removed

# %%
outliers = clean_data.report_outliers(
    emp, ["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
          "YearsSinceLastPromotion", "DistanceFromHome", "Age"])
outliers

# %% [markdown]
# `MonthlyIncome` flags around 8% of rows as IQR outliers. Those are directors and
# managers, not corrupted values - the column is genuinely right-skewed, as notebook 01
# showed. Clipping or dropping them would delete the most senior slice of the
# workforce, which is precisely the population where a wrong attrition call is most
# expensive.
#
# So: flagged and documented, values untouched. Tree models handle skew natively, and
# the linear baseline gets a scaler instead.

# %%
top_earners = emp.nlargest(5, "MonthlyIncome")[
    ["EmployeeID", "JobRole", "JobLevel", "TotalWorkingYears", "MonthlyIncome"]]
print("The 'outliers' in MonthlyIncome:")
print(top_earners.to_string(index=False))

# %% [markdown]
# All five are Managers or Research Directors with long careers. Real people, real
# salaries.

# %% [markdown]
# ## Skill name canonicalisation
#
# The problem from the build notes: `AWS`, `Amazon Web Services` and `AWS Cloud` are
# one skill. Left split, the gap engine counts one shortage three times and recommends
# training an employee already has.

# %%
examples = [
    "AWS", "Amazon Web Services", "AWS Cloud", "Amazon Web Services AWS software",
    "SAP software", "Salesforce software", "The MathWorks MATLAB",
    "Micosoft SQL Server Analysis Services SSAS", "Microsoft Office software",
    "  excel  ", "Applicant tracking software", "Electronic medical record EMR software",
]
print(f"{'raw':45s} -> canonical")
for e in examples:
    print(f"{e!r:45s} -> {normalize_skill(e)!r}")

# %% [markdown]
# ### What it actually merged in this data
#
# Worth checking honestly rather than assuming the normaliser earned its keep.

# %%
raw_soft = pd.read_csv(RAW_FILES["software_skills"])
audit = normalization_audit(raw_soft["Workplace Example"].dropna().unique())

print(f"canonical names that absorbed more than one raw spelling: {len(audit)}")
for canon, raws in list(audit.items())[:10]:
    print(f"  {canon!r}")
    for r in raws:
        print(f"      <- {r!r}")

# %%
soft_clean = cleaned["software_skills"]
print(f"software_skills rows: {len(raw_soft):,} -> {len(soft_clean):,} "
      f"({len(raw_soft) - len(soft_clean)} merged as duplicate (role, skill) pairs)")

req = cleaned["role_requirements"]
print(f"distinct required skills across all roles: {req['SkillName'].nunique()}")

# %% [markdown]
# Within the role-requirement vocabulary specifically, every raw name was already
# distinct, so normalisation merged nothing there. It still matters: it is what keeps
# the *employee* skills table joinable to requirements, and it is what will absorb the
# abbreviations ("AWS", "Excel") that a real HR skills export contains.

# %% [markdown]
# ## Re-validate after cleaning
#
# Cleaning is itself a transformation that can introduce bugs, so the same rules run
# again on the output.

# %%
from app.validation.employee_schema import validate_employee_frame
from app.validation.engagement_schema import (
    validate_employee_skills_frame,
    validate_engagement_frame,
)

print(validate_employee_frame(emp))
print(validate_engagement_frame(cleaned["engagement"]))
print(validate_employee_skills_frame(cleaned["employee_skills"],
                                     valid_employee_ids=set(emp["EmployeeID"])))

# %% [markdown]
# ## Output

# %%
for key in ["attrition", "engagement", "occupation", "essential_skills",
            "software_skills", "role_requirements", "employee_skills"]:
    path = PROCESSED_FILES[key]
    print(f"{path.name:45s} {path.stat().st_size / 1_000:>9,.0f} KB")

# %% [markdown]
# Every dataset now has a clean copy, keyed consistently on `EmployeeID`, with
# canonical skill names. Notebook 04 confirms how they join.
