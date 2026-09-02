# %% [markdown]
# # 02 - Data Validation
#
# Day 1, step 2. Before cleaning anything, define what "valid" actually means.
#
# The motivating case from the build notes: one day someone hands over a CSV with an
# `EngagementScore` of 250 out of 100. Without a rule that rejects it, that row flows
# silently into a department average and then into a dashboard someone makes a
# decision from.
#
# The rules live in `app/validation/` rather than in this notebook, so the same
# definitions are used by the batch pipelines, the API, and the unit tests. That is the
# step-19 refactor applied early - one rule set, not three that drift apart.

# %%
import pandas as pd

from app.utils.config import CATEGORICAL_DOMAINS, RAW_FILES, VALIDATION_RULES
from app.validation.employee_schema import validate_employee_frame
from app.validation.engagement_schema import (
    validate_employee_skills_frame,
    validate_engagement_frame,
)
from app.validation.schema_checks import ValidationReport, check_referential_integrity

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("Range rules in force:")
for col, (lo, hi) in VALIDATION_RULES.items():
    print(f"  {col:24s} [{lo}, {hi}]")
print("\nCategorical domains:")
for col, allowed in CATEGORICAL_DOMAINS.items():
    print(f"  {col:24s} {sorted(allowed)}")

# %% [markdown]
# ## What the rules cover
#
# | Check | Question it answers |
# |---|---|
# | schema | do the expected columns exist at all? |
# | type | is `MonthlyIncome` numeric, not text? |
# | range | is `Age` between 18 and 100, `EngagementScore` between 0 and 100? |
# | uniqueness | does `EmployeeID` ever repeat? |
# | category | does `Attrition` only ever contain Yes/No? |
# | nulls | are the columns that must be complete actually complete? |
# | referential | does every skill row point at a real employee? |

# %% [markdown]
# ## Validating the real data

# %%
emp = pd.read_csv(RAW_FILES["attrition"], encoding="utf-8-sig")
emp.columns = [c.strip() for c in emp.columns]
emp = emp.rename(columns={"EmployeeNumber": "EmployeeID"})

report_emp = validate_employee_frame(emp)
print(report_emp)

# %%
eng = pd.read_csv(RAW_FILES["engagement"])
report_eng = validate_engagement_frame(eng)
print(report_eng)

# %%
skills = pd.read_csv(RAW_FILES["employee_skills"])
report_skills = validate_employee_skills_frame(skills, valid_employee_ids=set(emp["EmployeeID"]))
print(report_skills)

# %% [markdown]
# All three pass. That is a real result, not a vacuous one - but a validator that has
# only ever seen clean data is untested. The next section proves the rules actually
# fire.

# %% [markdown]
# ## Proving the rules catch what they claim to
#
# The frame below is a **deliberately corrupted copy** used only inside this notebook.
# Nothing here is written to disk and none of it enters the pipeline; it exists to show
# that each rule rejects the failure it was written for.

# %%
broken = emp.head(60).copy()
broken.loc[broken.index[0], "Age"] = 7                       # below the legal minimum
broken.loc[broken.index[1], "Age"] = 150                     # implausible
broken.loc[broken.index[2], "Attrition"] = "Maybe"           # outside the domain
broken.loc[broken.index[3], "EmployeeID"] = broken["EmployeeID"].iloc[4]  # duplicate key
broken.loc[broken.index[5], "MonthlyIncome"] = None          # null in a required field
broken = broken.drop(columns=["JobSatisfaction"])            # missing required column

report_broken = validate_employee_frame(broken)
print(report_broken)

# %%
assert not report_broken.ok, "the corrupted frame should not pass validation"
print(f"Rejected with {len(report_broken.errors)} distinct errors, as intended.")

# %% [markdown]
# ### The engagement case the notes specifically call out

# %%
broken_eng = eng.head(40).copy()
broken_eng.loc[broken_eng.index[0], "EngagementScore"] = 250.0   # the 250-out-of-100 row
broken_eng.loc[broken_eng.index[1], "EngagementScore"] = -12.0
broken_eng.loc[broken_eng.index[2], "PerformanceRating"] = 9

report_broken_eng = validate_engagement_frame(broken_eng)
print(report_broken_eng)
assert not report_broken_eng.ok

# %% [markdown]
# Caught. Had this row passed, it would have shifted its department's mean engagement
# by roughly `(250 - 72) / n` - invisible on a dashboard, wrong in a decision.

# %% [markdown]
# ## Cross-dataset referential integrity
#
# Every engagement row and every skill row must point at an employee that exists.

# %%
report_refs = ValidationReport(dataset="cross-dataset references", n_rows=len(emp))
valid_ids = set(emp["EmployeeID"])

check_referential_integrity(eng, "EmployeeID", valid_ids, report_refs, "engagement")
check_referential_integrity(skills, "EmployeeID", valid_ids, report_refs, "employee_skills")

role_req = pd.read_csv(RAW_FILES["role_requirements"])
check_referential_integrity(role_req, "JobRole", set(emp["JobRole"]), report_refs, "role_requirements")

print(report_refs)

# %% [markdown]
# ## Coverage check: does every employee have skills recorded?
#
# A referential check confirms no *orphans*, but not that every employee is *covered*.
# The skill-gap engine has to handle an employee with nothing on file.

# %%
covered = set(skills["EmployeeID"].unique())
uncovered = valid_ids - covered
print(f"Employees with at least one recorded skill: {len(covered):,} / {len(valid_ids):,}")
print(f"Employees with no skills on file          : {len(uncovered):,}")
if uncovered:
    print(f"  e.g. {sorted(uncovered)[:5]}")
print("\nSkills per employee:")
print(skills.groupby("EmployeeID").size().describe().round(2).to_string())

# %% [markdown]
# Whatever this number is, notebook 13 must treat "no skills on file" as *every
# required skill missing*, not as *no gap*. Silently dropping those employees from a
# left join is the classic way that bug gets introduced.

# %% [markdown]
# ## Validation summary

# %%
pd.DataFrame([
    {"dataset": r.dataset, "rows": r.n_rows, "checks": r.checks_run,
     "errors": len(r.errors), "warnings": len(r.warnings), "status": "PASS" if r.ok else "FAIL"}
    for r in [report_emp, report_eng, report_skills, report_refs]
])

# %% [markdown]
# All raw inputs are structurally sound. The problems found in notebook 01 are
# *cleanliness* problems, not *validity* problems - a wrong column name, zero-variance
# columns, and inconsistent skill spellings. Those are notebook 03's job.
