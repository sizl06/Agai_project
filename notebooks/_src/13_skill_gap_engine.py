# %% [markdown]
# # 13 - Skill Gap Engine
#
# Day 3, step 13. The core of the whole project, and it is set subtraction:
#
# ```python
# required = {'Python', 'SQL', 'MLOps', 'Docker', 'AWS'}   # ML Engineer
# has      = {'Python', 'SQL', 'AWS'}                       # Employee 101
# gap      = required - has                                 # {'MLOps', 'Docker'}
# ```
#
# No model required. What the implementation adds is importance weighting and two
# specific defences against getting the joins wrong.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from app.services import skill_gap_service as gaps_service
from app.utils.config import PROCESSED_FILES

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

employees, requirements, skills = gaps_service.load_inputs()
print(f"employees {len(employees):,} | requirement rows {len(requirements)} | "
      f"skill records {len(skills):,}")

# %% [markdown]
# ## The two failure modes this has to avoid
#
# **1. Row explosion.** Employees join to requirements many-to-many through `JobRole`.
# Notebook 04 measured it: a naive merge turns 1,470 employees into 26,460 rows. So
# requirements are collapsed to one set per role *before* any comparison.
#
# **2. Silently dropping employees.** Someone with no skills on file must come out as
# missing *every* required skill. An inner join would drop them and understate the
# shortage - the failure would be invisible because the output still looks reasonable.

# %%
role_requirements = gaps_service.required_skills_by_role(requirements)
employee_skills = gaps_service.skills_by_employee(skills)

print(f"roles with requirements  : {len(role_requirements)}")
print(f"employees with skill sets: {len(employee_skills):,}")
print(f"\nexample - Research Scientist requires {len(role_requirements['Research Scientist'])} skills")

# %% [markdown]
# ## The calculation on one employee

# %%
example = employees.iloc[0]
employee_id = int(example["EmployeeID"])
role = example["JobRole"]

required = set(role_requirements[role])
held = employee_skills.get(employee_id, set())
gap = required - held

print(f"Employee {employee_id} - {role}\n")
print(f"required ({len(required)}):")
print("   ", ", ".join(sorted(required)))
print(f"\nhas ({len(held)}):")
print("   ", ", ".join(sorted(held)))
print(f"\ngap ({len(gap)}):")
print("   ", ", ".join(sorted(gap)))

# %% [markdown]
# ## Across the whole organisation

# %%
gaps = gaps_service.compute_employee_gaps(employees, requirements, skills)
print(f"gap rows: {len(gaps):,}")
gaps.head(10)

# %% [markdown]
# ### The spine is intact

# %%
print(f"distinct employees in gaps : {gaps['EmployeeID'].nunique():,}")
print(f"employees in the source    : {len(employees):,}")
print(f"gap rows                   : {len(gaps):,}  (one per employee/missing skill)")

summary = gaps_service.employee_gap_summary(gaps, employees)
print(f"\nsummary rows: {len(summary):,}  <- must equal the employee count")
assert len(summary) == len(employees), "the gap summary must stay at one row per employee"
assert not summary["EmployeeID"].duplicated().any()
print("No row explosion, no dropped employees.")

# %% [markdown]
# ## Distribution of gap sizes

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
counts = summary["SkillGapCount"]
ax1.hist(counts, bins=range(int(counts.min()), int(counts.max()) + 2),
         color="#C44E52", edgecolor="white", align="left")
ax1.set(xlabel="skills missing", ylabel="employees", title="Skill gaps per employee")

by_role = summary.groupby("JobRole")["SkillGapCount"].mean().sort_values()
ax2.barh(by_role.index, by_role.values, color="#4C72B0")
ax2.set(xlabel="average skills missing", title="Average gap by role")
plt.tight_layout()
plt.show()

print(f"average gap: {counts.mean():.1f} skills | "
      f"employees with no gap at all: {int((counts == 0).sum())}")

# %% [markdown]
# ## Weighting by importance
#
# Missing two trivial skills is not the same as missing two critical ones.
# `SkillGapScore` sums the O*NET importance of everything an employee lacks, so it
# ranks by *severity* rather than by count.

# %%
summary.nlargest(10, "SkillGapScore")[
    ["EmployeeID", "JobRole", "SkillGapCount", "SkillGapScore", "TopMissingSkill"]]

# %% [markdown]
# ## Count and severity are not the same ranking

# %%
by_count = set(summary.nlargest(50, "SkillGapCount")["EmployeeID"])
by_score = set(summary.nlargest(50, "SkillGapScore")["EmployeeID"])
overlap = len(by_count & by_score)
print(f"top 50 by gap count vs top 50 by weighted score: {overlap} in common, "
      f"{50 - overlap} differ")
print("\nRanking by raw count would send the wrong people to training first.")

# %% [markdown]
# ## By department

# %%
gaps_service.gaps_by_department(gaps)

# %% [markdown]
# ## Edge case: an employee with nothing on file
#
# The defence described at the top, tested rather than asserted.

# %%
from app.services.skill_gap_service import compute_employee_gaps

ghost = employees.head(1).copy()
ghost["EmployeeID"] = 999999
empty_skills = skills.head(0)

ghost_gaps = compute_employee_gaps(ghost, requirements, empty_skills)
expected = len(role_requirements[ghost["JobRole"].iloc[0]])
print(f"employee with no skills on file -> {len(ghost_gaps)} gaps "
      f"(role requires {expected})")
assert len(ghost_gaps) == expected, "an employee with no skills must be missing all of them"
print("Correct: treated as missing every required skill, not as having no gap.")

# %%
gaps.to_csv(PROCESSED_FILES["skill_gaps"], index=False)
print(f"written: {PROCESSED_FILES['skill_gaps'].name}  ({len(gaps):,} rows)")
