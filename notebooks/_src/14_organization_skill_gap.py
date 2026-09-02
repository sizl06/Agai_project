# %% [markdown]
# # 14 - Organisation-Wide Skill Gap
#
# Day 3, step 14. The same gaps rolled up across everyone, so the question changes from
# "what does this person need?" to "what is this company short of?".
#
# The build notes specify a severity rule: 100+ employees missing a skill is HIGH, 50+
# is MEDIUM, otherwise LOW. That rule is implemented exactly as written - and then
# tested against this dataset, where it turns out not to work.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from app.services import skill_gap_service as gaps_service
from app.utils.config import (
    PROCESSED_FILES,
    SEVERITY_BANDS,
    SEVERITY_BANDS_RELATIVE,
    gap_severity,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

employees = pd.read_csv(PROCESSED_FILES["attrition"])
gaps = pd.read_csv(PROCESSED_FILES["skill_gaps"])
print(f"{len(gaps):,} gap rows across {len(employees):,} employees")

# %% [markdown]
# ## Rolling up

# %%
org = gaps_service.organization_skill_gaps(gaps, total_employees=len(employees))
org.head(20)

# %% [markdown]
# ## The output the dashboard shows

# %%
top = org.head(12)
fig, ax = plt.subplots(figsize=(10, 6))
colours = {"HIGH": "#C44E52", "MEDIUM": "#DD8452", "LOW": "#55A868"}
ax.barh(top["MissingSkill"][::-1], top["EmployeesMissing"][::-1],
        color=[colours[s] for s in top["SeverityRelative"][::-1]])
for i, (n, pct) in enumerate(zip(top["EmployeesMissing"][::-1], top["pct_of_workforce"][::-1])):
    ax.text(n + 8, i, f"{n:,} ({pct}%)", va="center", fontsize=9)
ax.set(xlabel="employees missing the skill", title="Critical organisation skill gaps")
ax.set_xlim(0, top["EmployeesMissing"].max() * 1.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## The severity rule as specified, and why it fails here

# %%
print(f"Build-notes bands (absolute headcount): {SEVERITY_BANDS}")
print(f"\nDistribution over {len(org)} distinct skills:")
print(org["Severity"].value_counts().to_string())

# %% [markdown]
# **36 of 52 skills come out HIGH.** A severity rating that applies to 69% of the skill
# catalogue does not help anyone decide where to spend a training budget - the
# dashboard would show a wall of red and the ranking inside it would carry all the
# actual information.
#
# The cause is a scale mismatch, not a bug. The 100/50 thresholds were written with a
# ~2,500-employee organisation in mind. At 1,470 employees, 100 people is under 7% of
# the workforce - a low bar for "HIGH". And because every role shares the ten O*NET
# essential skills, common gaps naturally reach hundreds of employees.

# %%
distribution = org["EmployeesMissing"]
print(f"employees missing a given skill: min {distribution.min()}, "
      f"median {distribution.median():.0f}, max {distribution.max()}")
print(f"skills above the HIGH threshold of {SEVERITY_BANDS['HIGH']}: "
      f"{int((distribution >= SEVERITY_BANDS['HIGH']).sum())} of {len(org)}")

# %% [markdown]
# ## The proportional alternative
#
# Expressing severity as a share of the workforce is size-independent: HIGH at 40% of
# employees, MEDIUM at 20%. Both columns are carried in the output rather than one
# replacing the other, so the spec's rule stays available and the change is a reporting
# choice rather than a silent redefinition.

# %%
print(f"Relative bands (share of workforce): {SEVERITY_BANDS_RELATIVE}")
print()
print(pd.crosstab(org["Severity"], org["SeverityRelative"],
                  rownames=["spec (absolute)"], colnames=["relative"]))

# %%
comparison = pd.DataFrame({
    "spec (absolute)": org["Severity"].value_counts(),
    "relative": org["SeverityRelative"].value_counts(),
}).fillna(0).astype(int).reindex(["HIGH", "MEDIUM", "LOW"])

fig, ax = plt.subplots(figsize=(8, 4.5))
comparison.plot(kind="bar", ax=ax, color=["#C44E52", "#4C72B0"])
ax.set(ylabel="number of skills", title="Severity band distribution under each rule")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

comparison

# %% [markdown]
# Nine skills come out HIGH under the proportional rule instead of thirty-six. That is
# a list a training manager can actually act on.

# %%
critical = org[org["SeverityRelative"] == "HIGH"]
print("Critical organisation-wide gaps (>= 40% of the workforce):\n")
print(critical[["MissingSkill", "EmployeesMissing", "pct_of_workforce",
                "AvgImportance", "RolesAffected"]].to_string(index=False))

# %% [markdown]
# ## What these gaps actually are
#
# The critical list is dominated by O*NET *essential* skills - Active Learning,
# Writing, Critical Thinking, Speaking, Active Listening - plus the ubiquitous office
# tools. That is a meaningful distinction for what to do next.
#
# Essential skills are required by all nine roles, so a shortfall shows up across the
# whole workforce and points toward broad, centrally-run training. Software gaps are
# concentrated in specific roles and are better handled team by team.

# %%
requirements = pd.read_csv(PROCESSED_FILES["role_requirements"])
skill_types = requirements[["SkillName", "SkillType"]].drop_duplicates()
org_typed = org.merge(skill_types, left_on="MissingSkill", right_on="SkillName", how="left")

print(org_typed.groupby(["SkillType", "SeverityRelative"]).size()
      .unstack(fill_value=0).to_string())

# %% [markdown]
# ## Where the gaps sit

# %%
by_department = (gaps.groupby(["Department", "MissingSkill"]).size()
                 .rename("EmployeesMissing").reset_index())
pivot = (by_department.pivot(index="MissingSkill", columns="Department",
                             values="EmployeesMissing")
         .fillna(0).astype(int))
pivot = pivot.loc[org.head(12)["MissingSkill"]]

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(pivot.values, cmap="Reds", aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, pivot.values[i, j], ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.7, label="employees missing")
ax.set_title("Top skill gaps by department")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## A note on interpreting these totals
#
# The mechanism is real; the headcounts are not measurements. Which employee holds
# which skill is synthesised (notebook 12), so "879 employees are missing Active
# Learning" demonstrates that the roll-up works - it is not a finding about a real
# workforce. Point a real skills inventory at
# `data/raw/employee_current_skills.csv` and these same numbers become real.

# %%
org.to_csv(PROCESSED_FILES["org_skill_gaps"], index=False)
print(f"written: {PROCESSED_FILES['org_skill_gaps'].name}")
print(f"\nSeverity per the build notes : {dict(org['Severity'].value_counts())}")
print(f"Severity, proportional        : {dict(org['SeverityRelative'].value_counts())}")
