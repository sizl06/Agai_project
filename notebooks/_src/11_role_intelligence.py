# %% [markdown]
# # 11 - Role Intelligence
#
# Day 3, step 11. `occupation_data.csv` becomes the role master table. No ML - this is
# reference data.
#
# The point is to remove ambiguity: once every role has a stable O*NET code, "Research
# Scientist requires Python, R, SAS" is a statement about a specific, defined
# occupation rather than about a job title someone typed into a spreadsheet.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from app.utils.config import DOCS_DIR, PROCESSED_FILES

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)

occupations = pd.read_csv(PROCESSED_FILES["occupation"])
employees = pd.read_csv(PROCESSED_FILES["attrition"])
requirements = pd.read_csv(PROCESSED_FILES["role_requirements"])
mapping = pd.read_csv(DOCS_DIR / "role_to_onet_mapping.csv")

print(f"O*NET occupations available : {len(occupations):,}")
print(f"IBM job roles in use        : {employees['JobRole'].nunique()}")

# %% [markdown]
# ## The mapping, and why each choice was made
#
# The IBM dataset has nine job titles. O*NET has 1,016 occupations. Each title is
# mapped to exactly one occupation, so that role-level skill requirements never
# collide.

# %%
mapping.merge(
    employees.groupby("JobRole").size().rename("Headcount"),
    left_on="JobRole", right_index=True,
).sort_values("Headcount", ascending=False)

# %% [markdown]
# Two of these are judgement calls worth recording rather than burying:
#
# * **Healthcare Representative** is not a clinical role in this dataset - it is a
#   sales role, sitting in R&D and closest to pharmaceutical or medical-device sales.
#   Mapped to *Sales Representatives, Technical and Scientific Products*, which keeps
#   it distinct from the two general sales titles.
# * **Manager** is deliberately mapped to the generic *General and Operations Managers*
#   rather than to a department-specific management occupation. The IBM title carries
#   no department-specific meaning, and inventing one would put skills on the
#   requirement list that nobody asked for.
#
# Anyone disputing these can change one dictionary in
# `scripts/build_derived_sources.py`; nothing else in the pipeline hard-codes a role.

# %% [markdown]
# ## The role master table

# %%
role_master = (
    # ONET_Title exists in both frames; the occupation master is authoritative.
    mapping.drop(columns=["ONET_Title"])
    .merge(occupations, on="ONET_SOC_Code", how="left")
    .merge(employees.groupby("JobRole").size().rename("Headcount"), left_on="JobRole", right_index=True)
    .merge(
        employees.groupby("JobRole")["AttritionFlag"].mean().mul(100).round(1).rename("AttritionRate"),
        left_on="JobRole", right_index=True,
    )
    .merge(requirements.groupby("JobRole").size().rename("RequiredSkills"),
           left_on="JobRole", right_index=True)
    .sort_values("Headcount", ascending=False)
    .reset_index(drop=True)
)
role_master[["JobRole", "ONET_SOC_Code", "Headcount", "AttritionRate", "RequiredSkills"]]

# %% [markdown]
# ## Every role resolves to a real occupation

# %%
unresolved = role_master[role_master["ONET_Title"].isna()]
print("roles that failed to resolve:", len(unresolved))
assert unresolved.empty, "every mapped SOC code must exist in occupation_data"
print("All nine roles resolve to an O*NET occupation.")

# %% [markdown]
# ## What O*NET says each role does

# %%
for _, row in role_master.head(4).iterrows():
    print(f"\n--- {row['JobRole']}  ({row['ONET_SOC_Code']}) ---")
    print(f"O*NET title: {row['ONET_Title']}")
    print(f"{row['ONET_Description'][:320]}...")

# %% [markdown]
# ## Required skills per role

# %%
skill_profile = (
    requirements.groupby(["JobRole", "SkillType"]).size().unstack(fill_value=0)
    .assign(Total=lambda d: d.sum(axis=1))
    .sort_values("Total", ascending=False)
)
skill_profile

# %%
fig, ax = plt.subplots(figsize=(10, 5))
bottom = None
for column, colour in [("Essential Skill", "#4C72B0"), ("Software Skill", "#DD8452")]:
    if column in skill_profile.columns:
        ax.bar(skill_profile.index, skill_profile[column], bottom=bottom,
               label=column, color=colour)
        bottom = skill_profile[column] if bottom is None else bottom + skill_profile[column]
ax.set(ylabel="required skills", title="Skill requirements per role")
ax.legend()
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## A single role in full

# %%
role = "Research Scientist"
print(f"=== {role} ===")
print(requirements[requirements["JobRole"] == role]
      .sort_values(["SkillType", "RankInRole"])[["SkillType", "SkillName", "Importance"]]
      .to_string(index=False))

# %% [markdown]
# ## Which skills are demanded by the most roles
#
# A skill required across many roles is a stronger candidate for centralised training
# than one needed by a single team.

# %%
breadth = (requirements.groupby("SkillName")
           .agg(RolesRequiring=("JobRole", "nunique"), AvgImportance=("Importance", "mean"))
           .round(2)
           .sort_values(["RolesRequiring", "AvgImportance"], ascending=False))
breadth.head(15)

# %% [markdown]
# The ten essential skills are required by all nine roles - unsurprising, since O*NET
# rates every occupation on the same skill inventory. The software requirements are
# what actually differentiate roles, and they are where the interesting gaps appear in
# notebook 13.

# %%
role_master.to_csv(PROCESSED_FILES["occupation"].parent / "role_master.csv", index=False)
print("written: role_master.csv")
