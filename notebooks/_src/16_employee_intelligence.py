# %% [markdown]
# # 16 - Employee Intelligence Table
#
# Day 3, step 16. Everything from the last three days lands here: one row per employee
# carrying attrition risk, engagement, role, skill gaps and a recommendation.
#
# This table is the actual business output of the project. The dashboard is a view onto
# it, and the API mostly serves slices of it.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from app.utils.config import PROCESSED_FILES
from pipelines import build_intelligence

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 42)

intelligence = build_intelligence.build()
print(intelligence.shape)

# %% [markdown]
# ## The shape the build notes asked for
#
# ```
# Employee_ID | Dept | Attrition_Prob | Risk | Engagement | Role | Skill_Gap | Recommendation
# ```

# %%
intelligence[["EmployeeID", "Department", "AttritionProbability", "RiskLevel",
              "EngagementScore", "JobRole", "TopMissingSkill", "Recommendation"]].head(10)

# %% [markdown]
# ## The join discipline that keeps this correct
#
# Notebook 04 established the rule and this is where it pays off. The employee table is
# the spine; every other source is aggregated to one row per employee *before* joining.
#
# Skill gaps arrive at ~15,500 rows and recommendations at ~4,400. Joining either
# directly would multiply the spine and inflate every count on the dashboard - and the
# result would still look plausible, which is what makes it dangerous.

# %%
employees = pd.read_csv(PROCESSED_FILES["attrition"])
gaps = pd.read_csv(PROCESSED_FILES["skill_gaps"])
recommendations = pd.read_csv(PROCESSED_FILES["recommendations"])

print(f"employees (spine)      {len(employees):>7,}")
print(f"skill gap rows         {len(gaps):>7,}")
print(f"recommendation rows    {len(recommendations):>7,}")
print(f"intelligence table     {len(intelligence):>7,}  <- must equal the spine")

assert len(intelligence) == len(employees)
assert not intelligence["EmployeeID"].duplicated().any()
print("\nOne row per employee, no duplicates.")

# %% [markdown]
# `build_intelligence.build()` raises rather than writing a bad table if either check
# fails, so this cannot silently regress later.

# %% [markdown]
# ## Completeness

# %%
pd.DataFrame({
    "non_null": intelligence.notna().sum(),
    "null": intelligence.isna().sum(),
    "dtype": intelligence.dtypes.astype(str),
})

# %% [markdown]
# ## The dashboard KPIs

# %%
kpis = {
    "Employees": len(intelligence),
    "High risk": int((intelligence["RiskLevel"] == "HIGH").sum()),
    "Medium risk": int((intelligence["RiskLevel"] == "MEDIUM").sum()),
    "Avg engagement": round(intelligence["EngagementScore"].mean(), 1),
    "Employees with skill gaps": int((intelligence["SkillGapCount"] > 0).sum()),
    "Avg skill gaps": round(intelligence["SkillGapCount"].mean(), 1),
}
pd.Series(kpis).to_frame("value")

# %% [markdown]
# > **These risk counts are in-sample.** Every employee here was part of the data the
# > model was trained on, so the probabilities are optimistic in the way any model is
# > about data it has already seen. The honest performance estimate is the held-out test
# > result in notebook 07 (ROC-AUC 0.80), not anything computed from this table. In
# > production the model would score employees it had never seen.

# %% [markdown]
# ## Risk distribution

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
order = ["HIGH", "MEDIUM", "LOW"]
counts = intelligence["RiskLevel"].value_counts().reindex(order)
ax1.bar(order, counts.values, color=["#C44E52", "#DD8452", "#55A868"])
for i, v in enumerate(counts.values):
    ax1.text(i, v + 8, f"{v}\n({v / len(intelligence) * 100:.1f}%)", ha="center", fontsize=9)
ax1.set(ylabel="employees", title="Attrition risk bands")

ax2.hist(intelligence["AttritionProbability"], bins=40, color="#4C72B0", edgecolor="white")
ax2.set(xlabel="attrition probability", ylabel="employees", title="Probability distribution")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Attrition risk by department - the dashboard's main chart

# %%
by_department = (intelligence.groupby("Department")
                 .agg(Headcount=("EmployeeID", "count"),
                      HighRisk=("RiskLevel", lambda s: (s == "HIGH").sum()),
                      AvgProbability=("AttritionProbability", "mean"),
                      AvgEngagement=("EngagementScore", "mean"))
                 .round(3))
by_department["HighRisk%"] = (by_department["HighRisk"] / by_department["Headcount"] * 100).round(1)
by_department.sort_values("HighRisk%", ascending=False)

# %%
plot_data = by_department.sort_values("HighRisk%")
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(plot_data.index, plot_data["HighRisk%"], color="#C44E52")
for i, (pct, n) in enumerate(zip(plot_data["HighRisk%"], plot_data["HighRisk"])):
    ax.text(pct + 0.4, i, f"{pct}%  ({n})", va="center", fontsize=9)
ax.set(xlabel="% of department at HIGH risk", title="Attrition risk by department")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Where risk and skill gaps overlap
#
# The cross-tabulation is the argument for building all three subsystems rather than
# just the model: it identifies employees who are both likely to leave *and* under-
# skilled, where targeted development is a retention lever rather than only a
# capability one.

# %%
intelligence["GapBand"] = pd.cut(intelligence["SkillGapCount"],
                                 bins=[-1, 5, 9, 100],
                                 labels=["few (0-5)", "some (6-9)", "many (10+)"])
crosstab = pd.crosstab(intelligence["RiskLevel"], intelligence["GapBand"]).reindex(order)
crosstab

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
crosstab.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn_r")
ax.set(ylabel="employees", title="Attrition risk vs skill gap size")
plt.xticks(rotation=0)
plt.legend(title="skill gaps")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## The priority list
#
# High attrition risk, large skill gap, low engagement - the employees to act on first.

# %%
priority = (intelligence[intelligence["RiskLevel"] == "HIGH"]
            .sort_values(["SkillGapScore", "EngagementScore"], ascending=[False, True]))

priority.head(15)[["EmployeeID", "Department", "JobRole", "AttritionProbability",
                   "EngagementScore", "SkillGapCount", "TopMissingSkill", "Recommendation"]]

# %% [markdown]
# ## A single employee, end to end

# %%
employee = priority.iloc[0]
print(f"=== Employee {int(employee['EmployeeID'])} ===")
print(f"  Role              : {employee['JobRole']} ({employee['Department']})")
print(f"  Age / tenure      : {int(employee['Age'])} / {int(employee['YearsAtCompany'])} years")
print(f"  Overtime          : {employee['OverTime']}")
print(f"  Attrition risk    : {employee['AttritionProbability']:.1%} ({employee['RiskLevel']})")
print(f"  Engagement        : {employee['EngagementScore']} ({employee['EngagementBand']})")
print(f"  Skill gaps ({int(employee['SkillGapCount'])})    : {employee['SkillGaps']}")
print(f"  Recommendation    : {employee['Recommendation']}")
print(f"  Course            : {employee['CourseTitle']} ({employee['Provider']}, "
      f"{employee['DurationHours']:.0f}h)")

# %% [markdown]
# ## Day 3 summary
#
# | Step | Output |
# |---|---|
# | 10 Engagement | department and role breakdowns; disengaged high performers identified |
# | 11 Role intelligence | nine job roles mapped to O*NET occupations, mapping documented |
# | 12 Employee skills | confirmed absent from source data; controlled table built and validated |
# | 13 Skill gap engine | 15,509 employee/skill gaps, importance-weighted |
# | 14 Organisation gaps | severity per the notes, plus a proportional rule that discriminates |
# | 15 Recommendations | tag-based matching with prioritisation by importance x severity x risk |
# | 16 Intelligence table | 1,470 rows x 23 columns, one per employee |
#
# Day 4 turns this into a service.

# %%
print(f"written: {PROCESSED_FILES['intelligence']}")
print(f"columns: {list(intelligence.columns)}")
