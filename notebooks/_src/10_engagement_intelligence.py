# %% [markdown]
# # 10 - Engagement Intelligence
#
# Day 3, step 10. No ML here, and none is needed - these are aggregations. The build
# notes are right that a model adds nothing until there is a target worth predicting.
#
# > **Numbering note.** The build notes label this `09_engagement_intelligence`, which
# > collides with step 9 (model versioning). Renumbered to 10 so the notebook order
# > matches the checklist order.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from app.services import engagement_service as engagement
from app.utils.config import PROCESSED_FILES

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = engagement.load_engagement()
print(df.shape)
df.head()

# %% [markdown]
# ## Headline figures

# %%
summary = engagement.summary(df)
pd.Series(summary).to_frame("value")

# %% [markdown]
# ## Distribution

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
ax1.hist(df["EngagementScore"], bins=30, color="#4C72B0", edgecolor="white")
ax1.axvline(summary["average_engagement"], color="crimson", linestyle="--",
            label=f"mean = {summary['average_engagement']}")
ax1.set(xlabel="engagement score", ylabel="employees", title="Engagement distribution")
ax1.legend()

bands = engagement.band_distribution(df)
ax2.bar(bands["band"], bands["employees"], color=["#C44E52", "#DD8452", "#55A868"])
for i, (n, pct) in enumerate(zip(bands["employees"], bands["percent"])):
    ax2.text(i, n + 8, f"{n}\n({pct}%)", ha="center", fontsize=9)
ax2.set(ylabel="employees", title="Engagement bands")
plt.tight_layout()
plt.show()

bands

# %% [markdown]
# The distribution is lumpy rather than smooth. That is a direct consequence of the
# source data: engagement is built from five 1-4 Likert items, so only a finite set of
# weighted combinations can occur. It is a property of the instrument, not a data
# error - and a reason to treat small differences between employees as noise.

# %% [markdown]
# ## By department
#
# The build notes' one-liner, with the surrounding context that makes it actionable:
#
# ```python
# performance_df.groupby('Department')['EngagementScore'].mean().sort_values()
# ```

# %%
by_department = engagement.by_department(df)
by_department

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(by_department["Department"], by_department["avg_engagement"], color="#4C72B0")
for i, (v, n) in enumerate(zip(by_department["avg_engagement"], by_department["headcount"])):
    ax.text(v + 0.4, i, f"{v}  (n={n})", va="center", fontsize=9)
ax.set(xlabel="average engagement score", title="Average engagement by department")
ax.set_xlim(0, 80)
plt.tight_layout()
plt.show()

# %% [markdown]
# The spread between departments is a couple of points on a 100-point scale - small
# enough that, with these headcounts, it would be a mistake to act on the ranking
# alone. The variation *within* each department is far larger than the variation
# between them, which is why the individual-level list below is the more useful output.

# %% [markdown]
# ## By role

# %%
engagement.by_role(df)

# %% [markdown]
# ## The employees HR should actually look at

# %%
lowest = engagement.lowest_engaged(15, df)
lowest

# %% [markdown]
# ## High performers who are disengaged
#
# The most expensive group to lose: rated 4 on performance, engagement below 40. They
# are productive now and demonstrably unhappy.

# %%
at_risk = engagement.high_performer_low_engagement(df)
print(f"{len(at_risk)} high performers with engagement below 40")
at_risk.head(15)

# %%
if len(at_risk):
    counts = at_risk["Department"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color="#C44E52")
    ax.set(ylabel="employees", title="Disengaged high performers by department")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Engagement and attrition - with a caveat that matters
#
# The obvious next question is whether engagement predicts who leaves. It appears to,
# and the appearance is partly manufactured.

# %%
employees = pd.read_csv(PROCESSED_FILES["attrition"])
merged = employees[["EmployeeID", "AttritionFlag"]].merge(
    df[["EmployeeID", "EngagementScore"]], on="EmployeeID")

by_outcome = merged.groupby("AttritionFlag")["EngagementScore"].agg(["mean", "median", "count"])
by_outcome.index = ["stayed", "left"]
print(by_outcome.round(2).to_string())
print(f"\npoint-biserial correlation with attrition: "
      f"{merged['EngagementScore'].corr(merged['AttritionFlag']):.4f}")

# %% [markdown]
# > **This is not independent evidence.** `EngagementScore` is a deterministic weighted
# > combination of `JobSatisfaction`, `JobInvolvement`, `EnvironmentSatisfaction`,
# > `WorkLifeBalance` and `RelationshipSatisfaction` - five columns the attrition model
# > already uses as features. Leavers scoring lower on engagement is the same fact as
# > leavers scoring lower on job satisfaction, restated.
# >
# > It is reported because the relationship is real and useful for framing a
# > conversation, but it must not be presented as a second, corroborating signal, and
# > engagement is never fed back into the model as a feature. See
# > `scripts/build_derived_sources.py` and `docs/data_relationships.md`.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.boxplot([merged.loc[merged.AttritionFlag == 0, "EngagementScore"],
            merged.loc[merged.AttritionFlag == 1, "EngagementScore"]],
           labels=["stayed", "left"])
ax.set(ylabel="engagement score", title="Engagement by attrition outcome")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## What a model would add here
#
# Nothing yet, and it is worth being specific about why. There is no engagement
# *target* to predict - the score is an input, not an outcome. The options that would
# make sense later:
#
# * **Regression**, once engagement is measured over multiple periods, to predict next
#   quarter's score from this quarter's conditions.
# * **KMeans segmentation**, to find employee groups with distinct engagement profiles
#   rather than assuming departments are the right grouping.
#
# Both need data this snapshot does not contain. Building either now would be modelling
# for its own sake.
