# %% [markdown]
# # 05 - Feature Engineering
#
# Day 2, step 5. Turning cleaned columns into model-ready features.
#
# The rule from the build notes: **every engineered feature needs a stated
# statistical or business reason.** Anything that merely "seemed interesting" is noise
# the model has to learn to ignore, and with only 1,470 rows there is no budget for
# that.
#
# The implementation lives in `app/ml/features.py` and is embedded in the sklearn
# pipeline as a step, so training and serving cannot drift apart.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from app.ml.features import (
    ENGINEERED_COLUMNS,
    EXCLUDED_COLUMNS,
    build_preprocessor,
    check_leakage,
    engineer_features,
    split_column_types,
)
from app.utils.config import PROCESSED_FILES

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_csv(PROCESSED_FILES["attrition"])
y = df["AttritionFlag"].astype(int)
print("input:", df.shape, "| positive rate:", round(y.mean(), 4))

# %% [markdown]
# ## What is deliberately excluded

# %%
print("Never used as features:", EXCLUDED_COLUMNS)

# %% [markdown]
# `EmployeeID` matters most here. It is an arbitrary identifier, but a tree model will
# happily split on it and carve out "high risk" ID ranges that mean nothing at all -
# the model would score well in cross-validation and fail completely on new employees
# whose IDs fall outside the training range.

# %% [markdown]
# ## The engineered features

# %%
X = engineer_features(df)
print(f"columns: {df.shape[1]} -> {X.shape[1]}")
print(f"engineered: {ENGINEERED_COLUMNS}")
X[ENGINEERED_COLUMNS].describe().T.round(3)

# %% [markdown]
# | Feature | Reason it exists |
# |---|---|
# | `IncomePerYearAtCompany` | pay relative to tenure - long service on low pay is a classic flight risk that raw income cannot express |
# | `TenureRatio` | share of the whole career spent here; separates movers from the invested |
# | `PromotionStagnation` | time since promotion *relative to time in role* - being "due" is what matters, not the raw year count |
# | `AvgYearsPerCompany` | average stint length - a direct behavioural measure of job-hopping |
# | `IncomePerJobLevel` | pay fairness within a grade; under-payment relative to peers drives exits |
# | `SatisfactionIndex` | one stable signal from four correlated Likert items |
# | `ManagerStability` | a recent manager change is a well-documented attrition trigger |

# %% [markdown]
# ## Do they actually separate leavers from stayers?
#
# A feature with a reason still has to earn its place empirically.

# %%
comparison = pd.DataFrame({
    "stayed": X[y == 0][ENGINEERED_COLUMNS].mean(),
    "left": X[y == 1][ENGINEERED_COLUMNS].mean(),
})
comparison["difference_%"] = ((comparison["left"] - comparison["stayed"])
                              / comparison["stayed"].abs() * 100).round(1)
comparison.round(3)

# %%
fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for ax, col in zip(axes.ravel(), ENGINEERED_COLUMNS):
    data = [X.loc[y == 0, col].dropna(), X.loc[y == 1, col].dropna()]
    ax.boxplot(data, labels=["stayed", "left"], showfliers=False)
    ax.set_title(col, fontsize=10)
axes.ravel()[-1].axis("off")
fig.suptitle("Engineered features: leavers vs stayers", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# `IncomePerYearAtCompany` separates most sharply - leavers earn noticeably less per
# year of service. `TenureRatio` and `ManagerStability` are lower for leavers, matching
# the expectation that people who have invested less time here are likelier to go.

# %% [markdown]
# ## Leakage check
#
# Nothing may predict the target nearly perfectly. A feature that does is almost always
# a proxy for the answer that would not exist at prediction time.

# %%
leakage = check_leakage(X, y)
print("Any suspicious features?", bool(leakage["suspicious"].any()))
leakage.head(15)

# %% [markdown]
# The strongest single feature sits far below the threshold. No column is standing in
# for the answer - which is expected here, since the IBM extract contains no
# post-departure fields (exit dates, final pay, offboarding flags) of the kind that
# usually cause leakage in HR data.

# %% [markdown]
# ## Correlation among the engineered features

# %%
corr = X[ENGINEERED_COLUMNS].corr()
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(ENGINEERED_COLUMNS)))
ax.set_xticklabels(ENGINEERED_COLUMNS, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(ENGINEERED_COLUMNS)))
ax.set_yticklabels(ENGINEERED_COLUMNS, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Engineered feature correlation")
fig.tight_layout()
plt.show()

# %% [markdown]
# `TenureRatio` and `ManagerStability` are related - both are built from tenure - but
# not so tightly that one is redundant. Logistic regression is regularised by default,
# so this level of collinearity is handled rather than fatal.

# %% [markdown]
# ## Preprocessing
#
# Numeric columns are median-imputed and scaled; categoricals are mode-imputed and
# one-hot encoded with `handle_unknown="ignore"` so an unseen category at serving time
# cannot crash a live prediction.

# %%
numeric, categorical = split_column_types(X)
print(f"{len(numeric)} numeric, {len(categorical)} categorical")
print("categorical:", categorical)

preprocessor = build_preprocessor(X)
transformed = preprocessor.fit_transform(X)
print(f"\nfinal design matrix: {transformed.shape}")
print(f"{X.shape[1]} columns -> {transformed.shape[1]} after one-hot encoding")

# %% [markdown]
# ## Persist the engineered table
#
# Saved for inspection only. Training does not read this file - it applies the same
# engineering inside the pipeline, which is what keeps training and serving identical.

# %%
out = X.copy()
out.insert(0, "EmployeeID", df["EmployeeID"])
out["AttritionFlag"] = y
out.to_csv(PROCESSED_FILES["features"], index=False)
print(f"written: {PROCESSED_FILES['features'].name}  {out.shape}")
