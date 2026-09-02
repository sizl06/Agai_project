# %% [markdown]
# # 06 - Baseline Model
#
# Day 2, step 6. Logistic Regression, deliberately.
#
# The point of a baseline is to have something honest to compare against. It is fast,
# it is explainable, and it returns a real probability rather than just a label - which
# matters because the risk bands downstream need to rank employees, not just sort them
# into yes/no.
#
# No XGBoost yet. If a gradient-boosted model cannot beat this, that is worth knowing.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from app.ml.features import engineer_features
from app.utils.config import PROCESSED_FILES, RANDOM_STATE, TEST_SIZE
from pipelines.train_model import evaluate, make_pipeline, tune_threshold

pd.set_option("display.width", 200)

df = pd.read_csv(PROCESSED_FILES["attrition"])
y = df["AttritionFlag"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
print(f"train {len(X_train)}  test {len(X_test)}")
print(f"positive rate  train {y_train.mean():.4f}  test {y_test.mean():.4f}  (stratified)")

# %% [markdown]
# ## Fit
#
# `class_weight="balanced"` tells the model that the 16% minority matters as much as
# the 84% majority. Without it, the loss function is dominated by stayers and the model
# learns to predict "No" almost everywhere.

# %%
baseline = make_pipeline(
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    X_train,
)
baseline.fit(X_train, y_train)

probabilities = baseline.predict_proba(X_test)[:, 1]
print(f"ROC-AUC on the test set: {roc_auc_score(y_test, probabilities):.4f}")

# %% [markdown]
# ## Why accuracy is not the metric

# %%
naive_accuracy = (y_test == 0).mean()
model_accuracy = ((probabilities >= 0.5).astype(int) == y_test).mean()
print(f"Accuracy of predicting 'nobody leaves': {naive_accuracy:.4f}")
print(f"Accuracy of the baseline model       : {model_accuracy:.4f}")
print("\nThe useless model scores higher. Accuracy is meaningless on this target.")

# %% [markdown]
# ## The metrics that do mean something

# %%
print(classification_report(y_test, (probabilities >= 0.5).astype(int),
                            target_names=["stayed", "left"], digits=3))

# %% [markdown]
# ## Choosing the operating point
#
# 0.5 is an arbitrary cut. The build notes are explicit that missing a genuinely
# high-risk employee is the expensive mistake, so the threshold is tuned to maximise
# **F2**, which weights recall twice as heavily as precision.
#
# It is tuned on out-of-fold predictions over the training set only - never on the test
# set, which would make the reported numbers optimistic.

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_predict

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = cross_val_predict(baseline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
threshold, cv_f2 = tune_threshold(y_train.to_numpy(), oof)
print(f"tuned threshold: {threshold}   (out-of-fold F2 = {cv_f2})")

# %%
sweep = []
for t in np.arange(0.20, 0.85, 0.05):
    m = evaluate(y_test.to_numpy(), probabilities, round(float(t), 2))
    sweep.append({"threshold": round(float(t), 2), "precision": m["precision"],
                  "recall": m["recall"], "f1": m["f1"], "f2": m["f2"],
                  "flagged": int((probabilities >= t).sum())})
sweep_df = pd.DataFrame(sweep)
sweep_df

# %% [markdown]
# The trade-off is visible directly: lowering the cut catches more leavers and flags
# more people for an HR conversation that may not be needed. F2 encodes the judgement
# that the first error costs more than the second.

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(sweep_df["threshold"], sweep_df["precision"], marker="o", label="precision")
ax.plot(sweep_df["threshold"], sweep_df["recall"], marker="o", label="recall")
ax.plot(sweep_df["threshold"], sweep_df["f2"], marker="o", label="F2", linewidth=2.5)
ax.axvline(threshold, color="crimson", linestyle="--", label=f"tuned = {threshold}")
ax.set_xlabel("decision threshold")
ax.set_ylabel("score")
ax.set_title("Precision / recall trade-off on the test set")
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## Performance at the tuned threshold

# %%
final = evaluate(y_test.to_numpy(), probabilities, threshold)
pd.Series(final).to_frame("value")

# %%
predictions = (probabilities >= threshold).astype(int)
fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, display_labels=["stayed", "left"], cmap="Blues", ax=ax,
    colorbar=False)
ax.set_title(f"Confusion matrix at threshold {threshold}")
plt.show()

tn, fp, fn, tp = np.ravel(pd.crosstab(y_test, predictions).reindex(
    index=[0, 1], columns=[0, 1], fill_value=0).to_numpy())
print(f"caught {tp} of {tp + fn} actual leavers; missed {fn}")
print(f"flagged {fp} employees who in fact stayed")

# %% [markdown]
# ## ROC and precision-recall curves
#
# The PR curve is the more informative of the two on an imbalanced target, because it
# ignores the large, easy true-negative population.

# %%
fpr, tpr, _ = roc_curve(y_test, probabilities)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc_score(y_test, probabilities):.3f}")
ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
ax1.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(recall_curve, precision_curve, linewidth=2, color="darkorange")
ax2.axhline(y_test.mean(), color="k", linestyle="--", alpha=0.5,
            label=f"base rate = {y_test.mean():.3f}")
ax2.set(xlabel="recall", ylabel="precision", title="Precision-Recall curve")
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## What the linear model learned
#
# Coefficients are directly readable, which is exactly why a linear baseline is worth
# having before reaching for anything more complex.

# %%
feature_names = baseline.named_steps["preprocess"].get_feature_names_out()
coefficients = baseline.named_steps["model"].coef_[0]
coef_df = (pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
           .assign(magnitude=lambda d: d["coefficient"].abs())
           .sort_values("magnitude", ascending=False))

print("--- pushes attrition UP ---")
print(coef_df.nlargest(10, "coefficient")[["feature", "coefficient"]].to_string(index=False))
print("\n--- pushes attrition DOWN ---")
print(coef_df.nsmallest(10, "coefficient")[["feature", "coefficient"]].to_string(index=False))

# %%
top = coef_df.head(18).sort_values("coefficient")
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top["feature"], top["coefficient"],
        color=["crimson" if c > 0 else "steelblue" for c in top["coefficient"]])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Logistic regression coefficients (scaled features)")
ax.set_xlabel("coefficient - positive means higher attrition risk")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Reading the coefficients
#
# **Job role dominates.** `Laboratory Technician` and `Sales Representative` carry the
# largest positive coefficients, and `Research Director` the largest negative one -
# which lines up exactly with the raw attrition rates by role in notebook 01. The model
# has not discovered anything exotic; it has confirmed the pattern already visible in
# the data.
#
# **Overtime is the strongest single behavioural driver.** It is split across two dummy
# columns, `OverTime_Yes` (+0.83) and `OverTime_No` (-0.86), so its full effect is the
# span between them - roughly 1.69, larger than any individual role coefficient. Reading
# only the positive column would understate it. The same applies to business travel.
#
# `YearsSinceLastPromotion` and `MaritalStatus_Single` push risk up; longer average
# stints per company and manager stability push it down.
#
# One coefficient deserves a caveat: `MonthlyIncome` appears with a *positive* sign,
# which reads as "higher pay increases attrition risk" and is the opposite of what the
# raw data shows. It is an artefact of collinearity - `JobLevel`, `IncomePerJobLevel`
# and `IncomePerYearAtCompany` all encode pay, and the model distributes the effect
# across them. Individual coefficients in a collinear linear model are not reliable
# statements about isolated causes. That is precisely why notebook 08 uses SHAP, which
# attributes contributions per prediction rather than per column.
#
# The signals are coherent overall, so the baseline is behaving sensibly. Notebook 07
# checks whether the more complex models beat it.
