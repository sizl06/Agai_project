# %% [markdown]
# # 09 - Model Versioning
#
# Day 2, step 9. Simple versioning first, before reaching for a whole platform.
#
# The requirement is traceability: given a prediction made three months ago, it must be
# possible to say exactly which model produced it and how that model scored. That needs
# a version folder and a metadata file, not MLflow.
#
# The build notes are explicit about the sequencing - MLflow comes *after* hand-written
# metadata starts to hurt, not before. The end of this notebook records what the
# migration trigger actually is.

# %%
import json

import joblib
import pandas as pd

from app.utils.config import (
    MODELS_DIR,
    PROCESSED_FILES,
    RISK_BAND_MULTIPLIERS,
    available_model_versions,
    latest_model_version,
    model_metadata,
    model_path,
    risk_level,
)

pd.set_option("display.width", 200)

print("versions on disk:", available_model_versions())
print("latest          :", latest_model_version())

# %% [markdown]
# ## The registry layout
#
# ```
# models/
# └── v1/
#     ├── attrition_pipeline.joblib
#     └── metadata.json
# ```
#
# A retrain writes `v2/` beside it rather than overwriting. Nothing is ever lost, and
# rolling back is a matter of pointing at the previous folder.

# %%
for version in available_model_versions():
    folder = MODELS_DIR / version
    print(f"{version}/")
    for f in sorted(folder.iterdir()):
        print(f"    {f.name:32s} {f.stat().st_size / 1024:>8.1f} KB")

# %% [markdown]
# ## Metadata

# %%
metadata = model_metadata()
print(json.dumps(metadata, indent=2))

# %% [markdown]
# Two fields matter more than the rest.
#
# `decision_threshold` is not decoration. It was tuned on out-of-fold predictions in
# notebook 07, and the API must apply *this* model's threshold - a future model
# calibrated differently would need a different cut, and using a stale one would
# silently change who gets flagged.
#
# `random_state` is what makes the run reproducible. Without it, "retrain and compare"
# has no baseline to compare against.

# %% [markdown]
# ## Risk bands derive from the threshold
#
# The bands are multiples of the model's own threshold rather than fixed probabilities.
# The models are trained with class weighting, so their output is calibrated to a
# balanced world rather than to the true 16% base rate - a literal 0.50 does not mean
# "more likely than not" here.

# %%
threshold = metadata["decision_threshold"]
print(f"model threshold: {threshold}")
for band, multiplier in RISK_BAND_MULTIPLIERS.items():
    print(f"  {band:6s} >= {threshold} x {multiplier} = {threshold * multiplier:.3f}")

# %%
for p in [0.05, 0.20, 0.35, 0.40, 0.55, 0.60, 0.85]:
    print(f"  probability {p:.2f} -> {risk_level(p, threshold)}")

# %% [markdown]
# ## The artifact is self-contained
#
# The saved pipeline holds feature engineering, preprocessing and the model together.
# It takes a raw employee record and does the rest itself, so the API cannot compute a
# feature differently from training.

# %%
pipeline = joblib.load(model_path())
for name, step in pipeline.named_steps.items():
    print(f"  {name:12s} {type(step).__name__}")

# %%
df = pd.read_csv(PROCESSED_FILES["attrition"])
raw_record = df.drop(columns=["Attrition", "AttritionFlag"]).head(1)
print("input columns:", raw_record.shape[1], "(raw, unengineered)")
print("predicted probability:", round(float(pipeline.predict_proba(raw_record)[0, 1]), 4))

# %% [markdown]
# ## Traceability in practice

# %%
sample = df.drop(columns=["Attrition", "AttritionFlag"]).head(5)
probabilities = pipeline.predict_proba(sample)[:, 1]

traced = pd.DataFrame({
    "EmployeeID": sample["EmployeeID"],
    "probability": probabilities.round(4),
    "risk_level": [risk_level(p, threshold) for p in probabilities],
    "model_version": metadata["version"],
    "algorithm": metadata["algorithm"],
    "threshold": threshold,
})
traced

# %% [markdown]
# Every row carries the model that produced it. Step 21 writes these to
# `data/predictions/` so the record survives beyond the API response.

# %% [markdown]
# ## Comparing versions
#
# With one version this is trivial, but the function is what makes a retrain
# reviewable rather than a leap of faith.

# %%
def version_table() -> pd.DataFrame:
    rows = []
    for version in available_model_versions():
        meta_file = MODELS_DIR / version / "metadata.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        rows.append({
            "version": version,
            "algorithm": meta.get("algorithm"),
            "trained": meta.get("training_date"),
            "roc_auc": meta.get("roc_auc"),
            "pr_auc": meta.get("pr_auc"),
            "recall": meta.get("recall"),
            "f2": meta.get("f2"),
            "threshold": meta.get("decision_threshold"),
        })
    return pd.DataFrame(rows)


version_table()

# %% [markdown]
# ## When to move to MLflow
#
# Not yet. The trigger is specific rather than aspirational - adopt MLflow when any of
# these becomes true:
#
# 1. **More than a handful of versions.** Hand-comparing JSON stops scaling at roughly
#    ten.
# 2. **Hyperparameter search.** Dozens of runs per training session need automatic
#    parameter and metric capture; writing that by hand is where mistakes enter.
# 3. **More than one person training.** A shared tracking server beats a folder that
#    only exists on one laptop.
# 4. **Anything approaching real deployment.** Stage transitions and a model registry
#    are what MLflow is genuinely better at than a directory.
#
# Until one of those holds, this layout gives the same traceability with no server to
# run. The migration path is unblocked either way: `metadata.json` already records
# exactly the fields `mlflow.log_metric` and `mlflow.log_param` would take.

# %% [markdown]
# ## Day 2 summary
#
# | Step | Outcome |
# |---|---|
# | 05 Feature engineering | 7 engineered features, each with a stated reason; no leakage found |
# | 06 Baseline | Logistic Regression, threshold tuned to F2 on out-of-fold predictions |
# | 07 Comparison | Logistic Regression beat Random Forest and XGBoost on ROC-AUC and PR-AUC |
# | 08 Explainability | SHAP reconstructs the model exactly; contributions aggregated per raw column |
# | 09 Versioning | `models/v1/` with metadata; MLflow deferred with a stated trigger |
#
# Day 3 moves to the part that does not need a model at all - engagement, skill gaps and
# recommendations.
