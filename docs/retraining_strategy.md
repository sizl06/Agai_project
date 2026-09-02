# Retraining Strategy

The rule is written down in advance so that retraining is not a judgement call made
under pressure, after someone notices the dashboard looks wrong.

Implemented in `pipelines/monitor_performance.py::should_retrain()`, which returns the
decision *and* the reasons, so it is auditable.

## The rule

```
IF   feature drift PSI > 0.25
OR   live F1 < 0.45
OR   live F1 has fallen more than 15% below its released value
OR   the model is more than 182 days old
THEN retrain
```

| Trigger | Threshold | Constant | Why this value |
|---|---|---|---|
| Feature drift | PSI > 0.25 | `DRIFT_PSI_THRESHOLD` | The standard "significant shift" band. Below 0.10 is stable; 0.10-0.25 is worth watching but not acting on. |
| Absolute F1 floor | < 0.45 | `F1_FLOOR` | v1 released at F1 0.4815. Below 0.45 the model is close to being no better than a simple heuristic. |
| Relative F1 drop | > 15% | `F1_RELATIVE_DROP` | Catches decay from a *high* starting point that the absolute floor would miss. |
| Model age | 182 days | `MAX_MODEL_AGE_DAYS` | ~6 months, per the build notes. A calendar backstop for slow drift no single metric catches. |

Age alone is a sufficient trigger. The other three need data that may not have arrived
yet; time always passes.

## Why performance is not checked on a fixed schedule

Retraining on a calendar alone wastes effort when nothing has changed and is too slow
when something has. The calendar trigger exists only as a backstop.

The order the triggers actually fire in matters:

1. **Feature drift** fires first. It needs no outcomes at all - only incoming data.
2. **Prediction drift** fires next. The distribution of predicted probabilities moves
   before anyone knows who left. Tracked by `pipelines/monitor_drift.py` from the
   prediction log.
3. **Performance decay** fires last, because it needs ground truth.

## The outcome lag problem

Attrition outcomes arrive late. An employee flagged today may leave in eight months.

This has a specific consequence that `monitor_performance.py` handles explicitly: recent
predictions have no ground truth yet, and scoring them as wrong would make every model
look like it was decaying. Only predictions older than `OUTCOME_LAG_DAYS` (180) are
evaluated. Everything more recent is excluded from the metrics rather than counted
against the model.

This is also why feature and prediction drift carry the early-warning load - they are
the only signals available inside the lag window.

## Minimum sample sizes

An alerting system that cries wolf on its first day gets ignored, so:

* Prediction drift PSI is not computed below `MIN_PREDICTIONS_FOR_DRIFT` (100). On a
  dozen rows PSI is noise - a handful of high-risk lookups will report "significant
  drift" every time.
* Live performance metrics need at least a few hundred resolved outcomes before a 15%
  F1 move means anything on a target with a 16% positive rate.

## The retraining lifecycle

```
new data
   -> validation            app/validation/  - reject bad data before it trains anything
   -> cleaning              pipelines/clean_data.py
   -> training              pipelines/train_model.py  (all three candidates, same split)
   -> evaluation            held-out test set, ROC-AUC + PR-AUC
   -> versioning            models/vN/ with metadata.json
   -> approval              a human compares vN against vN-1 before promotion
   -> deploy                the API serves the latest version by default
```

Nothing is overwritten. A retrain writes `models/v2/` beside `models/v1/`, so a
rollback is a matter of pointing at the previous folder.

## Approval is deliberately manual

The pipeline does not auto-promote. `select_winner()` picks the best of the three
candidate algorithms within one training run, but promoting that run to production is a
human decision, because a model can improve on ROC-AUC while becoming worse in ways the
metric does not capture - a different threshold that flags three times as many people,
or a shift in which departments get flagged.

Before promoting `vN`, compare against `vN-1` on:

* ROC-AUC and PR-AUC on the held-out set (must not regress materially)
* the tuned threshold, and how many employees it flags
* the SHAP global importance ranking - a large reordering means the model is now
  reasoning differently and deserves scrutiny
* the risk-band distribution across departments, checked for a group that has suddenly
  become over-flagged

## Running the check

```bash
python -m pipelines.monitor_drift          # feature and prediction drift
python -m pipelines.monitor_performance    # live metrics + the retraining decision
python -m pipelines.train_model            # writes the next version if the answer is yes
```

Reports are written to `reports/` with the date in the filename, so the history of
decisions is on disk rather than in someone's memory.

## What is not automated yet, and why

* **Scheduling.** These run on demand. A cron entry or an Airflow DAG is the obvious
  next step, but scheduling something that has never been run in anger just produces
  unread reports.
* **Alerting.** No email or Slack integration. The thresholds are defined and the
  decision is machine-readable, so wiring an alert to `should_retrain()["retrain"]`
  is small - it is deferred until someone is actually on the receiving end.
* **Evidently AI.** The PSI and KS implementations here are ~50 lines with no extra
  dependency. Evidently is worth adopting when drift reports need to be shared and
  browsed rather than read from JSON.
