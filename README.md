# Enterprise HR AI — Workforce Intelligence & Upskilling Platform

Given employee data, answer three questions: **who is likely to leave**, **where the
organisation's skill gaps are**, and **what each employee should learn next**.

Built following `HR_AI_Project_Build_Notes.docx`, a day-by-day plan of 29 steps across
four days plus a hardening phase. All 29 are implemented. Where the data contradicted
the plan, the plan was followed *and* the contradiction documented — those cases are
listed under [Where the results differ from the plan](#where-the-results-differ-from-the-plan).

---

## Read this first

Two things about the data, stated up front because every number downstream depends on
them.

**Three of the five datasets are real.** `employee_attrition.csv` is the IBM HR
Analytics extract (1,470 employees) from IBM's own repository; `occupation_data.csv`,
`essential_skills.csv` and `software_skills.csv` come from O*NET database 31.0.

**Two are derived, and had to be.** The Kaggle engagement dataset requires API
credentials, and no public dataset records per-employee skill inventories.

* `hr_performance_engagement.csv` is a **transformation of real IBM columns** — the five
  Likert satisfaction/involvement items plus performance rating, rescaled to 0–100. No
  values are invented. An unrelated Kaggle HR file was rejected because it describes a
  different population, which would have made the 1:1 join the plan specifies
  meaningless.
* `employee_current_skills.csv` is **synthesised**, which the build notes explicitly
  sanction as the fallback. The skill vocabulary and role requirements are real O*NET
  data; *which employee holds which skill* is modelled from real attributes (education,
  tenure, seniority, training) under a fixed seed.

**What this means for the numbers.** The attrition model is trained on real labelled
data and its metrics are real. The skill-gap engine is a real mechanism producing
*illustrative* totals — "879 employees are missing Active Learning" demonstrates that
the roll-up works; it is not a measurement of a real workforce. Point a real skills
inventory at `data/raw/employee_current_skills.csv` and the same code produces real
numbers. See `scripts/build_derived_sources.py` and notebook 12.

---

## Results

| | |
|---|---|
| **Model** | Logistic Regression (beat Random Forest and XGBoost) |
| **ROC-AUC** | 0.8006 held out |
| **PR-AUC** | 0.5429 (base rate 0.16) |
| **Recall / Precision** | 0.553 / 0.426 at the tuned threshold of 0.58 |
| **Employees scored** | 1,470 |
| **Skill gaps found** | ~15,500 employee/skill pairs across 52 skills |
| **Tests** | 144 passing |

Attrition is 16% positive, so **accuracy is never reported** — predicting "nobody
leaves" scores 84% while being useless.

---

## Quick start

```bash
pip install -r requirements-dev.txt   # full stack; see note below

python scripts/fetch_raw_data.py           # download the real datasets
python scripts/build_derived_sources.py    # build the two derived tables
python scripts/build_course_catalogue.py   # training catalogue
python -m pipelines.run_all                # clean -> train -> intelligence -> monitor
```

Or in one step — `run_all` fetches whatever is missing:

```bash
python -m pipelines.run_all
```

Then:

```bash
uvicorn app.main:app --reload             # API       http://127.0.0.1:8000/docs
streamlit run frontend/streamlit_app.py   # dashboard http://localhost:8501
pytest                                    # 144 tests
python scripts/build_notebooks.py         # rebuild and execute all 16 notebooks
```

### Which requirements file

| File | Installs | Used by |
|---|---|---|
| `requirements.txt` | 4 packages — streamlit, pandas, plotly, requests | the dashboard; **Streamlit Community Cloud installs this** |
| `requirements-api.txt` | model + FastAPI stack | the backend and its Docker image |
| `requirements-dev.txt` | everything, incl. notebooks and tests | local development |

The dashboard genuinely imports none of scikit-learn, xgboost, shap or fastapi — it
reads the pre-built CSVs. Keeping the root file slim is what makes a cloud deploy fast
and reliable.

Docker:

```bash
python -m pipelines.run_all   # artifacts must exist on the host first
docker compose up --build
```

---

## How it works

```
                         USER
                          |
                 Streamlit UI (8501)
                          |
                   FastAPI (8000)
                          |
      +-------------------+-------------------+
      |                   |                   |
 ML Prediction      Skill Engine          Analytics
 (models/vN)        (set subtraction)     (aggregation)
      +-------------------+-------------------+
                          |
              Employee Intelligence table
                          |
                 Logging + Monitoring
                          |
                   Model Registry
```

Three subsystems converge on one table:

1. **Attrition prediction** — a supervised model emitting a *probability*, not a label,
   because the risk bands need to rank people.
2. **Engagement analytics** — pandas aggregation, no ML. Nothing to predict yet.
3. **Skill gaps and recommendations** — set subtraction (`required - held`), weighted by
   O*NET importance, then mapped to courses.

The **Employee Intelligence table** (`data/processed/employee_intelligence.csv`) is the
business output — one row per employee with risk, engagement, gaps and a
recommendation. The dashboard is a view onto it.

### Data model

```
EMPLOYEE (employee_attrition)
   |
   +-- EmployeeID ----- ENGAGEMENT              1:1
   |
   +-- EmployeeID ----- EMPLOYEE SKILLS         1:many
   |
   +-- JobRole -------- ROLE REQUIREMENTS       many:many
                             |
                             +-- ONET_SOC_Code -- OCCUPATION MASTER   many:1
```

Employees connect to skills *through their role*, never directly. Every join was
verified by key overlap and cardinality before being accepted — see
`docs/data_relationships.md`.

---

## Layout

```
app/                     the service
├── main.py              FastAPI app, logging middleware, health
├── api/                 routes: attrition, dashboard, skills
├── services/            business logic (engagement, skill gaps, recommendations, logging)
├── ml/                  features, model loader, predictor, SHAP explainer
├── validation/          dataframe rules + Pydantic request models
└── utils/               config and logging

pipelines/               batch jobs
├── clean_data.py        step 03
├── train_model.py       steps 06-09
├── build_intelligence.py steps 10-16
├── monitor_drift.py     step 25
├── monitor_performance.py steps 26-27
└── run_all.py           everything, in order

notebooks/               16 executed analysis notebooks
└── _src/                their source, as `# %%` percent-format .py files

frontend/streamlit_app.py  the dashboard
scripts/                 data fetch, derived sources, catalogue, notebook builder
tests/                   144 tests
docs/                    relationships, retraining strategy, deployment
models/vN/               pipeline .joblib + metadata.json
```

Notebooks are authored as `.py` files in `notebooks/_src/` and converted+executed by
`scripts/build_notebooks.py`. The `.py` files diff cleanly in review; the generated
`.ipynb` carries the outputs.

---

## The notebooks

| # | Notebook | What it establishes |
|---|---|---|
| 01 | Data understanding | 16% attrition; key is `EmployeeNumber` not `EmployeeID`; 3 constant columns |
| 02 | Data validation | rules in `app/validation/`, proven against deliberately corrupted frames |
| 03 | Data cleaning | processed copies; skill-name canonicalisation; outliers kept, not clipped |
| 04 | Data relationships | every join verified by cardinality and overlap |
| 05 | Feature engineering | 7 engineered features, each with a stated reason; no leakage |
| 06 | Baseline model | Logistic Regression; threshold tuned to F2 out-of-fold |
| 07 | Model comparison | LogReg beats RF and XGBoost on ROC-AUC and PR-AUC |
| 08 | Explainability | SHAP reconstructs the model exactly (error 2.7e-15) |
| 09 | Model versioning | `models/v1/` + metadata; MLflow deferred with a stated trigger |
| 10 | Engagement | department/role breakdowns; disengaged high performers |
| 11 | Role intelligence | 9 job roles mapped to O*NET occupations |
| 12 | Employee skills | confirmed absent from source data; controlled table built |
| 13 | Skill gap engine | set subtraction, with the row-explosion trap tested |
| 14 | Organisation gaps | the spec's severity rule, and why it fails at this scale |
| 15 | Recommendations | tag matching; the semantic fallback's limits demonstrated |
| 16 | Intelligence table | 1,470 rows, one per employee |

---

## Where the results differ from the plan

The build notes were followed. In five places the data disagreed with them, and the
disagreement is the useful part.

**1. The join key is not `EmployeeID`.** The notes assume it; the real IBM extract ships
`EmployeeNumber`, and IDs run to 2068 across 1,470 rows — sparse, so never usable as a
row index. Renamed during cleaning. *(notebook 01)*

**2. XGBoost lost.** The notes expected it to win, on the sound general principle that
gradient boosting dominates tabular problems. On 1,470 rows with 237 positives and a
near-linear signal, both tree models overfit and regularised Logistic Regression ranked
better on ROC-AUC *and* PR-AUC. This is exactly why the notes insist on a boring
baseline first. *(notebook 07)*

**3. `shap.TreeExplainer` would have crashed.** It only works on tree models.
`app/ml/explainer.py` selects the explainer from the fitted estimator instead.
*(notebook 08)*

**4. The severity rule does not discriminate here.** The notes specify absolute bands —
100+ employees missing a skill is HIGH. At 1,470 employees that marks **36 of 52 skills**
HIGH, which cannot guide a budget. Implemented as specified, and a proportional rule
(40% / 20% of workforce) is reported alongside it, narrowing the critical list to nine.
Both columns ship. *(notebook 14)*

**5. The semantic matcher cannot do what the notes describe.** v2 was specified as
sentence-transformer embeddings so that "MLOps" matches "Deploying and Monitoring
Machine Learning Systems". The TF-IDF implementation here **cannot** — it scores 0.018
on exactly that case. Notebook 15 demonstrates the failure rather than hiding it, and
documents the four-line swap. It is not installed because the catalogue is fully tagged,
so the fallback never fires in production. *(notebook 15)*

Two further decisions worth flagging:

* **The decision threshold is tuned, not left at 0.5**, and the risk bands are multiples
  of *that* threshold. Class-weighted models emit probabilities calibrated to a balanced
  world, so a literal 0.5 does not mean "more likely than not".
* **Engagement is never a model feature.** It is a deterministic function of five columns
  the model already uses, so feeding it back would be circular. It stays analytics-only.

---

## Design decisions

**Feature engineering lives inside the sklearn pipeline.** The saved `.joblib` takes a
raw employee record and does everything itself, so the API cannot compute a feature
differently from training.

**`EmployeeID` is never a feature.** A tree will happily carve an arbitrary identifier
into "risky" ranges that mean nothing and fail on new employees.

**Selection is separate from thresholding.** Models are ranked on ROC-AUC and PR-AUC
(threshold-independent); the operating point is chosen afterwards. Ranking on F2 would
let a model win by cutting low enough to flag most of the workforce — Random Forest
scored the best F2 of the three at 81% recall and 31% precision, while ranking worst.

**SHAP contributions aggregate to the raw column.** One-hot encoding splits `OverTime`
across two columns, and ranking them separately let one variable occupy two of the
three slots an HR manager sees — while displaying `OverTime_No: increases risk` for
someone who *does* work overtime. *(notebook 08)*

**Outliers are reported, not removed.** The `MonthlyIncome` tail is directors, not
errors. Clipping would delete the population where a wrong call is most expensive.

**Aggregate before joining.** Skill gaps (~15,500 rows) and recommendations (~4,400)
are collapsed to one row per employee before touching the spine.
`build_intelligence.build()` raises if the row count changes.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict/attrition` | score one employee (Pydantic-validated, 422 on bad input) |
| `GET /predict/attrition/{id}` | score a known employee |
| `GET /predict/model/info` | active version, algorithm, metrics, threshold |
| `GET /dashboard/summary` | KPI cards |
| `GET /dashboard/attrition-by-department` | chart data |
| `GET /dashboard/skill-gaps` | organisation-wide gaps |
| `GET /dashboard/recommendations` | per-employee recommendations |
| `GET /employees/{id}` | full intelligence record |
| `GET /health` | readiness, including artifact availability |

Every prediction returns its `model_version` and the top three contributing factors, so
a flag always arrives with a reason.

---

## Monitoring

```bash
python -m pipelines.monitor_drift        # PSI + KS per feature, and prediction drift
python -m pipelines.monitor_performance  # live metrics vs release, retraining decision
```

Drift uses PSI (bands 0.10 / 0.25) alongside a KS test. Run against the training data
it returns PSI 0.0 across every feature — a self-test confirming the metric is correct
— and shifting age by +12 years, the notes' own example, produces PSI 3.13.

The retraining rule is written down in advance and implemented in `should_retrain()`:
drift PSI > 0.25, **or** F1 below 0.45, **or** F1 down more than 15% from release,
**or** the model older than 182 days. Full reasoning in `docs/retraining_strategy.md`,
including how the attrition outcome lag is handled.

---

## Limitations

* **Dashboard figures are in-sample.** Every employee shown was in the training data.
  The honest estimate is the held-out ROC-AUC of 0.80.
* **Skill-gap totals are illustrative**, for the reason given at the top.
* **The course catalogue is synthetic** — plausible placeholders for an LMS feed.
* **No authentication.** Every endpoint is open; `GET /employees/{id}` exposes an
  individual's risk score to anyone who can reach the port.
* **Fairness has not been assessed.** The model uses `Gender`, `MaritalStatus` and
  `Age`. Before any real use these need a documented fairness review and probably
  removal. See `docs/deployment.md`.
* **The Docker images are not build-tested.** `docker compose config` validates and
  every copied path exists, but the Docker daemon was not running here, so
  `docker build` never ran.
* **No CI/CD, no cloud deployment, no MLflow.** Deferred deliberately, with the trigger
  for adopting each one written down rather than left vague.

A risk score is a prompt for a retention conversation, not evidence about a person.

---

## Not part of this project

`Sample - Superstore.csv` and the analysis questions at the bottom of `output.md` are a
separate retail-EDA exercise that happens to live in this folder. Nothing here reads
them, and they should not be joined to the HR data.
