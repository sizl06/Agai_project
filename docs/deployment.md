# Deployment

## Architecture

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
      |                   |                   |
      +-------------------+-------------------+
                          |
              Employee Intelligence table
              (data/processed/employee_intelligence.csv)
                          |
                 Logging + Monitoring
              (logs/, data/predictions/, reports/)
                          |
                   Model Registry
                      (models/)
```

Two processes, one shared filesystem. The Streamlit app talks to the API over HTTP and
falls back to reading the processed files directly if the API is unreachable, so a
backend restart degrades the dashboard rather than breaking it.

## Local

```bash
pip install -r requirements.txt
python -m pipelines.run_all

uvicorn app.main:app --reload            # http://127.0.0.1:8000/docs
streamlit run frontend/streamlit_app.py  # http://localhost:8501
```

## Docker

```bash
python -m pipelines.run_all   # artifacts must exist on the host first
docker compose up --build
```

Data and models are **bind-mounted, not baked into the images**. That is deliberate:

* a retrain does not require an image rebuild
* it is always clear which model a running container is serving
* the training data never ends up inside a distributed artifact

`data/raw`, `data/processed`, `data/external` and `models` mount read-only, so the API
cannot modify training data or a released model. Only `data/predictions`, `logs` and
`reports` are writable.

Both services run as a non-root user (uid 1000) and expose a `HEALTHCHECK`; the
dashboard waits on `service_healthy` for the API rather than racing it at startup.

## Configuration

| Variable | Default | Used by |
|---|---|---|
| `HR_AI_API_URL` | `http://127.0.0.1:8000` | Streamlit - set to `http://api:8000` inside compose |
| `PYTHONPATH` | `/app` | both images |

Paths and thresholds live in `app/utils/config.py` rather than in environment
variables, because they are decisions about the model rather than about the
environment, and they belong in version control where a change is reviewable.

## Health and readiness

`GET /health` reports `healthy` only when both a trained model and the intelligence
table are present:

```json
{
  "status": "healthy",
  "model_version": "v1",
  "model_available": true,
  "intelligence_table_available": true
}
```

If either artifact is missing the service reports `degraded` and the affected endpoints
return **503 with an actionable message** naming the command to run, rather than a 500.
That distinction matters for a load balancer: a degraded instance is misconfigured, not
crashed.

The model is loaded at startup rather than on first request, so a broken artifact shows
up in the startup log instead of as a slow, confusing first failure.

## Before running this against real employee data

This is a student project and has not been through the review that a system making
inferences about real people requires. At minimum, the following would need to be
addressed first.

**Access control.** There is none. Every endpoint is unauthenticated, and
`GET /employees/{id}` returns an individual's attrition risk, engagement score and
skill gaps to anyone who can reach the port. This needs authentication, and
authorisation narrow enough that a manager sees only their own reports.

**CORS.** `allow_origins=["*"]` in `app/main.py` is set for local development. A real
deployment lists the dashboard's origin explicitly.

**Transport and storage.** HTTPS termination, encryption at rest for
`data/processed/` and `data/predictions/`, and a retention policy. Prediction logs
accumulate indefinitely today.

**Fairness review.** The model uses `Gender`, `MaritalStatus` and `Age` as features.
They carry genuine signal in this dataset, but a system that flags people for
management attention on the basis of protected characteristics needs a documented
fairness assessment - per-group error rates at minimum - and probably needs those
features removed. This has not been done here.

**How predictions are used.** A risk score is a prompt for a retention conversation,
not evidence about an individual. Using one in a promotion, pay or termination decision
would be both unfair and, in many jurisdictions, unlawful. The `top_factors` field
exists so that a flag always arrives with a reason attached.

**Consent and disclosure.** Employees should know this exists and what it does.

## Scaling

Current shape: single instance, no shared state beyond the filesystem.

| Growth | What breaks first | Change |
|---|---|---|
| More requests | nothing - the model is in-process and stateless | run several API replicas behind a load balancer |
| More employees | CSV reads on every dashboard request | move the intelligence table to Postgres |
| Larger population | `build_intelligence` is single-process | batch it, or move the joins into SQL |
| More models | joblib files in folders | a real model registry (MLflow) |

The API is already stateless apart from reads, so horizontal scaling needs no code
change - only shared storage for `models/` and `data/processed/`.

## What is not done

Honest list, so nobody assumes otherwise:

* **The Docker images have not been build-tested.** `docker compose config` validates
  the compose file, and every path the Dockerfiles copy exists, but the Docker daemon
  was not running in the environment this was built in, so `docker build` never ran.
  Treat the Dockerfiles as reviewed-but-unverified and expect to fix something on the
  first build.
* no CI/CD pipeline
* no cloud deployment (the compose setup runs locally; nothing is provisioned)
* no Kubernetes manifests
* no secrets management (nothing currently needs a secret)
* no rate limiting
* no observability beyond application logs and the JSON reports in `reports/`
