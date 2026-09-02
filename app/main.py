"""FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload

Interactive docs at http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import attrition, dashboard, skills
from app.utils.config import PROCESSED_FILES, latest_model_version
from app.utils.logger import get_logger

log = get_logger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Log what the service found at startup, and warm the model.

    Loading the model here rather than on first request means a broken artifact
    shows up in the startup log instead of as a slow, confusing first 500.
    """
    log.info("Starting Enterprise HR AI API")

    version = latest_model_version()
    if version is None:
        log.warning("No trained model found - prediction endpoints will return 503. "
                    "Run: python -m pipelines.train_model")
    else:
        try:
            from app.ml.model_loader import load_pipeline
            load_pipeline()
            log.info("Model %s loaded", version)
        except Exception as exc:  # noqa: BLE001
            log.error("Model %s failed to load: %s", version, exc)

    if PROCESSED_FILES["intelligence"].exists():
        log.info("Intelligence table found: %s", PROCESSED_FILES["intelligence"].name)
    else:
        log.warning("No intelligence table - dashboard endpoints will return 503. "
                    "Run: python -m pipelines.build_intelligence")

    log.info("Startup complete")
    yield
    log.info("Shutting down")


app = FastAPI(
    title="Enterprise HR AI - Workforce Intelligence Platform",
    description=(
        "Predicts attrition risk, tracks engagement, finds organisational skill gaps "
        "and recommends what each employee should learn next."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Wide open because the Streamlit front end runs on a different port locally. A real
# deployment would list the front end's origin explicitly instead.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    log.info("%s %s -> %d (%.1f ms)",
             request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak a stack trace to a caller; always leave one in the log."""
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(attrition.router)
app.include_router(dashboard.router)
app.include_router(skills.router)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "Enterprise HR AI - Workforce Intelligence Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "POST /predict/attrition",
            "GET  /predict/attrition/{employee_id}",
            "GET  /predict/model/info",
            "GET  /dashboard/summary",
            "GET  /dashboard/attrition-by-department",
            "GET  /dashboard/skill-gaps",
            "GET  /dashboard/recommendations",
            "GET  /employees/{employee_id}",
        ],
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Readiness, including whether the artifacts the service needs are present."""
    model_version = latest_model_version()
    intelligence_ready = PROCESSED_FILES["intelligence"].exists()
    return {
        "status": "healthy" if (model_version and intelligence_ready) else "degraded",
        "model_version": model_version,
        "model_available": model_version is not None,
        "intelligence_table_available": intelligence_ready,
    }
