"""Load the trained pipeline once and keep it in memory.

Loading a joblib artifact per request would add tens of milliseconds and, worse,
rebuild the SHAP background on every explanation. The loader caches per version so
the API can still serve an older model on request.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from app.utils.config import (
    DEFAULT_DECISION_THRESHOLD,
    PROCESSED_FILES,
    latest_model_version,
    model_metadata,
    model_path,
)
from app.utils.logger import get_logger

log = get_logger("model_loader")

# Rows used as the SHAP reference distribution. Enough to be representative,
# small enough to explain a single prediction quickly.
BACKGROUND_SAMPLE_SIZE = 200


@lru_cache(maxsize=4)
def load_pipeline(version: str | None = None) -> Pipeline:
    path = model_path(version)
    pipeline = joblib.load(path)
    log.info("Model %s loaded from %s", version or latest_model_version(), path.name)
    return pipeline


@lru_cache(maxsize=4)
def load_metadata(version: str | None = None) -> dict:
    return model_metadata(version)


def decision_threshold(version: str | None = None) -> float:
    """This model's tuned cut. Never assume 0.5 - see notebook 09."""
    return float(load_metadata(version).get("decision_threshold", DEFAULT_DECISION_THRESHOLD))


@lru_cache(maxsize=1)
def load_background() -> pd.DataFrame:
    """A fixed sample of training-like rows for SHAP to explain against."""
    df = pd.read_csv(PROCESSED_FILES["attrition"])
    if len(df) > BACKGROUND_SAMPLE_SIZE:
        df = df.sample(BACKGROUND_SAMPLE_SIZE, random_state=42)
    return df.reset_index(drop=True)


def active_version() -> str:
    version = latest_model_version()
    if version is None:
        raise FileNotFoundError("No trained model. Run: python -m pipelines.train_model")
    return version
