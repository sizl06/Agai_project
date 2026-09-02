"""Run the whole pipeline end to end.

    python -m pipelines.run_all

Order matters and is the same order the build notes prescribe: data must be
understood and cleaned before features, features before the model, the model before
the intelligence table. Each stage fails loudly rather than letting a later one
produce plausible nonsense from missing inputs.

Options:
    --skip-fetch     do not re-download raw data
    --retrain        train a new model version even if one exists
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_stage(name: str, command: list[str]) -> bool:
    print(f"\n{'=' * 72}\n  {name}\n{'=' * 72}")
    started = time.time()
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.time() - started

    if result.returncode != 0:
        print(f"\n[FAILED] {name} (exit {result.returncode}) after {elapsed:.1f}s")
        return False
    print(f"\n[OK] {name} in {elapsed:.1f}s")
    return True


def main() -> int:
    skip_fetch = "--skip-fetch" in sys.argv
    retrain = "--retrain" in sys.argv
    python = sys.executable

    from app.utils.config import RAW_FILES, latest_model_version

    stages: list[tuple[str, list[str]]] = []

    raw_missing = [k for k, p in RAW_FILES.items() if not p.exists()]
    if raw_missing and not skip_fetch:
        stages.append(("Fetch raw data", [python, "scripts/fetch_raw_data.py"]))
        stages.append(("Build derived sources", [python, "scripts/build_derived_sources.py"]))
        stages.append(("Build course catalogue", [python, "scripts/build_course_catalogue.py"]))
    elif raw_missing:
        print(f"Raw files missing and --skip-fetch given: {raw_missing}")
        return 1

    stages.append(("Clean data (step 03)", [python, "-m", "pipelines.clean_data"]))

    if retrain or latest_model_version() is None:
        stages.append(("Train and compare models (steps 06-09)",
                       [python, "-m", "pipelines.train_model"]))
    else:
        print(f"Model {latest_model_version()} already exists - pass --retrain to build a new one")

    stages.append(("Build intelligence table (steps 10-16)",
                   [python, "-m", "pipelines.build_intelligence"]))
    stages.append(("Drift report (step 25)", [python, "-m", "pipelines.monitor_drift"]))
    stages.append(("Performance report (steps 26-27)",
                   [python, "-m", "pipelines.monitor_performance"]))

    started = time.time()
    for name, command in stages:
        if not run_stage(name, command):
            return 1

    print(f"\n{'=' * 72}")
    print(f"  Pipeline complete in {time.time() - started:.1f}s")
    print(f"{'=' * 72}")
    print("\nNext:")
    print("  uvicorn app.main:app --reload            # API   -> http://127.0.0.1:8000/docs")
    print("  streamlit run frontend/streamlit_app.py  # UI    -> http://localhost:8501")
    print("  python scripts/build_notebooks.py        # rebuild the analysis notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
