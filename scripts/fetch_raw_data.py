"""Download the raw datasets.

Three of the five source files are fetched from their public origins:

  employee_attrition.csv  IBM's own repo (IBM/employee-attrition-aif360)
  occupation_data.csv     O*NET database 31.0
  essential_skills.csv    O*NET database 31.0
  software_skills.csv     O*NET database 31.0

The remaining two are then derived by `scripts/build_derived_sources.py` - see that
file for why they are derived rather than downloaded.

    python scripts/fetch_raw_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.config import RAW_DIR  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; hr-ai-project/1.0)"}
ONET_BASE = "https://www.onetcenter.org/dl_files/database/db_31_0_csv/"

DOWNLOADS = {
    "employee_attrition.csv": (
        "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
    ),
    "occupation_data.csv": ONET_BASE + "occupation_data.csv",
    "essential_skills.csv": ONET_BASE + "essential_skills.csv",
    "software_skills.csv": ONET_BASE + "software_skills.csv",
}

MIN_BYTES = 1_000


def download(filename: str, url: str, force: bool = False) -> bool:
    target = RAW_DIR / filename
    if target.exists() and not force:
        print(f"SKIP  {filename:28s} already present ({target.stat().st_size:,} bytes)")
        return True

    try:
        response = requests.get(url, headers=HEADERS, timeout=120)
    except requests.RequestException as exc:
        print(f"FAIL  {filename:28s} {type(exc).__name__}: {exc}")
        return False

    if response.status_code != 200 or len(response.content) < MIN_BYTES:
        print(f"FAIL  {filename:28s} HTTP {response.status_code}, "
              f"{len(response.content):,} bytes")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    print(f"OK    {filename:28s} {len(response.content):>12,} bytes")
    return True


def main() -> int:
    force = "--force" in sys.argv
    print(f"Downloading to {RAW_DIR}\n")

    results = [download(name, url, force=force) for name, url in DOWNLOADS.items()]
    failed = results.count(False)

    if failed:
        print(f"\n{failed} download(s) failed.")
        print("O*NET publishes new database versions periodically; if a URL 404s, check")
        print("https://www.onetcenter.org/database.html for the current db_*_csv path.")
        return 1

    print(f"\nAll {len(results)} files downloaded.")
    print("Next: python scripts/build_derived_sources.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
