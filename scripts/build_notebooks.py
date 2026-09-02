"""Build and execute the analysis notebooks.

Notebooks are authored as plain Python in notebooks/_src/ using `# %%` cell markers
(the jupytext "percent" format), then converted to .ipynb and executed here. The .py
files are the source of truth: they diff cleanly in review, while the generated
.ipynb carries the outputs a reader wants to see.

Usage:
    python scripts/build_notebooks.py              # build + execute all
    python scripts/build_notebooks.py 01 02        # only notebooks starting 01 / 02
    python scripts/build_notebooks.py --no-exec    # convert without executing
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    # nbclient drives a zmq kernel; the default Proactor loop on Windows makes it
    # spin up an extra selector thread and emit a warning on every notebook.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "notebooks" / "_src"
OUT_DIR = ROOT / "notebooks"


def parse_percent_script(text: str) -> list:
    """Split a `# %%` percent-format script into notebook cells.

    `# %% [markdown]` starts a markdown cell whose body is written as `#` comments.
    `# %%` starts a code cell.
    """
    cells = []
    kind = "code"
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        body = "\n".join(buffer).strip("\n")
        if not body.strip():
            return
        if kind == "markdown":
            stripped = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body.split("\n")]
            cells.append(new_markdown_cell("\n".join(stripped).strip()))
        else:
            cells.append(new_code_cell(body))

    for line in text.split("\n"):
        if line.startswith("# %%"):
            flush()
            buffer = []
            kind = "markdown" if "[markdown]" in line else "code"
            continue
        buffer.append(line)
    flush()
    return cells


def build(path: Path, execute: bool = True) -> tuple[Path, bool, str]:
    nb = new_notebook(cells=parse_percent_script(path.read_text(encoding="utf-8")))
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out_path = OUT_DIR / f"{path.stem}.ipynb"

    status, message = True, "converted"
    if execute:
        started = time.time()
        try:
            # Notebooks live in notebooks/, so run from ROOT to keep `import app` working.
            NotebookClient(nb, timeout=1800, kernel_name="python3", resources={
                "metadata": {"path": str(ROOT)}
            }).execute()
            message = f"executed in {time.time() - started:.1f}s"
        except Exception as exc:  # noqa: BLE001 - report and continue to next notebook
            status = False
            message = f"FAILED: {type(exc).__name__}: {str(exc).strip().splitlines()[-1][:200]}"

    nbformat.write(nb, out_path)
    return out_path, status, message


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    execute = "--no-exec" not in sys.argv

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(SRC_DIR.glob("*.py"))
    if args:
        sources = [s for s in sources if any(s.stem.startswith(a) for a in args)]

    if not sources:
        print("no notebook sources matched")
        return 1

    failures = 0
    for src in sources:
        out, ok, msg = build(src, execute=execute)
        print(f"[{'OK ' if ok else 'ERR'}] {out.name:44s} {msg}")
        failures += 0 if ok else 1

    print(f"\n{len(sources) - failures}/{len(sources)} notebooks built successfully")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
