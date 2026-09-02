"""A tiny dataframe validation engine.

The build notes start with scattered `assert` statements in notebooks and plan to
consolidate them into one place later; this is that place. Rules are data, so the
same definitions drive the notebooks, the batch pipelines and the unit tests.

Deliberately not Pandera yet - the whole engine is ~120 lines and has no
dependency beyond pandas. Swap it out when the rule set outgrows this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass
class ValidationReport:
    """Collected outcome of validating one dataframe."""

    dataset: str
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_run: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def raise_if_failed(self) -> None:
        if not self.ok:
            joined = "\n  - ".join(self.errors)
            raise ValueError(f"Validation failed for {self.dataset}:\n  - {joined}")

    def __str__(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [
            f"[{status}] {self.dataset}: {self.n_rows:,} rows, "
            f"{self.checks_run} checks, {len(self.errors)} errors, {len(self.warnings)} warnings"
        ]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  WARNING {w}" for w in self.warnings]
        return "\n".join(lines)


def check_required_columns(df: pd.DataFrame, required: Sequence[str], report: ValidationReport) -> None:
    report.checks_run += 1
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.add_error(f"missing required columns: {missing}")


def check_numeric(df: pd.DataFrame, columns: Sequence[str], report: ValidationReport) -> None:
    for col in columns:
        report.checks_run += 1
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            report.add_error(f"column '{col}' should be numeric, found dtype {df[col].dtype}")


def check_ranges(df: pd.DataFrame, rules: Mapping[str, tuple[float, float]], report: ValidationReport) -> None:
    """Range rules are the ones that catch an EngagementScore of 250 out of 100."""
    for col, (low, high) in rules.items():
        if col not in df.columns:
            continue
        report.checks_run += 1
        if not pd.api.types.is_numeric_dtype(df[col]):
            report.add_error(f"range check on '{col}' skipped: column is not numeric")
            continue
        offenders = df[~df[col].between(low, high) & df[col].notna()]
        if len(offenders):
            observed = (offenders[col].min(), offenders[col].max())
            report.add_error(
                f"'{col}' has {len(offenders)} value(s) outside [{low}, {high}]; "
                f"observed range of offenders {observed}"
            )


def check_unique(df: pd.DataFrame, column: str, report: ValidationReport) -> None:
    report.checks_run += 1
    if column not in df.columns:
        report.add_error(f"uniqueness check failed: no column '{column}'")
        return
    duplicated = int(df[column].duplicated().sum())
    if duplicated:
        report.add_error(f"'{column}' must be unique but has {duplicated} duplicate value(s)")


def check_categories(df: pd.DataFrame, domains: Mapping[str, Iterable], report: ValidationReport) -> None:
    for col, allowed in domains.items():
        if col not in df.columns:
            continue
        report.checks_run += 1
        allowed = set(allowed)
        found = set(df[col].dropna().unique())
        unexpected = found - allowed
        if unexpected:
            report.add_error(f"'{col}' contains unexpected values {sorted(map(str, unexpected))}")


def check_no_nulls(df: pd.DataFrame, columns: Sequence[str], report: ValidationReport) -> None:
    for col in columns:
        report.checks_run += 1
        if col not in df.columns:
            continue
        n = int(df[col].isna().sum())
        if n:
            report.add_error(f"'{col}' has {n} null value(s) but must be complete")


def check_no_duplicate_rows(df: pd.DataFrame, report: ValidationReport) -> None:
    report.checks_run += 1
    n = int(df.duplicated().sum())
    if n:
        report.add_warning(f"{n} fully duplicated row(s) present")


def check_referential_integrity(
    child: pd.DataFrame, child_key: str, parent_keys: Iterable, report: ValidationReport, label: str
) -> None:
    """Every foreign key in `child` must exist in the parent key set."""
    report.checks_run += 1
    if child_key not in child.columns:
        report.add_error(f"referential check failed: no column '{child_key}'")
        return
    parent_keys = set(parent_keys)
    orphans = set(child[child_key].dropna().unique()) - parent_keys
    if orphans:
        sample = sorted(map(str, orphans))[:5]
        report.add_error(
            f"{len(orphans)} '{child_key}' value(s) in {label} have no matching parent "
            f"(e.g. {sample})"
        )
