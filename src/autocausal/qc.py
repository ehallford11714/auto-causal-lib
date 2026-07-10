"""Data-frame QC gate: ID leakage, bad keys, and basic tabular hygiene.

Hook before ``discover`` — warn or block on serious issues.
Epistemic honesty: QC flags data hygiene risks; it does not prove causality.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Sequence

import pandas as pd

from autocausal.roles import ColumnRole, infer_column_roles

__all__ = ["QCIssue", "QCReport", "validate_frame"]

QCSeverity = Literal["info", "warn", "block"]


@dataclass
class QCIssue:
    code: str
    severity: QCSeverity
    message: str
    columns: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QCReport:
    ok: bool
    issues: list[QCIssue] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    blocked: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "blocked": self.blocked,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "issues": [i.to_dict() for i in self.issues],
            "notes": list(self.notes),
        }

    @property
    def warnings(self) -> list[QCIssue]:
        return [i for i in self.issues if i.severity == "warn"]

    def block_issues(self) -> list[QCIssue]:
        return [i for i in self.issues if i.severity == "block"]


_ID_NAME_RE = re.compile(
    r"(^id$|_id$|^uuid$|guid|ssn|email|phone|account_?number|customer_?key|row_?id|index$)",
    re.I,
)
_LEAK_NAME_RE = re.compile(
    r"(target_leak|future_|label_encoded|y_true|ground_truth|outcome_raw)",
    re.I,
)


def validate_frame(
    df: pd.DataFrame,
    *,
    key_columns: Optional[Sequence[str]] = None,
    block_on: Sequence[str] = ("id_leakage_high_cardinality", "empty_frame", "duplicate_columns"),
    max_id_cardinality_ratio: float = 0.95,
    min_rows: int = 2,
) -> QCReport:
    """Validate a frame for discovery hygiene.

    Parameters
    ----------
    df :
        Input tabular frame.
    key_columns :
        Optional join/entity keys to check for nulls / uniqueness.
    block_on :
        Issue codes that flip ``blocked=True`` (and ``ok=False``).
    max_id_cardinality_ratio :
        Flag numeric/object columns named like IDs when nunique/n >= this.
    min_rows :
        Minimum rows expected for discovery.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    issues: list[QCIssue] = []
    n_rows, n_cols = len(df), len(df.columns)
    notes = [
        "QC is a hygiene gate for exploratory discovery — not a causal identification test.",
    ]

    if n_rows == 0 or n_cols == 0:
        issues.append(
            QCIssue("empty_frame", "block", f"Frame is empty ({n_rows}×{n_cols}).")
        )

    if n_rows < min_rows and n_rows > 0:
        issues.append(
            QCIssue(
                "too_few_rows",
                "warn",
                f"Only {n_rows} row(s); discovery will be unstable.",
                meta={"min_rows": min_rows},
            )
        )

    # duplicate column names
    cols = [str(c) for c in df.columns]
    dupes = sorted({c for c in cols if cols.count(c) > 1})
    if dupes:
        issues.append(
            QCIssue(
                "duplicate_columns",
                "block",
                f"Duplicate column names: {dupes[:8]}",
                columns=dupes,
            )
        )

    roles = infer_column_roles(df)
    id_like: list[str] = []
    for col in df.columns:
        name = str(col)
        s = df[col]
        nunique = int(s.nunique(dropna=True))
        ratio = nunique / max(n_rows, 1)
        role = roles.get(col, ColumnRole.UNKNOWN)
        looks_id = bool(_ID_NAME_RE.search(name)) or role == ColumnRole.ID
        if looks_id and ratio >= max_id_cardinality_ratio and n_rows >= 5:
            id_like.append(name)
            issues.append(
                QCIssue(
                    "id_leakage_high_cardinality",
                    "block" if "id_leakage_high_cardinality" in block_on else "warn",
                    f"Column `{name}` looks like an ID (cardinality={nunique}/{n_rows}). "
                    "Including IDs in discovery can leak entity identity into edges.",
                    columns=[name],
                    meta={"nunique": nunique, "ratio": round(ratio, 4)},
                )
            )
        if _LEAK_NAME_RE.search(name):
            issues.append(
                QCIssue(
                    "suspicious_leak_name",
                    "warn",
                    f"Column `{name}` name suggests label/future leakage.",
                    columns=[name],
                )
            )
        # constant columns
        if nunique <= 1 and n_rows > 0:
            issues.append(
                QCIssue(
                    "constant_column",
                    "info",
                    f"Column `{name}` is constant.",
                    columns=[name],
                )
            )
        miss = float(s.isna().mean())
        if miss >= 0.5:
            issues.append(
                QCIssue(
                    "high_missingness",
                    "warn",
                    f"Column `{name}` has {miss:.0%} missing.",
                    columns=[name],
                    meta={"missing_pct": round(miss * 100, 2)},
                )
            )

    # key checks
    for key in key_columns or []:
        if key not in df.columns:
            issues.append(
                QCIssue(
                    "missing_key",
                    "block",
                    f"Declared key `{key}` not in columns.",
                    columns=[key],
                )
            )
            continue
        nulls = int(df[key].isna().sum())
        if nulls:
            issues.append(
                QCIssue(
                    "null_key",
                    "warn",
                    f"Key `{key}` has {nulls} null(s).",
                    columns=[key],
                    meta={"nulls": nulls},
                )
            )
        if not df[key].is_unique:
            issues.append(
                QCIssue(
                    "nonunique_key",
                    "warn",
                    f"Key `{key}` is not unique — panel/join may duplicate rows.",
                    columns=[key],
                )
            )

    # all-null rows
    if n_rows > 0 and n_cols > 0:
        all_null = int(df.isna().all(axis=1).sum())
        if all_null:
            issues.append(
                QCIssue(
                    "all_null_rows",
                    "warn",
                    f"{all_null} fully-null row(s).",
                    meta={"count": all_null},
                )
            )

    block_codes = set(block_on)
    blocked = any(i.severity == "block" or i.code in block_codes and i.severity != "info" for i in issues if i.severity == "block")
    # recompute blocked strictly from severity==block
    blocked = any(i.severity == "block" for i in issues)
    ok = not blocked
    if id_like:
        notes.append(f"ID-like columns flagged: {id_like[:8]}")
    return QCReport(
        ok=ok,
        issues=issues,
        n_rows=n_rows,
        n_cols=n_cols,
        blocked=blocked,
        notes=notes,
    )
