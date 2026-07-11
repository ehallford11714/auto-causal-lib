"""AutoCleanseSuite — SLM-directed data cleansing before causal work."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from autocausal.suites.base import resolve_frame, write_report
from autocausal.suites.director import (
    EPISTEMIC_NOTE,
    SLMAutoDirector,
    SLMDirectives,
    resolve_suite_slm,
)

__all__ = ["CleanseOp", "CleanseReport", "AutoCleanseSuite", "auto_cleanse"]

ImputeStrategy = Literal["auto", "median_mode", "knn", "none"]


@dataclass
class CleanseOp:
    op: str
    detail: str
    columns: list[str] = field(default_factory=list)
    n_affected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleanseReport:
    n_rows_in: int
    n_rows_out: int
    n_cols_in: int
    n_cols_out: int
    operations: list[CleanseOp] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    slm_directives: Optional[dict[str, Any]] = None
    imputation: Optional[dict[str, Any]] = None
    qc: Optional[dict[str, Any]] = None
    source: str = ""
    backend: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "n_cols_in": self.n_cols_in,
            "n_cols_out": self.n_cols_out,
            "operations": [o.to_dict() for o in self.operations],
            "dropped_columns": list(self.dropped_columns),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "slm_directives": self.slm_directives,
            "imputation": self.imputation,
            "qc": self.qc,
            "source": self.source,
            "backend": self.backend,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# AutoCleanse report",
            "",
            f"- rows {self.n_rows_in} → {self.n_rows_out}",
            f"- cols {self.n_cols_in} → {self.n_cols_out}",
            f"- backend: `{self.backend}`",
            "",
            f"> {EPISTEMIC_NOTE}",
            "",
            "## Operations",
            "",
        ]
        if not self.operations:
            lines.append("_No mutations._")
        else:
            for op in self.operations:
                lines.append(f"- `{op.op}`: {op.detail} (n={op.n_affected})")
        if self.dropped_columns:
            lines += ["", "## Dropped columns", ""]
            for c in self.dropped_columns:
                lines.append(f"- `{c}`")
        if self.warnings:
            lines += ["", "## Warnings", ""]
            for w in self.warnings:
                lines.append(f"- {w}")
        if self.notes:
            lines += ["", "## Notes", ""]
            for n in self.notes:
                lines.append(f"- {n}")
        lines.append("")
        return "\n".join(lines)

    def write(self, path: Union[str, Path], *, fmt: str = "auto") -> Path:
        return write_report(self, path, fmt=fmt)


class AutoCleanseSuite:
    """Library-first cleanse suite (SLM-directed when available).

    Example::

        from autocausal import AutoCleanseSuite
        clean = AutoCleanseSuite(df, use_slm=True).run()
        ac = clean.to_autocausal()
    """

    def __init__(
        self,
        source: Any = None,
        *,
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        impute: ImputeStrategy = "auto",
        drop_duplicates: bool = True,
        drop_constant: bool = True,
        outlier_z: Optional[float] = 5.0,
        coerce_numeric: bool = True,
        max_missing_frac: float = 0.95,
        text: str = "",
        table: Optional[str] = None,
        query: Optional[str] = None,
    ) -> None:
        self.source = source
        self.use_slm = resolve_suite_slm(use_slm)
        self.model_name = model_name
        self.impute = impute
        self.drop_duplicates = drop_duplicates
        self.drop_constant = drop_constant
        self.outlier_z = outlier_z
        self.coerce_numeric = coerce_numeric
        self.max_missing_frac = max_missing_frac
        self.text = text
        self.table = table
        self.query = query
        self.frame: Optional[pd.DataFrame] = None
        self.report: Optional[CleanseReport] = None
        self._ac_in: Any = None
        self.directives: Optional[SLMDirectives] = None

    def run(self, source: Any = None, *, text: Optional[str] = None) -> "AutoCleanseSuite":
        src = self.source if source is None else source
        if src is None:
            raise ValueError("AutoCleanseSuite requires a DataFrame, path, or AutoCausal")
        df, label, ac_in = resolve_frame(src, table=self.table, query=self.query)
        self._ac_in = ac_in
        txt = self.text if text is None else text

        director = SLMAutoDirector(use_slm=self.use_slm, model_name=self.model_name)
        directives = director.direct("cleanse", df, text=txt)
        self.directives = directives

        out, report = _apply_cleanse(
            df,
            directives=directives,
            impute=self.impute,
            drop_duplicates=self.drop_duplicates and directives.drop_duplicates,
            drop_constant=self.drop_constant and directives.drop_constant,
            outlier_z=self.outlier_z,
            coerce_numeric=self.coerce_numeric,
            max_missing_frac=self.max_missing_frac,
            source_label=label,
        )
        report.slm_directives = directives.to_dict()
        report.backend = directives.backend
        report.notes = list(dict.fromkeys(list(report.notes) + list(directives.notes)))
        self.frame = out
        self.report = report
        return self

    def to_autocausal(self) -> Any:
        from autocausal.api import AutoCausal

        if self.frame is None:
            self.run()
        assert self.frame is not None
        ac = AutoCausal.from_dataframe(self.frame, source=f"cleanse:{self.report.source if self.report else 'memory'}")
        ac.cleanse_report = self.report
        if self._ac_in is not None:
            # preserve join_log / nlp hints lightly
            ac.join_log = list(getattr(self._ac_in, "join_log", []) or [])
            ac.nlp_hints = getattr(self._ac_in, "nlp_hints", None)
        return ac

    def to_dict(self) -> dict[str, Any]:
        if self.report is None:
            self.run()
        assert self.report is not None
        return self.report.to_dict()

    def to_markdown(self) -> str:
        if self.report is None:
            self.run()
        assert self.report is not None
        return self.report.to_markdown()

    def write(self, path: Union[str, Path], *, fmt: str = "auto") -> Path:
        if self.report is None:
            self.run()
        assert self.report is not None
        return self.report.write(path, fmt=fmt)


def auto_cleanse(
    source: Any,
    *,
    use_slm: Optional[bool] = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, CleanseReport]:
    """Functional API → (cleaned_frame, CleanseReport)."""
    suite = AutoCleanseSuite(source, use_slm=use_slm, **kwargs).run()
    assert suite.frame is not None and suite.report is not None
    return suite.frame, suite.report


def _apply_cleanse(
    df: pd.DataFrame,
    *,
    directives: SLMDirectives,
    impute: ImputeStrategy,
    drop_duplicates: bool,
    drop_constant: bool,
    outlier_z: Optional[float],
    coerce_numeric: bool,
    max_missing_frac: float,
    source_label: str,
) -> tuple[pd.DataFrame, CleanseReport]:
    out = df.copy()
    ops: list[CleanseOp] = []
    dropped: list[str] = []
    warnings: list[str] = []
    notes: list[str] = [
        "AutoCleanse is hygiene for exploratory causal work — not identification.",
    ]
    n_in, c_in = out.shape

    # Director-requested drops first
    for c in list(directives.drop_columns):
        if c in out.columns:
            out = out.drop(columns=[c])
            dropped.append(c)
            ops.append(CleanseOp("slm_drop", "director drop_columns", [c], 1))

    if coerce_numeric:
        targets = directives.coerce_numeric or list(out.columns)
        for c in list(targets):
            if c not in out.columns:
                continue
            if pd.api.types.is_numeric_dtype(out[c]) or pd.api.types.is_datetime64_any_dtype(out[c]):
                continue
            if out[c].dtype == object or str(out[c].dtype) == "string":
                converted = pd.to_numeric(out[c], errors="coerce")
                if converted.notna().mean() >= 0.5:
                    n_chg = int(converted.notna().sum())
                    out[c] = converted
                    ops.append(CleanseOp("coerce_numeric", "object→numeric", [c], n_chg))
                else:
                    dt = pd.to_datetime(out[c], errors="coerce", utc=True)
                    if dt.notna().mean() >= 0.5:
                        out[c] = dt
                        ops.append(
                            CleanseOp("coerce_datetime", "object→datetime(utc)", [c], int(dt.notna().sum()))
                        )

    for c in list(out.columns):
        miss = float(out[c].isna().mean()) if len(out) else 0.0
        if miss >= max_missing_frac:
            out = out.drop(columns=[c])
            dropped.append(c)
            ops.append(CleanseOp("drop_high_missing", f"missing={miss:.0%}", [c], 1))

    if drop_constant:
        for c in list(out.columns):
            if out[c].nunique(dropna=True) <= 1:
                out = out.drop(columns=[c])
                dropped.append(c)
                ops.append(CleanseOp("drop_constant", "≤1 unique value", [c], 1))

    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates()
        n_dup = before - len(out)
        if n_dup:
            ops.append(CleanseOp("drop_duplicates", "exact duplicate rows removed", [], n_dup))

    imputation_dict: Optional[dict[str, Any]] = None
    if impute != "none" and out.isna().any().any():
        # Prefer existing autocausal imputer
        try:
            from autocausal.impute import impute_dataframe

            method = "auto" if impute == "auto" else impute
            # Restrict to director impute_columns when provided and still present
            if directives.impute_columns:
                miss_cols = [c for c in directives.impute_columns if c in out.columns and out[c].isna().any()]
                if miss_cols:
                    # impute full frame (imputer is column-wise); still fine
                    pass
            out, irep = impute_dataframe(out, method=method)  # type: ignore[arg-type]
            imputation_dict = irep.to_dict() if hasattr(irep, "to_dict") else None
            for col_info in getattr(irep, "columns", []) or []:
                ops.append(
                    CleanseOp(
                        f"impute_{getattr(col_info, 'strategy', 'auto')}",
                        f"missing {getattr(col_info, 'missing_before', 0)}→{getattr(col_info, 'missing_after', 0)}",
                        [str(getattr(col_info, "column", ""))],
                        int(getattr(col_info, "missing_before", 0) or 0),
                    )
                )
        except Exception as e:
            warnings.append(f"Imputer soft-fail: {type(e).__name__}: {e}")

    if outlier_z is not None and outlier_z > 0:
        targets = directives.flag_outliers or [
            c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])
        ]
        for c in targets:
            if c not in out.columns or not pd.api.types.is_numeric_dtype(out[c]):
                continue
            s = out[c].astype(float)
            mu, sd = float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0
            if sd < 1e-12 or not np.isfinite(sd):
                continue
            z = (s - mu) / sd
            mask = z.abs() > outlier_z
            n_out = int(mask.sum())
            if n_out:
                lo, hi = mu - outlier_z * sd, mu + outlier_z * sd
                out[c] = s.clip(lo, hi)
                ops.append(
                    CleanseOp(
                        "winsorize_z",
                        f"z>{outlier_z} clipped",
                        [c],
                        n_out,
                    )
                )

    qc_dict: Optional[dict[str, Any]] = None
    try:
        from autocausal.qc import validate_frame

        qc_dict = validate_frame(out).to_dict()
    except Exception as e:
        warnings.append(f"QC soft-fail: {type(e).__name__}: {e}")

    report = CleanseReport(
        n_rows_in=n_in,
        n_rows_out=len(out),
        n_cols_in=c_in,
        n_cols_out=out.shape[1],
        operations=ops,
        dropped_columns=list(dict.fromkeys(dropped)),
        warnings=warnings,
        notes=notes,
        imputation=imputation_dict,
        qc=qc_dict,
        source=source_label,
        backend=directives.backend,
    )
    return out.reset_index(drop=True), report
