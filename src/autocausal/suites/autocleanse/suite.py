"""AutoCleanseSuite — SLM-directed orchestration over CleanseActions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import pandas as pd

from autocausal.suites.autocleanse.actions import CLEANSE_REGISTRY, CleanseActions
from autocausal.suites.autocleanse.report import CleanseOp, CleanseReport
from autocausal.suites.base import resolve_frame
from autocausal.suites.director import (
    SLMAutoDirector,
    SLMDirectives,
    resolve_suite_slm,
)

__all__ = ["AutoCleanseSuite", "auto_cleanse"]

ImputeStrategy = Literal["auto", "median_mode", "knn", "none"]


class AutoCleanseSuite:
    """Library-first cleanse suite (SLM picks action sequence when available).

    Example::

        from autocausal.suites.autocleanse import AutoCleanseSuite, CleanseActions
        CleanseActions.impute(df, method="auto")
        clean = AutoCleanseSuite(df, use_slm=True).run()
    """

    def __init__(
        self,
        source: Any = None,
        *,
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        impute: ImputeStrategy = "auto",
        actions: Optional[Sequence[str]] = None,
        outlier_z: float = 5.0,
        max_missing_frac: float = 0.95,
        drop_id_cols: bool = False,
        text: str = "",
        table: Optional[str] = None,
        query: Optional[str] = None,
    ) -> None:
        self.source = source
        self.use_slm = resolve_suite_slm(use_slm)
        self.model_name = model_name
        self.impute = impute
        self.actions_override = list(actions) if actions else None
        self.outlier_z = outlier_z
        self.max_missing_frac = max_missing_frac
        self.drop_id_cols = drop_id_cols
        self.text = text
        self.table = table
        self.query = query
        self.frame: Optional[pd.DataFrame] = None
        self.report: Optional[CleanseReport] = None
        self.directives: Optional[SLMDirectives] = None
        self._ac_in: Any = None

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

        sequence = self.actions_override or list(directives.actions) or CleanseActions.default_sequence()
        # Filter unknown actions softly
        sequence = [a for a in sequence if a in CLEANSE_REGISTRY]
        if not sequence:
            sequence = CleanseActions.default_sequence()

        out = df.copy()
        ops: list[CleanseOp] = []
        dropped: list[str] = []
        warnings: list[str] = []
        notes: list[str] = [
            "AutoCleanse is hygiene for exploratory causal work — not identification.",
        ]
        action_results: list[dict[str, Any]] = []
        actions_run: list[str] = []
        imputation: Optional[dict[str, Any]] = None
        qc: Optional[dict[str, Any]] = None
        missingness: Optional[dict[str, Any]] = None
        n_in, c_in = out.shape

        # Apply director drop_columns first as explicit ops
        for c in list(directives.drop_columns):
            if c in out.columns and c not in dropped:
                out = out.drop(columns=[c])
                dropped.append(c)
                ops.append(CleanseOp("slm_drop", "director drop_columns", [c], 1))

        for name in sequence:
            kwargs: dict[str, Any] = {}
            if name == "coerce_types" and directives.coerce_numeric:
                kwargs["columns"] = directives.coerce_numeric
            elif name == "drop_high_null_cols":
                kwargs["max_missing_frac"] = self.max_missing_frac
                if directives.drop_columns:
                    # already dropped; still scan remaining
                    pass
            elif name == "flag_outliers":
                kwargs["z"] = self.outlier_z
                if directives.flag_outliers:
                    kwargs["columns"] = [c for c in directives.flag_outliers if c in out.columns]
            elif name == "impute":
                kwargs["method"] = self.impute
                if directives.impute_columns:
                    kwargs["columns"] = [c for c in directives.impute_columns if c in out.columns]
            elif name == "strip_id_leakage":
                kwargs["drop"] = self.drop_id_cols

            try:
                result = CLEANSE_REGISTRY.run(name, out, **kwargs)
            except Exception as e:
                warnings.append(f"Action `{name}` soft-fail: {type(e).__name__}: {e}")
                continue

            actions_run.append(name)
            action_results.append(result.to_dict())
            warnings.extend(result.warnings)
            notes.extend(result.notes)
            if result.frame is not None:
                out = result.frame
            for op in result.ops:
                ops.append(
                    CleanseOp(
                        op=str(op.get("op", name)),
                        detail=str(op.get("detail", "")),
                        columns=list(op.get("columns") or []),
                        n_affected=int(op.get("n_affected") or 0),
                    )
                )
            payload = result.payload or {}
            if "dropped_columns" in payload:
                dropped.extend(payload["dropped_columns"])
            if name == "impute" and payload.get("imputation"):
                imputation = payload["imputation"]
            if name == "qc_snapshot" and payload.get("qc"):
                qc = payload["qc"]
            if name == "profile_missingness" and payload.get("missingness"):
                missingness = payload["missingness"]

        report = CleanseReport(
            n_rows_in=n_in,
            n_rows_out=len(out),
            n_cols_in=c_in,
            n_cols_out=out.shape[1],
            operations=ops,
            dropped_columns=list(dict.fromkeys(dropped)),
            action_results=action_results,
            actions_run=actions_run,
            warnings=warnings,
            notes=list(dict.fromkeys(notes + list(directives.notes))),
            slm_directives=directives.to_dict(),
            imputation=imputation,
            qc=qc,
            missingness=missingness,
            source=label,
            backend=directives.backend,
        )
        self.frame = out.reset_index(drop=True)
        self.report = report
        return self

    def to_autocausal(self) -> Any:
        from autocausal.api import AutoCausal

        if self.frame is None:
            self.run()
        assert self.frame is not None
        ac = AutoCausal.from_dataframe(
            self.frame, source=f"cleanse:{self.report.source if self.report else 'memory'}"
        )
        ac.cleanse_report = self.report
        if self._ac_in is not None:
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
    suite = AutoCleanseSuite(source, use_slm=use_slm, **kwargs).run()
    assert suite.frame is not None and suite.report is not None
    return suite.frame, suite.report
