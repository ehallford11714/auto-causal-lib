"""AutoCleanseSuite — SLM-directed orchestration over CleanseActions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import pandas as pd

from autocausal.suites.autocleanse.actions import CLEANSE_REGISTRY, CleanseActions
from autocausal.suites.autocleanse.report import CleanseOp, CleanseReport
from autocausal.suites.base import resolve_frame
from autocausal.suites.director import (
    SLMAutoDirector,
    SLMDirectives,
    resolve_suite_slm,
)
from autocausal.production import (
    GateReport,
    GateResult,
    ProductionGateError,
    ProductionPolicy,
    build_data_fingerprint,
    privacy_scan,
    resolve_policy,
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
        mode: str = "exploratory",
        policy: Optional[ProductionPolicy | Mapping[str, Any]] = None,
        dry_run: bool = False,
        schema: Optional[Mapping[str, str]] = None,
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
        self.mode = mode
        self.policy_input = policy
        self.dry_run = bool(dry_run)
        self.expected_schema = {
            str(column): str(dtype) for column, dtype in (schema or {}).items()
        }
        self.frame: Optional[pd.DataFrame] = None
        self.report: Optional[CleanseReport] = None
        self.directives: Optional[SLMDirectives] = None
        self._ac_in: Any = None
        self._before_frame: Optional[pd.DataFrame] = None

    def run(self, source: Any = None, *, text: Optional[str] = None) -> "AutoCleanseSuite":
        src = self.source if source is None else source
        if src is None:
            raise ValueError("AutoCleanseSuite requires a DataFrame, path, or AutoCausal")
        df, label, ac_in = resolve_frame(src, table=self.table, query=self.query)
        self._ac_in = ac_in
        txt = self.text if text is None else text
        inherited_policy = (
            getattr(ac_in, "policy", None) if self.policy_input is None else None
        )
        inherited_mode = getattr(ac_in, "mode", self.mode)
        policy = resolve_policy(
            inherited_mode,
            self.policy_input or inherited_policy,
        )
        quality = policy.data_quality
        assert quality is not None
        self._before_frame = df.copy(deep=True)

        effective_use_slm = self.use_slm and policy.allow_slm
        director = SLMAutoDirector(
            use_slm=effective_use_slm, model_name=self.model_name
        )
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
        before_fingerprint = build_data_fingerprint(out)
        schema_violations: list[dict[str, Any]] = []
        range_violations: list[dict[str, Any]] = []
        gate_report = GateReport(
            profile=policy.profile,
            policy_version=policy.policy_version,
        )
        for column, expected in self.expected_schema.items():
            if column not in out.columns:
                schema_violations.append(
                    {"column": column, "issue": "missing_column", "count": n_in}
                )
            elif str(out[column].dtype) != expected:
                schema_violations.append(
                    {
                        "column": column,
                        "issue": f"dtype={out[column].dtype}; expected={expected}",
                        "count": n_in,
                    }
                )
        for column, limits in quality.range_constraints.items():
            if column not in out.columns:
                continue
            values = pd.to_numeric(out[column], errors="coerce")
            lower, upper = limits
            invalid = pd.Series(False, index=out.index)
            if lower is not None:
                invalid |= values < lower
            if upper is not None:
                invalid |= values > upper
            count = int(invalid.fillna(False).sum())
            if count:
                range_violations.append(
                    {
                        "column": str(column),
                        "issue": f"outside configured range [{lower}, {upper}]",
                        "count": count,
                    }
                )
        gate_report.add(
            GateResult(
                id="cleanse_schema_validation",
                ok=not schema_violations,
                status=(
                    "pass"
                    if not schema_violations
                    else "warn"
                    if policy.profile == "exploratory"
                    else "fail"
                ),
                detail=(
                    "Schema matches configured expectations."
                    if not schema_violations
                    else f"{len(schema_violations)} schema violation(s)."
                ),
                metric=len(schema_violations),
                threshold=0,
                remediation="Correct source schema or add an audited coercion override.",
                stage="cleanse",
            ),
            GateResult(
                id="cleanse_range_validation",
                ok=not range_violations,
                status=(
                    "pass"
                    if not range_violations
                    else "warn"
                    if policy.profile == "exploratory"
                    else "fail"
                ),
                detail=(
                    "Configured ranges passed."
                    if not range_violations
                    else f"{len(range_violations)} range constraint(s) violated."
                ),
                metric=sum(item["count"] for item in range_violations),
                threshold=0,
                remediation="Correct impossible values under an approved rule; raw rows were not logged.",
                stage="cleanse",
            ),
        )

        # Apply director drop_columns first as explicit ops
        for c in list(directives.drop_columns):
            if (
                c in out.columns
                and c not in dropped
                and quality.allow_destructive_column_drop
            ):
                out = out.drop(columns=[c])
                dropped.append(c)
                ops.append(CleanseOp("slm_drop", "director drop_columns", [c], 1))
            elif c in out.columns:
                warnings.append(
                    f"Blocked director drop of `{c}` by data-quality policy."
                )

        for name in sequence:
            kwargs: dict[str, Any] = {}
            if name == "coerce_types" and directives.coerce_numeric:
                kwargs["columns"] = directives.coerce_numeric
            if name == "coerce_types" and not quality.allow_type_coercion:
                warnings.append("Blocked `coerce_types` by data-quality policy.")
                continue
            elif name == "drop_high_null_cols":
                if not quality.allow_destructive_column_drop:
                    warnings.append(
                        "Flagged high-null columns; destructive drop blocked by policy."
                    )
                    continue
                kwargs["max_missing_frac"] = self.max_missing_frac
                if directives.drop_columns:
                    # already dropped; still scan remaining
                    pass
            elif name == "flag_outliers":
                kwargs["z"] = self.outlier_z
                kwargs["winsorize"] = quality.allow_winsorization
                if directives.flag_outliers:
                    kwargs["columns"] = [c for c in directives.flag_outliers if c in out.columns]
            elif name == "impute":
                if not quality.allow_imputation:
                    warnings.append("Blocked imputation by data-quality policy.")
                    continue
                kwargs["method"] = self.impute
                if directives.impute_columns:
                    kwargs["columns"] = [c for c in directives.impute_columns if c in out.columns]
            elif name == "strip_id_leakage":
                kwargs["drop"] = (
                    self.drop_id_cols and quality.allow_destructive_column_drop
                )
            elif name == "drop_constant_cols":
                if not quality.allow_destructive_column_drop:
                    warnings.append(
                        "Flagged constant columns; destructive drop blocked by policy."
                    )
                    continue
            elif name == "drop_duplicates":
                if (
                    not quality.allow_row_drop
                    or quality.duplicate_action != "drop"
                ):
                    duplicate_count = int(out.duplicated().sum())
                    if duplicate_count:
                        warnings.append(
                            f"Flagged {duplicate_count} duplicate rows; row drop blocked by policy."
                        )
                    continue

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

        max_missing = (
            float(out.isna().mean().max()) if len(out.columns) else 0.0
        )
        gate_report.add(
            GateResult(
                id="cleanse_min_rows",
                ok=len(out) >= quality.min_rows,
                status=(
                    "pass"
                    if len(out) >= quality.min_rows
                    else "warn"
                    if policy.profile == "exploratory"
                    else "fail"
                ),
                detail=f"Rows after preview={len(out)}; minimum={quality.min_rows}.",
                metric=len(out),
                threshold=quality.min_rows,
                remediation="Collect more rows or reduce design complexity.",
                stage="cleanse",
            ),
            GateResult(
                id="cleanse_missingness",
                ok=max_missing <= quality.max_missing_fraction,
                status=(
                    "pass"
                    if max_missing <= quality.max_missing_fraction
                    else "warn"
                    if policy.profile == "exploratory"
                    else "fail"
                ),
                detail=f"Maximum column missingness={max_missing:.3f}.",
                metric=max_missing,
                threshold=quality.max_missing_fraction,
                remediation="Investigate collection and apply approved leakage-safe imputation.",
                stage="cleanse",
            ),
        )
        pii = privacy_scan(out)
        pii_found = bool(pii.get("pii_columns"))
        pii_fails = bool(policy.fail_on_pii and pii_found)
        gate_report.add(
            GateResult(
                id="cleanse_privacy",
                ok=not pii_fails,
                status=(
                    "pass"
                    if not pii_found
                    else "fail"
                    if pii_fails and policy.profile != "exploratory"
                    else "warn"
                ),
                detail=(
                    "No likely PII column names detected."
                    if not pii_found
                    else f"Likely PII columns detected: {pii.get('pii_columns')}"
                ),
                metric=len(pii.get("pii_columns") or []),
                threshold=0 if policy.fail_on_pii else "warn",
                remediation="Redact/tokenize PII before external use.",
                stage="privacy",
            )
        )

        def redact(value: Any, key: str = "") -> Any:
            if not policy.redact_sample_values:
                return value
            if key.lower() in {
                "fill_value",
                "sample",
                "sample_value",
                "example",
                "raw_value",
            }:
                return "<redacted>"
            if isinstance(value, dict):
                return {str(k): redact(v, str(k)) for k, v in value.items()}
            if isinstance(value, list):
                return [redact(item, key) for item in value]
            return value

        action_results = redact(action_results)
        imputation = redact(imputation)
        after_fingerprint = build_data_fingerprint(out)

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
            dry_run=self.dry_run,
            reversible=True,
            policy_profile=policy.profile,
            before_fingerprint=before_fingerprint,
            after_fingerprint=after_fingerprint,
            schema_violations=schema_violations,
            range_violations=range_violations,
            pii_summary={
                "pii_columns": list(pii.get("pii_columns") or []),
                "high_cardinality_columns": list(
                    pii.get("high_cardinality_columns") or []
                ),
                "raw_values_included": False,
            },
            gate_report=gate_report.to_dict(),
        )
        self.frame = (
            df.copy(deep=True)
            if self.dry_run
            else out.reset_index(drop=True)
        )
        self.report = report
        if policy.profile == "production" and gate_report.failed:
            raise ProductionGateError(
                "AutoCleanse failed production data-quality/privacy gates.",
                code="cleanse_gate_failed",
                gates=gate_report.failed,
                recommendations=[
                    gate.remediation or "Review data quality."
                    for gate in gate_report.failed
                ],
                partial_result=report,
            )
        return self

    def rollback(self) -> pd.DataFrame:
        """Restore and return the pre-cleanse frame retained in memory."""
        if self._before_frame is None:
            raise RuntimeError("Run the cleanse suite before rollback().")
        self.frame = self._before_frame.copy(deep=True)
        return self.frame

    @staticmethod
    def transform_train_test(
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Median/mode imputation fitted on train only.

        Fill values are intentionally omitted from the returned ledger.
        """
        if list(train.columns) != list(test.columns):
            raise ValueError("Train/test columns must match in the same order.")
        train_out = train.copy()
        test_out = test.copy()
        transformed: list[dict[str, Any]] = []
        for column in train.columns:
            if not train[column].isna().any() and not test[column].isna().any():
                continue
            if pd.api.types.is_numeric_dtype(train[column]):
                fill = train[column].median()
                strategy = "train_median"
            else:
                modes = train[column].mode(dropna=True)
                fill = modes.iloc[0] if len(modes) else ""
                strategy = "train_mode"
            train_missing = int(train_out[column].isna().sum())
            test_missing = int(test_out[column].isna().sum())
            train_out[column] = train_out[column].fillna(fill)
            test_out[column] = test_out[column].fillna(fill)
            transformed.append(
                {
                    "column": str(column),
                    "strategy": strategy,
                    "train_missing": train_missing,
                    "test_missing": test_missing,
                    "fit_scope": "train_only",
                    "fill_value": "<redacted>",
                }
            )
        return train_out, test_out, {
            "schema": "AutoCausalSplitTransform.v1",
            "fit_scope": "train_only",
            "transformations": transformed,
        }

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
