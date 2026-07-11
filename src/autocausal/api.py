"""Public AutoCausal API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import pandas as pd

from autocausal.discovery import discover_ensemble, discover_relationships
from autocausal.impute import ImputationReport, impute_dataframe
from autocausal.ingest import load_csv, load_parquet, load_sqlalchemy
from autocausal.production import (
    EvidenceGateError,
    GateResult,
    ProductionGateError,
    ProductionPolicy,
    ResourceLimitError,
    RunManifest,
    RunRecorder,
    UnsafePayloadError,
    annotate_and_gate_edges,
    build_manifest,
    check_required_engines,
    is_production,
    privacy_scan,
    resolve_mode,
    resolve_policy,
)
from autocausal.results import AutoResult, DiscoveryResult
from autocausal.roles import ColumnRole, infer_column_roles


ImputeMethod = Literal["auto", "median_mode", "knn"]
QCMode = Literal["off", "warn", "block"]
AnalysisMode = Literal["exploratory", "review", "production"]

__all__ = ["AutoCausal", "DiscoveryResult", "AutoResult"]


class AutoCausal:
    """Load tabular data, impute missing fields, discover exploratory causal edges."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        source: str = "memory",
        mode: AnalysisMode = "exploratory",
        strict: bool = False,
        policy: Optional[ProductionPolicy | dict[str, Any]] = None,
        random_state: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        self._raw = df.copy()
        self._df = df.copy()
        self.source = source
        self.mode: AnalysisMode = resolve_mode(mode, strict=strict if strict else None)
        self.policy = resolve_policy(
            self.mode, policy, random_state=random_state
        )
        self.random_state = int(self.policy.random_state)
        self.run_manifest: RunManifest = build_manifest(
            self._df,
            mode=self.mode,
            policy=self.policy,
            run_id=run_id,
            config={"source": self.source},
        )
        self.run_id = self.run_manifest.run_id
        self._recorder = RunRecorder(self.run_manifest)
        self.imputation: Optional[ImputationReport] = None
        self.result: Optional[DiscoveryResult] = None
        self.mining: Any = None
        self.guide_result: Any = None
        self.direction_plan: Any = None
        self.creation_result: Any = None
        self.inference_result: Any = None
        self.grounding: Any = None
        self.physics_result: Any = None
        self.nlp_hints: Any = None
        self.behavioral_result: Any = None
        self.join_log: list[dict[str, Any]] = []
        self.roles: dict[str, ColumnRole] = infer_column_roles(self._df)
        self.qc_report: Any = None
        self.sensitivity_report: Any = None
        self.panel_spec: Any = None
        self.refute_results: list[Any] = []
        self.estimate_results: list[Any] = []
        self.correlation_results: list[Any] = []
        self.causal_inference_results: list[Any] = []
        # First-class suite reports (AutoCleanse / AutoEDA / AutoMine)
        self.cleanse_report: Any = None
        self.eda_report: Any = None
        self.mine_report: Any = None
        self._suite_use_slm: Optional[bool] = None
        self.grail_report: Any = None
        # Optional IV role overrides (set_iv_roles / discover candidates=)
        self._iv_roles: dict[str, list[str]] = {}


    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        mode: AnalysisMode = "exploratory",
        strict: bool = False,
        policy: Optional[ProductionPolicy | dict[str, Any]] = None,
        random_state: Optional[int] = None,
        run_id: Optional[str] = None,
        **read_csv_kwargs: Any,
    ) -> "AutoCausal":
        df = load_csv(path, **read_csv_kwargs)
        return cls(
            df,
            source=f"csv:{path}",
            mode=mode,
            strict=strict,
            policy=policy,
            random_state=random_state,
            run_id=run_id,
        )

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        mode: AnalysisMode = "exploratory",
        strict: bool = False,
        policy: Optional[ProductionPolicy | dict[str, Any]] = None,
        random_state: Optional[int] = None,
        run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> "AutoCausal":
        df = load_parquet(path, **kwargs)
        return cls(
            df,
            source=f"parquet:{path}",
            mode=mode,
            strict=strict,
            policy=policy,
            random_state=random_state,
            run_id=run_id,
        )

    @classmethod
    def from_sqlalchemy(
        cls,
        url: str,
        *,
        table: Optional[str] = None,
        query: Optional[str] = None,
        schema: Optional[str] = None,
        limit: Optional[int] = None,
        chunksize: Optional[int] = None,
        sample_n: Optional[int] = None,
        sample_seed: Optional[int] = None,
        mode: AnalysisMode = "exploratory",
        strict: bool = False,
        policy: Optional[ProductionPolicy | dict[str, Any]] = None,
        random_state: Optional[int] = None,
        run_id: Optional[str] = None,
        **engine_kwargs: Any,
    ) -> "AutoCausal":
        if sample_seed is None and random_state is not None:
            sample_seed = int(random_state)
        df = load_sqlalchemy(
            url,
            table=table,
            query=query,
            schema=schema,
            limit=limit,
            chunksize=chunksize,
            sample_n=sample_n,
            sample_seed=sample_seed,
            **engine_kwargs,
        )
        # Do not persist SQL text/parameters into source labels or manifests.
        label = table or ("query" if query else "sql")
        return cls(
            df,
            source=f"sql:{label}",
            mode=mode,
            strict=strict,
            policy=policy,
            random_state=random_state,
            run_id=run_id,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        source: str = "dataframe",
        mode: AnalysisMode = "exploratory",
        strict: bool = False,
        policy: Optional[ProductionPolicy | dict[str, Any]] = None,
        random_state: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> "AutoCausal":
        return cls(
            df,
            source=source,
            mode=mode,
            strict=strict,
            policy=policy,
            random_state=random_state,
            run_id=run_id,
        )

    @classmethod
    def connect(cls, url: Optional[str] = None, /, **kwargs: Any) -> Any:
        """Unified DB connect — see autocausal.db.connect."""
        from autocausal.db import connect

        return connect(url, **kwargs)

    @classmethod
    def ping(cls, url: str, *, timeout: float = 5.0) -> Any:
        from autocausal.db import ping

        return ping(url, timeout=timeout)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def policy_dict(self) -> dict[str, Any]:
        """Return the active serializable run policy."""
        return self.policy.to_dict()

    def manifest_dict(self) -> dict[str, Any]:
        """Return the current privacy-safe run manifest."""
        return self.run_manifest.to_dict()

    def correlate(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        *,
        columns: Optional[Sequence[str]] = None,
        method: str = "auto",
        controls: Optional[Sequence[str]] = None,
        weights: Optional[str] = None,
        cluster: Optional[str] = None,
        bootstrap_n: int = 0,
        permutation_n: int = 0,
        alpha: float = 0.05,
    ) -> Any:
        """Run descriptive association analysis; never causal identification.

        Pass ``x`` and ``y`` for one typed association, or omit both for a
        pairwise matrix scan with BH-FDR q-values.
        """
        from autocausal.correlation import correlation, correlation_matrix

        if (x is None) != (y is None):
            raise ValueError("Pass both x= and y=, or omit both for a matrix.")
        with self._recorder.span(
            "correlation",
            method=method,
            matrix=x is None,
            n_columns=len(columns or self._df.columns),
        ):
            if x is None:
                result = correlation_matrix(
                    self._df,
                    columns=columns,
                    method=method,
                    alpha=alpha,
                    bootstrap_n=bootstrap_n,
                    permutation_n=permutation_n,
                    random_state=self.random_state,
                )
            else:
                result = correlation(
                    x,
                    y,
                    data=self._df,
                    method=method,
                    controls=controls,
                    weights=weights,
                    cluster=cluster,
                    bootstrap_n=bootstrap_n,
                    permutation_n=permutation_n,
                    random_state=self.random_state,
                )
        self.correlation_results.append(result)
        self.run_manifest.config.setdefault("correlations", []).append(
            {
                "x": x,
                "y": y,
                "columns": list(columns or []),
                "method": method,
                "controls": list(controls or []),
                "weights": weights,
                "cluster": cluster,
                "bootstrap_n": bootstrap_n,
                "permutation_n": permutation_n,
                "alpha": alpha,
                "identification_evidence": False,
            }
        )
        return result

    def infer(
        self,
        *,
        spec: Any,
        method: str = "aipw",
        **kwargs: Any,
    ) -> Any:
        """Fit an explicit causal design through the unified inference API."""
        from autocausal.inference import AutoInference, CausalSpec

        resolved_spec = (
            spec
            if isinstance(spec, CausalSpec)
            else CausalSpec.from_dict(dict(spec))
        )
        inference = AutoInference(
            resolved_spec,
            policy=self.policy,
            mode=self.mode,
            random_state=self.random_state,
        )
        result = inference.fit(self._df, method=method, **kwargs)
        self.causal_inference_results.append(result)
        self.run_manifest.config.setdefault("unified_inference", []).append(
            {
                "method": result.method,
                "estimand": result.estimand,
                "run_id": result.provenance.get("run_id"),
                "evidence_grade": result.evidence_grade,
            }
        )
        self.run_manifest.gates.extend(result.gates.results)
        return result

    def production_check(
        self,
        *,
        treatment: str,
        outcome: str,
        instrument: Optional[str | Sequence[str]] = None,
        confounders: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run strict cleanse/EDA/statistical gates without choosing a method."""
        from autocausal.production import (
            ProductionPolicy,
            run_production_pipeline,
        )

        effective_policy = (
            self.policy
            if self.policy.profile == "production"
            else ProductionPolicy.strict(random_state=self.random_state)
        )
        return run_production_pipeline(
            self._df,
            treatment=treatment,
            outcome=outcome,
            instrument=instrument,
            confounders=confounders,
            policy=effective_policy,
            random_state=self.random_state,
            method=None,
            dry_run_cleanse=True,
            **kwargs,
        )

    def run_production(
        self,
        *,
        treatment: str,
        outcome: str,
        method: Optional[str] = None,
        instrument: Optional[str | Sequence[str]] = None,
        confounders: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run the unified production pipeline; no method means review-only."""
        from autocausal.production import (
            ProductionPolicy,
            run_production_pipeline,
        )

        effective_policy = (
            self.policy
            if self.policy.profile == "production"
            else ProductionPolicy.strict(random_state=self.random_state)
        )
        run = run_production_pipeline(
            self._df,
            treatment=treatment,
            outcome=outcome,
            method=method,
            instrument=instrument,
            confounders=confounders,
            policy=effective_policy,
            random_state=self.random_state,
            **kwargs,
        )
        self.cleanse_report = run.cleanse_report
        self.eda_report = run.eda_report
        if run.inference_result is not None:
            self.causal_inference_results.append(run.inference_result)
        self.run_manifest = run.manifest
        return run

    def external_payload(self, *, include_frame: bool = False) -> dict[str, Any]:
        """Build an MCP/SLM-safe session payload.

        Raw frames are denied by default and always denied in production unless
        the reviewed policy explicitly opts in.
        """
        if include_frame and not self.policy.allow_raw_data_external:
            raise UnsafePayloadError(
                "Raw frame export to MCP/SLM payload is disabled by policy.",
                code="raw_external_payload_forbidden",
                recommendations=[
                    "Use manifest/fingerprint summaries or explicitly review "
                    "allow_raw_data_external=True."
                ],
                manifest=self.run_manifest,
            )
        payload: dict[str, Any] = {
            "schema": "AutoCausalExternalContext.v1",
            "run_id": self.run_id,
            "mode": self.mode,
            "manifest": self.run_manifest.to_dict(),
            "columns": [str(column) for column in self._df.columns],
            "shape": [int(len(self._df)), int(len(self._df.columns))],
            "result": self.result.to_dict() if self.result is not None else None,
            "contains_raw_values": bool(include_frame),
        }
        if include_frame:
            payload["frame"] = self._df.to_dict(orient="records")
        return payload

    def _assert_slm_allowed(self, requested: Optional[bool]) -> bool:
        """SLM guides by default; only an explicit False turns it off.

        ``None`` resolves to True when the active policy allows SLM.
        """
        from autocausal.suites.director import resolve_suite_slm

        if requested is False:
            return False
        use_slm = resolve_suite_slm(True if requested is None else requested)
        if use_slm and not self.policy.allow_slm:
            raise UnsafePayloadError(
                "SLM execution is disabled by the active production policy.",
                code="slm_forbidden",
                recommendations=[
                    "Set policy.allow_slm=True (default), or pass use_slm=False "
                    "to force deterministic rule guidance only."
                ],
                manifest=self.run_manifest,
            )
        return use_slm

    def _validate_slm_result(self, result: Any, *, requested: bool) -> None:
        """Prefer SLM guidance; allow audited rule soft-fallback when needed."""
        if not (requested and is_production(self.mode)):
            return
        payload = (
            result.to_dict()
            if hasattr(result, "to_dict")
            else {"result": str(result)}
        )
        backend = str(payload.get("backend") or "").lower()
        notes_blob = " ".join(str(note) for note in payload.get("notes") or []).lower()
        raw_text = str(payload.get("raw_text") or "")
        soft_fallback = (
            backend in ("", "rule", "fallback", "missing")
            or "heuristic" in notes_blob
            or ("soft" in notes_blob and "fallback" in notes_blob)
            or bool(raw_text)
        )
        if soft_fallback:
            detail = (
                f"SLM guidance soft-fell back to backend={backend or 'rule'}"
                + ("; raw SLM text ignored in favor of structured rules." if raw_text else ".")
                + " Deterministic rules still guided the step."
            )
            gate = GateResult(
                id="slm_soft_fallback",
                ok=True,
                detail=detail,
                recommendation="Install/configure a structured local SLM for fuller guidance.",
            )
            self.run_manifest.gates.append(gate)

    def _limit_work_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Apply policy row/column limits without silently truncating production."""
        warnings: list[str] = []
        row_over = len(frame) > self.policy.max_rows
        col_over = len(frame.columns) > self.policy.max_columns
        if not (row_over or col_over):
            return frame, warnings
        details = (
            f"frame={len(frame)}x{len(frame.columns)} exceeds policy "
            f"max_rows={self.policy.max_rows}, max_columns={self.policy.max_columns}"
        )
        gate = GateResult(
            id="resource_shape_limit",
            ok=False,
            detail=details,
            recommendation="Sample/select columns explicitly or raise reviewed limits.",
        )
        self.run_manifest.gates.append(gate)
        if is_production(self.mode) or self.policy.fallback_behavior == "fail":
            self.run_manifest.finish("aborted")
            raise ResourceLimitError(
                details,
                code="shape_limit_exceeded",
                gates=[gate],
                recommendations=[gate.recommendation or ""],
                manifest=self.run_manifest,
            )
        work = frame
        if row_over:
            work = work.sample(
                n=self.policy.max_rows,
                random_state=self.random_state,
            ).sort_index()
            warnings.append(
                f"EXPLORATORY fallback: deterministically sampled {self.policy.max_rows} rows."
            )
        if col_over:
            work = work.iloc[:, : self.policy.max_columns]
            warnings.append(
                f"EXPLORATORY fallback: retained first {self.policy.max_columns} columns."
            )
        return work, warnings

    def _limit_rounds(self, requested: int) -> int:
        rounds = max(1, int(requested))
        if rounds <= self.policy.max_rounds:
            return rounds
        gate = GateResult(
            id="max_rounds",
            ok=False,
            detail=(
                f"requested max_rounds={rounds} exceeds policy "
                f"max_rounds={self.policy.max_rounds}"
            ),
            recommendation="Reduce rounds or raise the reviewed policy limit.",
        )
        self.run_manifest.gates.append(gate)
        if is_production(self.mode) or self.policy.fallback_behavior == "fail":
            raise ResourceLimitError(
                gate.detail,
                code="round_limit_exceeded",
                gates=[gate],
                recommendations=[gate.recommendation or ""],
                manifest=self.run_manifest,
            )
        self.run_manifest.warnings.append(
            f"EXPLORATORY fallback: capped rounds at {self.policy.max_rounds}."
        )
        return self.policy.max_rounds

    def join_public(
        self,
        public_id: Union[str, list[str]],
        *,
        on: Optional[Union[str, list[str]]] = None,
        how: str = "left",
        allow_network: bool = False,
    ) -> "AutoCausal":
        """Join one or more public-suite tables into the current frame."""
        from autocausal.public_suite import join_public_frames

        joined, log = join_public_frames(
            self._df,
            public_id,
            on=on,
            how=how,
            allow_network=allow_network,
        )
        self._df = joined
        self.join_log.extend(log)
        self.roles = infer_column_roles(self._df)
        self.source = f"{self.source}+join:{public_id}"
        return self

    @classmethod
    def mine_public(
        cls,
        sources: Optional[Union[str, list[str]]] = None,
        *,
        join_on: Optional[Union[str, list[str]]] = None,
        how: str = "outer",
        allow_network: bool = False,
        discover: bool = True,
        use_iv: bool = True,
        min_score: float = 0.15,
        min_abs_corr: float = 0.12,
        validate: bool = False,
        base: Optional[pd.DataFrame] = None,
    ) -> Any:
        """Mine + discover causal edges across public suite sources.

        Returns a :class:`~autocausal.public_causal.PublicCausalReport`.
        See also :class:`~autocausal.public_causal.PublicCausalMiner`.
        """
        from autocausal.public_causal import mine_public

        return mine_public(
            sources,
            join_on=join_on,
            how=how,
            allow_network=allow_network,
            discover=discover,
            use_iv=use_iv,
            min_score=min_score,
            min_abs_corr=min_abs_corr,
            validate=validate,
            base=base,
        )

    def join_frames(
        self,
        *frames: pd.DataFrame,
        keys: Optional[Union[str, list[str]]] = None,
        how: str = "outer",
    ) -> "AutoCausal":
        """Generic multi-frame align into the current frame (see ``autocausal.join.align``)."""
        from autocausal.join import align

        all_frames = [self._df, *frames]
        joined, report = align(all_frames, keys=keys, how=how)
        self._df = joined
        self.join_log.append(report.to_dict())
        self.roles = infer_column_roles(self._df)
        self.source = f"{self.source}+align:{report.keys}"
        return self

    def validate_qc(
        self,
        *,
        key_columns: Optional[Sequence[str]] = None,
        mode: QCMode = "warn",
    ) -> Any:
        """Run QC gate on the current frame. See ``autocausal.qc.validate_frame``."""
        from autocausal.qc import validate_frame

        report = validate_frame(self._df, key_columns=key_columns)
        self.qc_report = report
        if mode == "block" and report.blocked:
            codes = [i.code for i in report.block_issues()]
            gate = GateResult(
                id="qc_block",
                ok=False,
                detail=f"QC blocked discovery: {codes}",
                recommendation="Resolve blocking QC issues before production discovery.",
            )
            self.run_manifest.gates.append(gate)
            if is_production(self.mode):
                self.run_manifest.finish("aborted")
                raise ProductionGateError(
                    gate.detail,
                    code="qc_blocked",
                    gates=[gate],
                    recommendations=[gate.recommendation or ""],
                    manifest=self.run_manifest,
                )
            raise ValueError(f"{gate.detail}. See ac.qc_report.")
        privacy = privacy_scan(self._df)
        for warning in privacy.get("warnings") or []:
            if warning not in report.notes:
                report.notes.append(warning)
            if warning not in self.run_manifest.warnings:
                self.run_manifest.warnings.append(warning)
        self.run_manifest.privacy = privacy
        if (
            is_production(self.mode)
            and self.policy.fail_on_pii
            and privacy.get("pii_columns")
        ):
            gate = GateResult(
                id="pii_detected",
                ok=False,
                detail=(
                    "Potential PII columns: "
                    + ", ".join(privacy.get("pii_columns") or [])
                ),
                recommendation="Remove/tokenize PII or use a reviewed privacy policy.",
            )
            self.run_manifest.gates.append(gate)
            self.run_manifest.finish("aborted")
            raise ProductionGateError(
                gate.detail,
                code="pii_gate_failed",
                gates=[gate],
                recommendations=[gate.recommendation or ""],
                manifest=self.run_manifest,
            )
        return report

    def enrich_from_text(self, text: str) -> "AutoCausal":
        """Extract NLP causal hints and merge into guide/direct context."""
        from autocausal.nlp import TextCausalHints

        hints = TextCausalHints.extract(text or "")
        self.nlp_hints = hints
        return self

    def mine(self, *, min_score: float = 0.15) -> "AutoCausal":
        from autocausal.mining import mine

        self.mining = mine(self._df, min_score=min_score)
        return self

    def cleanse(
        self,
        *,
        use_slm: Optional[bool] = None,
        text: str = "",
        impute: str = "auto",
        **kwargs: Any,
    ) -> "AutoCausal":
        """Run :class:`~autocausal.suites.AutoCleanseSuite` (SLM-directed when available).

        Stores ``cleanse_report`` and replaces the working frame with the cleaned frame.
        ``use_slm`` defaults to try-SLM (soft rule fallback); pass ``False`` to force rules.
        """
        from autocausal.suites import AutoCleanseSuite

        requested = use_slm if use_slm is not None else self._suite_use_slm
        slm = self._assert_slm_allowed(requested)
        self._suite_use_slm = slm
        suite = AutoCleanseSuite(
            self,
            use_slm=slm,
            text=text,
            impute=impute,  # type: ignore[arg-type]
            **kwargs,
        ).run()
        assert suite.frame is not None and suite.report is not None
        self._df = suite.frame
        self.cleanse_report = suite.report
        if suite.report.imputation:
            # keep a light pointer; full ImputationReport may be nested in dict
            pass
        self.roles = infer_column_roles(self._df)
        self.source = f"{self.source}+cleanse"
        return self

    def eda(
        self,
        *,
        use_slm: Optional[bool] = None,
        text: str = "",
        **kwargs: Any,
    ) -> "AutoCausal":
        """Run :class:`~autocausal.suites.AutoEDASuite`; stores ``eda_report``."""
        from autocausal.suites import AutoEDASuite

        requested = use_slm if use_slm is not None else self._suite_use_slm
        slm = self._assert_slm_allowed(requested)
        self._suite_use_slm = slm
        suite = AutoEDASuite(self, use_slm=slm, text=text, **kwargs).run()
        self.eda_report = suite.report
        self.source = f"{self.source}+eda"
        return self

    def automine(
        self,
        *,
        use_slm: Optional[bool] = None,
        text: str = "",
        min_score: float = 0.15,
        join_public: Optional[Union[str, list[str]]] = None,
        **kwargs: Any,
    ) -> "AutoCausal":
        """Run :class:`~autocausal.suites.AutoMineSuite`; stores ``mine_report`` + ``mining``."""
        from autocausal.suites import AutoMineSuite

        requested = use_slm if use_slm is not None else self._suite_use_slm
        slm = self._assert_slm_allowed(requested)
        self._suite_use_slm = slm
        suite = AutoMineSuite(
            self,
            use_slm=slm,
            text=text,
            min_score=min_score,
            join_public=join_public,
            **kwargs,
        ).run()
        assert suite.frame is not None
        self._df = suite.frame
        self.mine_report = suite.report
        self.mining = suite._mining
        self.roles = infer_column_roles(self._df)
        self.source = f"{self.source}+automine"
        return self

    def impute(self, method: ImputeMethod = "auto", *, knn_k: int = 5) -> "AutoCausal":
        self._df, self.imputation = impute_dataframe(self._df, method=method, knn_k=knn_k)
        self.roles = infer_column_roles(self._df)
        return self


    def set_iv_roles(
        self,
        *,
        treatment: Optional[Union[str, Sequence[str]]] = None,
        outcome: Optional[Union[str, Sequence[str]]] = None,
        instrument: Optional[Union[str, Sequence[str]]] = None,
        confounder: Optional[Union[str, Sequence[str]]] = None,
    ) -> "AutoCausal":
        """Pin IV role candidates for subsequent ``discover`` / ``to_causaliv_request``.

        Overrides name heuristics. Pass a string or list per role. Clears a role
        when given an empty list.
        """

        def _as_list(v: Optional[Union[str, Sequence[str]]]) -> Optional[list[str]]:
            if v is None:
                return None
            if isinstance(v, str):
                return [v]
            return [str(x) for x in v]

        for key, val in (
            ("treatment", treatment),
            ("outcome", outcome),
            ("instrument", instrument),
            ("confounder", confounder),
        ):
            parsed = _as_list(val)
            if parsed is not None:
                self._iv_roles[key] = parsed
        return self

    def auto_add_instrument(
        self,
        *,
        treatment: Optional[str] = None,
        seed: Optional[int] = None,
        force: bool = False,
        col: Optional[str] = None,
    ) -> "AutoCausal":
        """Add a synthetic exploratory instrument column to the working frame.

        Prefer real IV columns (``iv_demo``, ``instruments_demo``, ``set_iv_roles``)
        when available. This helper is for demo plumbing when Z is missing.

        Raises
        ------
        ValueError
            In ``mode="production"`` / ``strict=True`` — synthetic Z is forbidden.
        """
        from autocausal.iv import AUTO_INSTRUMENT_COL, synthesize_auto_instrument
        if is_production(self.mode):
            raise EvidenceGateError(
                "Production mode refuses auto_add_instrument() / synthetic Z. "
                "Provide a real instrument column or switch to mode='exploratory'.",
                code="synthetic_iv_forbidden",
                recommendations=["Provide an observed Z and document the IV design."],
                manifest=self.run_manifest,
            )
        resolved_seed = self.random_state if seed is None else int(seed)
        treat = treatment
        if treat is None:
            roles = self._iv_roles or {}
            treat = (roles.get("treatment") or [None])[0]
        if treat is None and self.result is not None:
            treat = (self.result.candidates.get("treatment") or [None])[0]
        if treat is None:
            # name heuristic fallback
            for c in self._df.columns:
                low = str(c).lower()
                if any(h in low for h in ("treat", "treatment", "exposure", "dose")):
                    treat = str(c)
                    break
        if treat is None:
            # last resort: first numeric non-id column
            from autocausal.roles import ColumnRole as CR

            roles_map = infer_column_roles(self._df)
            nums = [c for c, r in roles_map.items() if r == CR.NUMERIC]
            treat = nums[0] if nums else None
        if treat is None:
            raise ValueError("auto_add_instrument requires a treatment column")

        name = col or AUTO_INSTRUMENT_COL
        self._df, notes = synthesize_auto_instrument(
            self._df, treat, seed=resolved_seed, col=name, force=force
        )
        self.roles = infer_column_roles(self._df)
        self._iv_roles.setdefault("instrument", [])
        if name not in self._iv_roles["instrument"]:
            self._iv_roles["instrument"] = [name] + list(self._iv_roles.get("instrument") or [])
        if treatment or treat:
            self._iv_roles.setdefault("treatment", [])
            if treat not in self._iv_roles["treatment"]:
                self._iv_roles["treatment"] = [treat] + list(self._iv_roles.get("treatment") or [])
        # stash notes on a light attribute for discover to pick up
        prev = getattr(self, "_auto_instrument_notes", None) or []
        self._auto_instrument_notes = list(prev) + list(notes)
        return self

    def discover(
        self,
        *,
        alpha: float = 0.05,
        max_cond_size: int = 2,
        min_abs_corr: float = 0.15,
        use_iv: Optional[bool] = None,
        auto_instrument: bool = False,
        allow_iv_fallback: bool = False,
        candidates: Optional[dict[str, Sequence[str]]] = None,
        focus_columns: Optional[list[str]] = None,
        stability: Optional[bool] = None,
        bootstrap_n: Optional[int] = None,
        ensemble: Optional[bool] = None,
        methods: Optional[list[str]] = None,
        method: Optional[str] = None,
        min_methods: Optional[int] = None,
        qc: Optional[QCMode] = None,
        drop_id_columns: bool = True,
        seed: Optional[int] = None,
        random_state: Optional[int] = None,
        include_optional: bool = True,
        mode: Optional[AnalysisMode] = None,
        strict: Optional[bool] = None,
        policy: Optional[ProductionPolicy | dict[str, Any]] = None,
    ) -> DiscoveryResult:
        """Discover exploratory edges; optionally run IV / auto-instrument.

        Parameters
        ----------
        use_iv:
            Attempt IV / 2SLS edges when treatment, outcome, and instrument
            candidates exist.
        auto_instrument:
            When True (**opt-in**; default False) and instruments are missing but
            treatments and outcomes exist, synthesize ``auto_instrument_z``
            (exploratory demo only; ``identification=none``). Production mode
            refuses synthetic IV entirely.
        allow_iv_fallback:
            When True, propose weak numeric correlates as instrument *candidates*
            if no name-heuristic Z is found (default False — prefer real columns).
        mode / strict:
            ``mode="production"`` or ``strict=True`` → no synthetic IV, QC block
            (unless overridden), prefer ensemble+stability, escalate missing engines.
        candidates:
            Optional role injection, e.g.
            ``{"treatment":[...],"outcome":[...],"instrument":[...]}``.
            Merged with ``set_iv_roles`` (explicit args win).
        """
        from autocausal.production import apply_mode_defaults

        resolved_mode = resolve_mode(
            mode if mode is not None else self.mode,
            strict=strict,
        )
        if policy is not None:
            effective_policy = resolve_policy(
                resolved_mode,
                policy,
                random_state=(
                    random_state
                    if random_state is not None
                    else seed if seed is not None else self.random_state
                ),
            )
        elif resolved_mode != self.mode:
            effective_policy = resolve_policy(
                resolved_mode,
                random_state=(
                    random_state
                    if random_state is not None
                    else seed if seed is not None else self.random_state
                ),
            )
        else:
            effective_policy = resolve_policy(
                resolved_mode,
                self.policy,
                random_state=(
                    random_state
                    if random_state is not None
                    else seed if seed is not None else self.random_state
                ),
            )
        settings = apply_mode_defaults(
            mode=resolved_mode,
            policy=effective_policy,
            auto_instrument=auto_instrument,
            allow_iv_fallback=allow_iv_fallback,
            qc=qc,
            stability=stability,
            bootstrap_n=bootstrap_n,
            ensemble=ensemble,
            use_iv=use_iv,
            min_methods=min_methods,
        )
        self.mode = settings.mode  # type: ignore[assignment]
        self.policy = effective_policy
        self.random_state = int(effective_policy.random_state)
        resolved_seed = self.random_state
        auto_instrument = settings.auto_instrument
        allow_iv_fallback = settings.allow_iv_fallback
        use_iv = settings.use_iv
        stability = settings.stability
        bootstrap_n = settings.bootstrap_n
        ensemble = settings.ensemble
        min_methods = settings.min_methods
        qc_mode: QCMode = settings.qc  # type: ignore[assignment]

        discover_config = {
            "alpha": alpha,
            "max_cond_size": max_cond_size,
            "min_abs_corr": min_abs_corr,
            "use_iv": use_iv,
            "auto_instrument": auto_instrument,
            "allow_iv_fallback": allow_iv_fallback,
            "candidates": {
                str(key): [str(value) for value in values]
                for key, values in (candidates or {}).items()
            },
            "focus_columns": list(focus_columns or []),
            "stability": stability,
            "bootstrap_n": bootstrap_n,
            "ensemble": ensemble,
            "methods": list(methods or []),
            "method": method,
            "min_methods": min_methods,
            "qc": qc_mode,
            "drop_id_columns": drop_id_columns,
            "random_state": resolved_seed,
            "include_optional": include_optional,
            "mode": settings.mode,
        }
        self.run_manifest = build_manifest(
            self._df,
            mode=settings.mode,
            policy=self.policy,
            config={"source": self.source, "discover": discover_config},
        )
        self.run_id = self.run_manifest.run_id
        self._recorder = RunRecorder(self.run_manifest)

        with self._recorder.span("policy", mode=settings.mode):
            if is_production(settings.mode):
                check_required_engines(self.policy, manifest=self.run_manifest)

        work, limit_notes = self._limit_work_frame(self._df)
        self.run_manifest.warnings.extend(limit_notes)
        self._recorder.check_deadline(self.policy.max_seconds)

        if qc_mode != "off":
            with self._recorder.span("qc", mode=qc_mode):
                self.validate_qc(mode=qc_mode)
        self._recorder.check_deadline(self.policy.max_seconds)

        if self.imputation is None and work.isna().any().any():
            with self._recorder.span("impute", method="auto"):
                work, self.imputation = impute_dataframe(work, method="auto")
                if len(work) == len(self._df) and list(work.columns) == list(self._df.columns):
                    self._df = work.copy()
        self._recorder.check_deadline(self.policy.max_seconds)

        if drop_id_columns:
            from autocausal.roles import ColumnRole as CR

            roles_tmp = infer_column_roles(work)
            keep_cols = [c for c, r in roles_tmp.items() if r != CR.ID]
            if len(keep_cols) >= 2:
                work = work[keep_cols]
        if focus_columns:
            keep = [c for c in focus_columns if c in work.columns]
            if len(keep) >= 2:
                work = work[keep]
        self.roles = infer_column_roles(work)

        merged_candidates: dict[str, list[str]] = {}
        if self._iv_roles:
            merged_candidates = {k: list(v) for k, v in self._iv_roles.items() if v}
        if candidates:
            from autocausal.iv import merge_role_candidates

            merged_candidates = merge_role_candidates(merged_candidates, candidates)

        # Persist injected instruments onto the working frame when auto_instrument
        # already ran via helper, or when discover will synthesize.
        discover_kwargs: dict[str, Any] = {
            "alpha": alpha,
            "max_cond_size": max_cond_size,
            "min_abs_corr": min_abs_corr,
            "use_iv": use_iv,
            "auto_instrument": auto_instrument,
            "allow_iv_fallback": allow_iv_fallback,
            "candidates": merged_candidates or None,
            "stability": stability,
            "bootstrap_n": bootstrap_n,
            "seed": resolved_seed,
            "mode": settings.mode,
            "policy": self.policy,
        }

        with self._recorder.span(
            "discover",
            ensemble=bool(ensemble or methods),
            stability=stability,
            bootstrap_n=bootstrap_n,
        ):
            if ensemble or methods:
                result = discover_ensemble(
                    work,
                    roles=self.roles,
                    methods=methods,  # type: ignore[arg-type]
                    min_methods=min_methods,
                    include_optional=include_optional if methods is None else False,
                    **discover_kwargs,
                )
            else:
                result = discover_relationships(
                    work,
                    roles=self.roles,
                    method=method or "score_pc_lite",  # type: ignore[arg-type]
                    **discover_kwargs,
                )
        self._recorder.check_deadline(self.policy.max_seconds, partial_result=result)

        result.imputation = self.imputation
        result.mode = settings.mode
        result.run_id = self.run_id
        result.policy = self.policy.to_dict()
        if self.mining is not None:
            result.mining = self.mining.to_dict() if hasattr(self.mining, "to_dict") else self.mining
        if self.qc_report is not None:
            result.notes = list(result.notes) + [
                f"QC ok={self.qc_report.ok} issues={len(self.qc_report.issues)}"
            ]
        extra_notes = getattr(self, "_auto_instrument_notes", None) or []
        if extra_notes:
            result.notes = list(result.notes) + list(extra_notes)
        result.notes = list(result.notes) + list(settings.notes) + limit_notes
        for note in result.notes:
            if "fallback" in str(note).lower() or "soft-skip" in str(note).lower():
                warning = f"Discovery fallback: {note}"
                if warning not in self.run_manifest.warnings:
                    self.run_manifest.warnings.append(warning)

        with self._recorder.span("evidence_gates"):
            accepted, rejected, evidence_gates = annotate_and_gate_edges(
                result.edges,
                source=self.source,
                run_id=self.run_id,
                method=result.method,
                policy=self.policy,
                mode=settings.mode,
                source_columns=[str(column) for column in work.columns],
            )
            result.edges = accepted
            result.rejected_edges = rejected
            result.evidence_gates = [gate.to_dict() for gate in evidence_gates]
            self.run_manifest.gates.extend(evidence_gates)
            if rejected:
                result.notes.append(
                    f"Evidence gates rejected {len(rejected)} edge(s); "
                    "see rejected_edges / evidence_gates."
                )
            # The graph must not expose production-rejected edges as accepted.
            result.graph = dict(result.graph)
            result.graph["edges"] = [dict(edge) for edge in result.edges]

        if self.sensitivity_report is not None:
            result.sensitivity_report = (
                self.sensitivity_report.to_dict()
                if hasattr(self.sensitivity_report, "to_dict")
                else self.sensitivity_report
            )
        # If auto_instrument added a column inside discovery, mirror onto session frame
        auto_z = "auto_instrument_z"
        bind_frame = work.copy()
        if auto_z in (result.candidates.get("instrument") or []) and not is_production(
            settings.mode
        ):
            from autocausal.iv import synthesize_auto_instrument

            treat = (result.candidates.get("treatment") or [None])[0]
            if treat and treat in self._df.columns and auto_z not in self._df.columns:
                self._df, _ = synthesize_auto_instrument(
                    self._df, treat, seed=resolved_seed
                )
                self.roles = infer_column_roles(self._df)
            if treat and treat in bind_frame.columns and auto_z not in bind_frame.columns:
                bind_frame, _ = synthesize_auto_instrument(
                    bind_frame, treat, seed=resolved_seed
                )
        # Bind session + working frame so result.estimate / refute / fabric work standalone
        result.bind_session(self, frame=bind_frame, source=self.source)
        result.manifest = self.run_manifest
        self.result = result
        self._recorder.check_deadline(self.policy.max_seconds, partial_result=result)
        if is_production(settings.mode) and not result.edges:
            gate = GateResult(
                id="no_production_eligible_edges",
                ok=False,
                detail=(
                    "No edge passed the configured production evidence gates "
                    f"(required_evidence={self.policy.required_evidence})."
                ),
                recommendation=(
                    "Collect stronger evidence, increase bootstrap/method agreement, "
                    "or review a less strict policy without calling results identified."
                ),
            )
            result.evidence_gates.append(gate.to_dict())
            self.run_manifest.gates.append(gate)
            self.run_manifest.finish("failed")
            raise EvidenceGateError(
                gate.detail,
                code="no_supported_edges",
                gates=[gate, *evidence_gates],
                recommendations=[gate.recommendation or ""],
                partial_result=result,
                manifest=self.run_manifest,
            )
        self.run_manifest.finish("ok")
        return result

    def discover_ensemble(self, **kwargs: Any) -> DiscoveryResult:
        """Multi-method consensus discovery (pc_lite + corr_skeleton + mi_binned)."""
        kwargs.setdefault("ensemble", True)
        return self.discover(**kwargs)

    def _guide_context(self, text: Optional[str] = None) -> dict[str, Any]:
        if text and self.nlp_hints is None:
            self.enrich_from_text(text)
        elif text and self.nlp_hints is not None:
            # refresh if new text provided
            prev = getattr(self.nlp_hints, "text", None)
            if prev != text:
                self.enrich_from_text(text)

        ctx: dict[str, Any] = {
            "text": text or (getattr(self.nlp_hints, "text", "") if self.nlp_hints else ""),
            "columns": (
                self.mining.columns
                if self.mining is not None
                else [{"name": c} for c in self._df.columns]
            ),
            "associations": self.mining.associations if self.mining is not None else [],
            "edges": self.result.edges if self.result is not None else [],
            "candidates": self.result.candidates if self.result is not None else {},
        }
        if self.nlp_hints is not None:
            nlp_ctx = (
                self.nlp_hints.to_guide_context()
                if hasattr(self.nlp_hints, "to_guide_context")
                else {}
            )
            # merge NLP candidates into guide context without clobbering discovery candidates
            ctx["nlp_hints"] = (
                self.nlp_hints.to_dict() if hasattr(self.nlp_hints, "to_dict") else self.nlp_hints
            )
            nlp_cands = nlp_ctx.get("candidates") or {}
            merged = dict(ctx.get("candidates") or {})
            for role, items in nlp_cands.items():
                # normalize plural keys from NLP
                key = {
                    "confounders": "confounder",
                    "instruments": "instrument",
                }.get(role, role)
                existing = list(merged.get(key) or [])
                for item in items or []:
                    if item not in existing:
                        existing.append(item)
                merged[key] = existing
            ctx["candidates"] = merged
            if nlp_ctx.get("focus_columns"):
                ctx["focus_columns"] = nlp_ctx["focus_columns"]
            ctx["modality_markers"] = nlp_ctx.get("modality_markers") or []
            notes = list(ctx.get("notes") or [])
            notes.extend(nlp_ctx.get("notes") or [])
            ctx["notes"] = notes
        return ctx

    def guide(
        self,
        *,
        text: Optional[str] = None,
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        backends: Optional[list[str]] = None,
    ) -> Any:
        """Run guide backends; SLM guides by default (soft-falls to rules)."""
        use_slm = self._assert_slm_allowed(use_slm)
        if text:
            self.enrich_from_text(text)
        context = self._guide_context(text)
        if backends:
            from autocausal.guides import direct

            plan = direct(
                context, backends=backends, use_slm=use_slm, model_name=model_name
            )
            self.direction_plan = plan
            self.guide_result = plan.as_guide_result()
        else:
            from autocausal.slm import guide_pipeline

            self.guide_result = guide_pipeline(
                context, use_slm=use_slm, model_name=model_name
            )
        self._validate_slm_result(self.guide_result, requested=use_slm)
        if self.result is not None:
            self.result.guide = self.guide_result.to_dict()
            if self.direction_plan is not None:
                self.result.guide["direction_plan"] = self.direction_plan.to_dict()
        return self.guide_result

    def direct(
        self,
        *,
        text: Optional[str] = None,
        backends: Optional[list[str]] = None,
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        second_pass: bool = True,
        **discover_kwargs: Any,
    ) -> Any:
        """
        Steer causal direction with one or more guide backends.

        SLM guides by default. Merges outputs into a ``DirectionPlan``, optionally
        re-runs discover focused on plan columns (second pass).
        """
        from autocausal.guides import direct as run_direct

        use_slm = self._assert_slm_allowed(use_slm)
        if text:
            self.enrich_from_text(text)
        if self.mining is None:
            self.mine()
        if self.imputation is None and self._df.isna().any().any():
            self.impute(method="auto")
        if self.result is None:
            self.discover(**discover_kwargs)

        names = backends or ["llmintent", "retracement", "kineteq_pivot", "rule"]
        plan = run_direct(
            self._guide_context(text),
            backends=names,
            use_slm=use_slm,
            model_name=model_name,
        )
        self._validate_slm_result(plan, requested=use_slm)
        self.direction_plan = plan
        self.guide_result = plan.as_guide_result()
        if self.result is not None:
            self.result.guide = self.guide_result.to_dict()
            self.result.guide["direction_plan"] = plan.to_dict()

        if second_pass and plan.focus_columns:
            focus = [c for c in plan.focus_columns if c in self._df.columns]
            if len(focus) >= 2:
                self.discover(focus_columns=focus, **discover_kwargs)
                # refresh plan notes after second pass
                plan.notes = list(plan.notes) + [f"second-pass focus: {focus[:12]}"]
                if self.result is not None:
                    self.result.guide = self.guide_result.to_dict()
                    self.result.guide["direction_plan"] = plan.to_dict()
        boost = getattr(plan, "boost_edges", None) or []
        if boost:
            self._merge_boost_edges(boost)
        return plan

    def _merge_boost_edges(self, boost_edges: Sequence[Any]) -> int:
        """Merge boost edges into ``self.result.edges`` when endpoints exist.

        Dedupes by undirected (source, target). Tags method ``grail_boost``.
        Returns number of newly appended edges.
        """
        if self.result is None or not boost_edges:
            return 0
        cols = set(self._df.columns)
        existing = {
            tuple(sorted((str(e.get("source")), str(e.get("target")))))
            for e in (self.result.edges or [])
            if e.get("source") and e.get("target")
        }
        added = 0
        notes = list(self.result.notes or [])
        for raw in boost_edges:
            if not isinstance(raw, dict):
                continue
            src = str(raw.get("source") or raw.get("from") or "")
            tgt = str(raw.get("target") or raw.get("to") or "")
            if not src or not tgt or src not in cols or tgt not in cols:
                continue
            key = tuple(sorted((src, tgt)))
            if key in existing:
                continue
            edge = {
                "source": src,
                "target": tgt,
                "score": float(raw.get("score") or raw.get("weight") or 0.35),
                "confidence": float(raw.get("confidence") or 0.3),
                "pvalue": raw.get("pvalue"),
                "type": str(raw.get("type") or "association"),
                "orientation": str(raw.get("orientation") or "grail_boost"),
                "method": "grail_boost",
            }
            self.result.edges.append(edge)
            existing.add(key)
            added += 1
        if added:
            notes.append(f"Merged {added} boost_edges (method=grail_boost).")
            self.result.notes = notes
            # keep graph.edges in sync when present
            graph = getattr(self.result, "graph", None)
            if isinstance(graph, dict) and "edges" in graph:
                graph["edges"] = [
                    {
                        "source": e["source"],
                        "target": e["target"],
                        "score": e.get("score"),
                        "confidence": e.get("confidence"),
                        "stability": e.get("stability"),
                        "type": e.get("type", "association"),
                        "method": e.get("method"),
                    }
                    for e in self.result.edges
                ]
        return added

    def apply_grail(
        self,
        text: str = "",
        *,
        report: Any = None,
        second_pass: bool = True,
        **discover_kwargs: Any,
    ) -> Any:
        """Run GRAIL and optionally merge focus / boost_edges into discovery.

        If ``report`` is None, runs ``GrailEngine().run(...)``. When
        ``second_pass`` and the report has ``focus_columns``, rediscovers on
        that focus. Merges ``boost_edges`` whose endpoints exist in the frame.
        """
        from autocausal.grail import GrailEngine

        if report is None:
            eng = GrailEngine()
            goal = text or "causal discovery"
            context = {
                "columns": list(self._df.columns),
                "text": text or goal,
                "edges": self.result.edges if self.result is not None else [],
                "candidates": self.result.candidates if self.result is not None else {},
            }
            report = eng.run(goal, context=context)
        self.grail_report = report

        if self.result is None:
            self.discover(**discover_kwargs)

        if second_pass and getattr(report, "focus_columns", None):
            focus = [c for c in report.focus_columns if c in self._df.columns]
            if len(focus) >= 2:
                self.discover(focus_columns=focus, **discover_kwargs)

        boost = list(getattr(report, "boost_edges", None) or [])
        if boost:
            self._merge_boost_edges(boost)
        return report

    def session_snapshot(self) -> dict[str, Any]:
        """Lightweight in-memory session metadata (full DF is NOT persisted)."""
        methods: list[str] = []
        n_edges = 0
        notes: list[str] = []
        if self.result is not None:
            n_edges = len(self.result.edges or [])
            m = getattr(self.result, "method", None)
            if m:
                methods.append(str(m))
            ens = getattr(self.result, "ensemble_methods", None) or []
            for x in ens:
                if str(x) not in methods:
                    methods.append(str(x))
            # collect unique edge methods
            for e in self.result.edges or []:
                em = e.get("method")
                if em and str(em) not in methods:
                    methods.append(str(em))
            notes = list(self.result.notes or [])[:8]
        return {
            "schema": "AutoCausalSessionSnapshot.v1",
            "run_id": self.run_id,
            "mode": self.mode,
            "policy": self.policy.to_dict(),
            "source": self.source,
            "shape": [int(self._df.shape[0]), int(self._df.shape[1])],
            "n_rows": int(self._df.shape[0]),
            "n_cols": int(self._df.shape[1]),
            "columns": [str(c) for c in self._df.columns],
            "has_result": self.result is not None,
            "n_edges": n_edges,
            "methods": methods,
            "has_grail": self.grail_report is not None,
            "manifest": self.run_manifest.to_dict(),
            "notes": notes
            + [
                "Full DataFrame is NOT persisted — in-memory session only.",
                "Use MCP SessionStore for agent multi-step; EpisodeStore/persist_dir for loop JSONL.",
            ],
        }

    def sensitivity(
        self,
        *,
        text: str = "",
        domain: Optional[str] = None,
        n_boot: int = 8,
        seed: Optional[int] = None,
    ) -> Any:
        """Compute sensitivity metrics and attach to discovery / AutoResult."""
        from autocausal.sensitivity import compute_sensitivity

        resolved_seed = self.random_state if seed is None else int(seed)
        edges = self.result.edges if self.result is not None else None
        traj = None
        if self.physics_result is not None:
            traj = getattr(self.physics_result, "trajectory", None)
        report = compute_sensitivity(
            self._df,
            edges=edges,
            trajectory=traj,
            text=text,
            domain=domain,
            n_boot=n_boot,
            seed=resolved_seed,
        )
        self.sensitivity_report = report
        if self.result is not None:
            self.result.sensitivity_report = report.to_dict()
            self.result.notes = list(self.result.notes) + report.to_mine_notes()
        return report

    def to_causaliv_request(
        self,
        *,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        instrument: Optional[str] = None,
        confounders: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        """Structured CausalIV handoff spec (soft if ``causaliv`` missing)."""
        cands = dict(self._iv_roles) if self._iv_roles else {}
        if self.result is not None:
            for k, v in (self.result.candidates or {}).items():
                if k not in cands or not cands[k]:
                    cands[k] = list(v or [])
        y = outcome or (cands.get("outcome") or [None])[0]
        d = treatment or (cands.get("treatment") or [None])[0]
        z = instrument or (cands.get("instrument") or [None])[0]
        w = list(confounders) if confounders is not None else list(cands.get("confounder") or [])

        causaliv_available = False
        try:
            from importlib.util import find_spec

            causaliv_available = find_spec("causaliv") is not None
        except Exception:
            causaliv_available = False

        from autocausal.panel import iv_handoff_notes

        spec = {
            "schema": "CausalIVRequest.v1",
            "produced_by": "autocausal",
            "y": y,
            "d": d,
            "z": z,
            "w": w,
            "n_rows": len(self._df),
            "columns": [str(c) for c in self._df.columns],
            "edges": (self.result.edges[:20] if self.result is not None else []),
            "causaliv_available": causaliv_available,
            "notes": iv_handoff_notes(
                treatment=d, outcome=y, instrument=z, confounders=w
            ),
            "soft": True,
        }
        return spec

    def set_panel(
        self,
        entity: str,
        time: str,
        *,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        covariates: Optional[Sequence[str]] = None,
    ) -> "AutoCausal":
        """Attach a :class:`~autocausal.panel.PanelSpec` and soft DiD notes."""
        from autocausal.panel import PanelSpec, did_handoff_notes

        spec = PanelSpec(
            entity=entity,
            time=time,
            treatment=treatment,
            outcome=outcome,
            covariates=list(covariates or []),
            notes=did_handoff_notes(
                PanelSpec(
                    entity=entity,
                    time=time,
                    treatment=treatment,
                    outcome=outcome,
                    covariates=list(covariates or []),
                )
            ),
        )
        problems = spec.validate(self._df)
        if problems:
            spec.notes = list(spec.notes) + [f"validate: {p}" for p in problems]
        self.panel_spec = spec
        return self

    def panel_features(
        self,
        columns: Sequence[str],
        *,
        kind: Literal["lag", "diff", "within"] = "lag",
        periods: int = 1,
    ) -> Any:
        """Create panel-aware lag/diff/within features (requires ``set_panel``)."""
        from autocausal.panel import panel_diff, panel_lag, panel_within

        if self.panel_spec is None:
            raise ValueError("Call set_panel(entity, time) before panel_features().")
        if kind == "diff":
            feat = panel_diff(self._df, self.panel_spec, columns, periods=periods)
        elif kind == "within":
            feat = panel_within(self._df, self.panel_spec, columns)
        else:
            feat = panel_lag(self._df, self.panel_spec, columns, periods=periods)
        self._df = feat.df
        self.roles = infer_column_roles(self._df)
        return feat

    def estimate(
        self,
        *,
        backend: str = "builtin_ols",
        y: Optional[str] = None,
        d: Optional[str] = None,
        x: Optional[Sequence[str]] = None,
        z: Optional[str] = None,
        mode: Optional[AnalysisMode] = None,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Estimate ATE/CATE via builtin / DoubleML / EconML (soft-optional).

        Primary session API. Equivalent chaining on the discover handle::

            result = ac.discover()
            result.estimate(backend="builtin_ols")

        Production mode requires explicit ``y`` / ``d`` (and ``z`` for IV
        backends) and refuses ``auto_instrument_z``.
        """
        from autocausal.engines import estimate as eng_estimate
        from autocausal.engines import engine_status
        from autocausal.production import is_synthetic_instrument, refuse_synthetic_iv

        resolved = mode if mode is not None else self.mode
        prod = is_production(resolved, strict=strict)
        backend_l = (backend or "").lower()
        if prod:
            if not y or not d:
                raise ProductionGateError(
                    "Production mode requires explicit y= and d= for estimate(). "
                    "Do not rely on heuristic candidates alone.",
                    code="explicit_roles_required",
                    recommendations=["Pass reviewed y= and d= column names."],
                    manifest=self.run_manifest,
                )
            refuse_synthetic_iv(z, mode="production")
            if is_synthetic_instrument(z):
                raise EvidenceGateError(
                    "Production mode refuses auto_instrument_z.",
                    code="synthetic_iv_forbidden",
                    manifest=self.run_manifest,
                )
            if backend_l in ("builtin_2sls", "2sls", "iv") and not z:
                raise EvidenceGateError(
                    "Production IV estimate requires an explicit observed z= instrument.",
                    code="observed_instrument_required",
                    recommendations=["Pass z= for a reviewed observed instrument."],
                    manifest=self.run_manifest,
                )
            engine_alias = {
                "dml": "doubleml",
                "plr": "doubleml",
                "linear_dml": "econml_linear_dml",
                "causal_forest": "econml_causal_forest",
                "cate": "econml_causal_forest",
            }.get(backend_l, backend_l)
            if backend_l not in ("builtin_ols", "builtin_2sls", "2sls", "iv"):
                status = engine_status(engine_alias)
                if not status.get("available"):
                    gate = GateResult(
                        id=f"estimate_engine:{engine_alias}",
                        ok=False,
                        detail=f"Production estimate engine `{engine_alias}` is unavailable.",
                        recommendation="Install auto-causal-lib[causal-extra] or use an available reviewed engine.",
                    )
                    self.run_manifest.gates.append(gate)
                    raise ProductionGateError(
                        gate.detail,
                        code="estimate_engine_missing",
                        gates=[gate],
                        recommendations=[gate.recommendation or ""],
                        manifest=self.run_manifest,
                    )

        candidates = (
            None
            if prod
            else self.result.candidates if self.result is not None else None
        )
        resolved_x = (
            list(x)
            if x is not None
            else [] if prod else None
        )
        kwargs.setdefault("random_state", self.random_state)
        unified_method = {
            "builtin_ols": "regression",
            "builtin_2sls": "iv_2sls",
            "2sls": "iv_2sls",
            "iv": "iv_2sls",
            "doubleml": "doubleml",
            "dml": "doubleml",
            "plr": "doubleml",
            "econml": "econml_linear_dml",
            "linear_dml": "econml_linear_dml",
            "econml_linear_dml": "econml_linear_dml",
            "causal_forest": "econml_causal_forest",
            "econml_causal_forest": "econml_causal_forest",
        }.get(backend_l)
        self.run_manifest.config.setdefault("estimates", []).append(
            {
                "backend": backend,
                "unified_method_mapping": unified_method,
                "y": y,
                "d": d,
                "x": list(x or []),
                "z": z,
                "random_state": self.random_state,
            }
        )
        with self._recorder.span(
            "estimate",
            backend=backend,
            y=y,
            d=d,
            z=z,
        ):
            result = eng_estimate(
                self._df,
                backend=backend,
                y=y,
                d=d,
                x=resolved_x,
                z=z,
                candidates=candidates,
                mode=resolved,
                **kwargs,
            )
        if prod:
            notes = list(getattr(result, "notes", None) or [])
            notes.append(
                "PRODUCTION estimate: explicit roles passed; assumptions remain unverified."
            )
            if hasattr(result, "notes"):
                result.notes = notes
            if getattr(result, "soft_skip", False) or not getattr(result, "ok", False):
                gate = GateResult(
                    id="estimate_soft_fallback",
                    ok=False,
                    detail=(
                        f"Estimator `{backend}` returned soft_skip="
                        f"{getattr(result, 'soft_skip', False)} / ok="
                        f"{getattr(result, 'ok', False)}."
                    ),
                    recommendation="Resolve estimator inputs/dependencies; production refuses soft fallback.",
                )
                self.run_manifest.gates.append(gate)
                raise ProductionGateError(
                    gate.detail,
                    code="estimate_failed_closed",
                    gates=[gate],
                    recommendations=[gate.recommendation or ""],
                    partial_result=result,
                    manifest=self.run_manifest,
                )
            estimate_payload = getattr(result, "estimate", None) or {}
            first_stage = (
                estimate_payload.get("first_stage_f")
                if isinstance(estimate_payload, dict)
                else None
            )
            if z and first_stage is not None and float(first_stage) < self.policy.min_first_stage_f:
                gate = GateResult(
                    id="instrument_strength",
                    ok=False,
                    detail=(
                        f"first_stage_f={float(first_stage):.3f} < "
                        f"{self.policy.min_first_stage_f:.3f}"
                    ),
                    recommendation="Use a stronger observed instrument or do not report IV.",
                )
                self.run_manifest.gates.append(gate)
                raise EvidenceGateError(
                    gate.detail,
                    code="weak_instrument",
                    gates=[gate],
                    recommendations=[gate.recommendation or ""],
                    partial_result=result,
                    manifest=self.run_manifest,
                )
        elif getattr(result, "soft_skip", False):
            self.run_manifest.warnings.append(
                f"EXPLORATORY estimator `{backend}` soft-skipped."
            )
        if not hasattr(self, "estimate_results") or self.estimate_results is None:
            self.estimate_results = []
        self.estimate_results.append(result)
        if self.result is not None:
            self.result.estimate_results.append(result)
            for edge_item in self.result.edges:
                if edge_item.get("source") == d and edge_item.get("target") == y:
                    provenance = dict(edge_item.get("provenance") or {})
                    provenance["estimator"] = str(
                        getattr(result, "backend", None)
                        or getattr(result, "method", backend)
                    )
                    edge_item["provenance"] = provenance
        self._recorder.check_deadline(
            self.policy.max_seconds,
            partial_result=self.result,
        )
        self.run_manifest.finish("ok")
        return result

    def refute(
        self,
        edge: Optional[dict[str, Any]] = None,
        *,
        method: str = "placebo",
        y: Optional[str] = None,
        d: Optional[str] = None,
        mode: Optional[AnalysisMode] = None,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Soft refute hook (DoWhy real path / builtin placebo) — no-ops gracefully.

        Primary session API. Equivalent chaining on the discover handle::

            result = ac.discover()
            result.refute(method="placebo")

        Production mode requires explicit ``y`` / ``d`` (or an edge with both)
        and refuses synthetic instruments.
        """
        from autocausal.engines import engine_status
        from autocausal.production import is_synthetic_instrument
        from autocausal.suite_tools import refute as suite_refute

        resolved = mode if mode is not None else self.mode
        prod = is_production(resolved, strict=strict)
        if edge is None and self.result is not None and self.result.edges:
            edge = self.result.edges[0]
        if prod:
            if not y or not d:
                raise ProductionGateError(
                    "Production mode requires explicit y= and d= for refute().",
                    code="explicit_roles_required",
                    recommendations=["Pass reviewed y= and d= column names."],
                    manifest=self.run_manifest,
                )
            z = kwargs.get("z") or (edge or {}).get("instrument")
            if is_synthetic_instrument(z):
                raise EvidenceGateError(
                    "Production mode refuses refute() on synthetic auto_instrument_z.",
                    code="synthetic_iv_forbidden",
                    manifest=self.run_manifest,
                )
            method_l = (method or "").lower()
            if method_l.startswith("dowhy") or method_l in (
                "placebo_treatment_refuter",
                "data_subset_refuter",
            ):
                if not engine_status("dowhy").get("available"):
                    gate = GateResult(
                        id="refute_engine:dowhy",
                        ok=False,
                        detail="Production refute requested DoWhy, but it is unavailable.",
                        recommendation="Install auto-causal-lib[causal-extra].",
                    )
                    self.run_manifest.gates.append(gate)
                    raise ProductionGateError(
                        gate.detail,
                        code="refute_engine_missing",
                        gates=[gate],
                        recommendations=[gate.recommendation or ""],
                        manifest=self.run_manifest,
                    )
        candidates = None
        if self.result is not None:
            candidates = self.result.candidates
        kwargs.setdefault("seed", self.random_state)
        self.run_manifest.config.setdefault("refutations", []).append(
            {
                "method": method,
                "y": y,
                "d": d,
                "random_state": self.random_state,
            }
        )
        with self._recorder.span("refute", method=method, y=y, d=d):
            result = suite_refute(
                edge or {},
                method=method,
                df=self._df,
                candidates=candidates,
                y=y,
                d=d,
                **kwargs,
            )
        if prod:
            notes = list(getattr(result, "notes", None) or [])
            notes.append("PRODUCTION refute: explicit roles required.")
            if hasattr(result, "notes"):
                result.notes = notes
            if getattr(result, "soft_skip", False) or not getattr(result, "ok", False):
                gate = GateResult(
                    id="refute_soft_fallback",
                    ok=False,
                    detail=(
                        f"Refuter `{method}` returned soft_skip="
                        f"{getattr(result, 'soft_skip', False)} / ok="
                        f"{getattr(result, 'ok', False)}."
                    ),
                    recommendation="Resolve refuter inputs/dependencies; production refuses soft fallback.",
                )
                self.run_manifest.gates.append(gate)
                raise ProductionGateError(
                    gate.detail,
                    code="refute_failed_closed",
                    gates=[gate],
                    recommendations=[gate.recommendation or ""],
                    partial_result=result,
                    manifest=self.run_manifest,
                )
        elif getattr(result, "soft_skip", False):
            self.run_manifest.warnings.append(
                f"EXPLORATORY refuter `{method}` soft-skipped."
            )
        self.refute_results.append(result)
        if self.result is not None:
            self.result.refute_results.append(result)
            for edge_item in self.result.edges:
                if edge_item.get("source") == d and edge_item.get("target") == y:
                    provenance = dict(edge_item.get("provenance") or {})
                    refuters = list(provenance.get("refuters") or [])
                    refuter_name = str(
                        getattr(result, "backend", None)
                        or getattr(result, "method", method)
                    )
                    if refuter_name not in refuters:
                        refuters.append(refuter_name)
                    provenance["refuters"] = refuters
                    edge_item["provenance"] = provenance
                    data = getattr(result, "data", None) or {}
                    if isinstance(data, dict) and data.get("refute_passed") is False:
                        edge_item["evidence_grade"] = "refuted"
                        edge_item["refuted"] = True
        self._recorder.check_deadline(
            self.policy.max_seconds,
            partial_result=self.result,
        )
        self.run_manifest.finish("ok")
        return result
    def engines_status(self) -> dict[str, Any]:
        """Unified discovery/estimate/refute/package engine status."""
        from autocausal.engines import engine_status

        return engine_status()

    def to_fabric_bundle(self, *, insight: Any = None) -> dict[str, Any]:
        """Export MineReport + CausalEdges (+ optional InsightPack) Fabric bundle."""
        from autocausal.contracts import fabric_bundle

        return fabric_bundle(
            mining=self.mining,
            discovery=self.result,
            insight=insight,
            n_rows=len(self._df),
            n_cols=len(self._df.columns),
            source=self.source,
            sensitivity=self.sensitivity_report,
            extra={
                "qc": self.qc_report.to_dict() if self.qc_report is not None else None,
                "nlp_hints": (
                    self.nlp_hints.to_dict()
                    if self.nlp_hints is not None and hasattr(self.nlp_hints, "to_dict")
                    else None
                ),
                "inference": [
                    result.to_fabric_metadata()["inference"]
                    for result in self.causal_inference_results
                    if hasattr(result, "to_fabric_metadata")
                ],
                "correlation": [
                    {
                        "schema": getattr(result, "schema", None),
                        "epistemic_notice": getattr(
                            result, "epistemic_notice", None
                        ),
                        "pair_count": len(getattr(result, "results", []) or []),
                        "measure": getattr(result, "measure", None),
                    }
                    for result in self.correlation_results
                ],
            },
        )
    def create(
        self,
        *,
        text: Optional[str] = None,
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> Any:
        """SLM/rule *creation*: propose questions, instruments, morphemes.

        SLM guides by default and soft-falls to rules when unavailable.
        """
        from autocausal.slm import create_from_context

        use_slm = self._assert_slm_allowed(use_slm)
        context: dict[str, Any] = {
            "text": text or "",
            "columns": (
                self.mining.columns
                if self.mining is not None
                else [{"name": c} for c in self._df.columns]
            ),
            "candidates": self.result.candidates if self.result is not None else {},
            "edges": self.result.edges if self.result is not None else [],
        }
        if extra_context:
            context.update(extra_context)
        self.creation_result = create_from_context(
            context, use_slm=use_slm, model_name=model_name
        )
        self._validate_slm_result(self.creation_result, requested=use_slm)
        return self.creation_result

    def interpret(
        self,
        *,
        text: Optional[str] = None,
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> Any:
        """SLM/rule *inference* narrative + caveats over discovery/IV results.

        SLM guides by default and soft-falls to rules when unavailable.
        """
        from autocausal.slm import infer_from_results

        use_slm = self._assert_slm_allowed(use_slm)
        iv = None
        if self.result is not None:
            for e in self.result.edges or []:
                if e.get("type") == "iv_2sls" or e.get("first_stage_f") is not None:
                    iv = {
                        "coef": e.get("score"),
                        "first_stage_f": e.get("first_stage_f"),
                        "pvalue": e.get("pvalue"),
                        "instrument": e.get("instrument"),
                    }
                    break
        context: dict[str, Any] = {
            "text": text or "",
            "edges": self.result.edges if self.result is not None else [],
            "iv": iv,
            "candidates": self.result.candidates if self.result is not None else {},
        }
        if extra_context:
            context.update(extra_context)
        self.inference_result = infer_from_results(
            context, use_slm=use_slm, model_name=model_name
        )
        self._validate_slm_result(self.inference_result, requested=use_slm)
        return self.inference_result

    def validate_tools(
        self,
        *,
        y: Optional[str] = None,
        d: Optional[str] = None,
        z: Optional[str] = None,
        claims_text: str = "",
    ) -> Any:
        """Run suite_tools.validate_pipeline on current frame/result."""
        from autocausal.suite_tools import validate_pipeline

        report: dict[str, Any] = {}
        if self.mining is not None and hasattr(self.mining, "to_dict"):
            report.update(self.mining.to_dict())
        if self.result is not None:
            report["edges"] = self.result.edges
            report["candidates"] = self.result.candidates
        return validate_pipeline(
            report,
            df=self._df,
            claims_text=claims_text,
            y=y,
            d=d,
            z=z,
        )

    def ground(self, *, use_web: bool = False, timeout: float = 3.0) -> Any:
        from autocausal.grounding import ground_edges

        edges = self.result.edges if self.result is not None else []
        self.grounding = ground_edges(edges, use_web=use_web, timeout=timeout)
        if self.result is not None:
            self.result.grounding = self.grounding.to_dict()
        return self.grounding

    def physics_loop(
        self,
        *,
        horizon: int = 5,
        text: Optional[str] = None,
        domain: Union[str, list[str]] = "auto",
        system: str = "damped_oscillator",
        use_slm: Optional[bool] = None,
        second_pass: bool = True,
        use_web_ground: bool = False,
        impute_method: ImputeMethod = "auto",
        **discover_kwargs: Any,
    ) -> Any:
        """Run autocausal physics suite: mine → discover → rollout → physical ground → guide."""
        from autocausal.physics import PhysicsCausalSuite

        use_slm = self._assert_slm_allowed(use_slm)
        suite = PhysicsCausalSuite.from_autocausal(
            self,
            system=system,  # type: ignore[arg-type]
            prefer_nfs=True,
        )
        result = suite.loop(
            horizon=horizon,
            text=text,
            domain=domain,
            use_slm=use_slm,
            second_pass=second_pass,
            use_web_ground=use_web_ground,
            impute_method=impute_method,
            **discover_kwargs,
        )
        self.physics_result = result
        # keep shared state in sync (suite mutates the same AutoCausal)
        return result

    def ml_loop(
        self,
        *,
        text: str = "",
        use_slm: Optional[bool] = None,
        use_torch: Optional[bool] = None,
        guides: Optional[list[str]] = None,
        horizon: int = 5,
        physics: bool = True,
        **kwargs: Any,
    ) -> Any:
        """KPI-mined loop: mine → SLM ModelConstructPlan → impute → discover → FitReport."""
        from autocausal.ml import KPIMinedCausalLoop

        use_slm = self._assert_slm_allowed(use_slm)
        loop = KPIMinedCausalLoop.from_autocausal(self)
        return loop.run(
            text=text,
            use_slm=use_slm,
            use_torch=use_torch,
            guides=guides,
            horizon=horizon,
            physics=physics,
            **kwargs,
        )

    @classmethod
    def from_text_hints(
        cls,
        text: str,
        *,
        apply_guide: bool = False,
        use_slm: Optional[bool] = None,
    ) -> "AutoCausal":
        """Build an empty-frame AutoCausal seeded with NLP causal hints.

        Prefer the library API directly for apps::

            from autocausal.nlp import extract_causal_hints_from_text
            hints = extract_causal_hints_from_text(text)

        This facade stores hints on ``.nlp_hints`` and optionally runs ``guide``.
        """
        from autocausal.nlp import TextCausalHints

        hints = TextCausalHints.extract(text)
        # Minimal placeholder frame so mine/discover are not required
        df = pd.DataFrame({"_nlp_seed": [0]})
        ac = cls(df, source="text_hints")
        ac.nlp_hints = hints
        if apply_guide:
            # Guide with text only — mining on seed frame is uninformative
            from autocausal.slm import guide_pipeline

            resolved = ac._assert_slm_allowed(use_slm)
            ac.guide_result = guide_pipeline(
                hints.to_guide_context(), use_slm=resolved
            )
        return ac

    def apply_text_features(
        self,
        text_col: str,
        *,
        prefix: str = "nlp_",
        drop_text: bool = False,
    ) -> "AutoCausal":
        """Append NLP feature columns from ``text_col`` onto the current frame."""
        from autocausal.nlp import NlpFeatureBuilder

        builder = NlpFeatureBuilder(prefix=prefix)
        self._df = builder.transform_frame(self._df, text_col, drop_text=drop_text)
        self.roles = infer_column_roles(self._df)
        self.source = f"{self.source}+nlp_features:{text_col}"
        return self

    @classmethod
    def mine_behavioral_traces(
        cls,
        source: Union[str, Path] = "habit_loop",
        *,
        discover: bool = False,
        min_score: float = 0.15,
        **discover_kwargs: Any,
    ) -> Any:
        """Mine bundled/file behavioral traces → panel → report.

        Library equivalent::

            from autocausal.behavioral import mine_behavioral_traces
            result = mine_behavioral_traces("habit_loop", discover=True)
        """
        from autocausal.behavioral import mine_behavioral_traces as _mine

        return _mine(
            source,
            discover=discover,
            min_score=min_score,
            **discover_kwargs,
        )

    def attach_behavioral(
        self,
        source: Union[str, Path] = "habit_loop",
        *,
        on: str = "subject_id",
        how: str = "left",
        discover: bool = False,
    ) -> "AutoCausal":
        """Join a behavioral demo/file panel into the current frame and optionally discover."""
        from autocausal.behavioral import BehavioralTraceStore, join_traces_to_frame

        store = (
            BehavioralTraceStore.from_demo(str(source))
            if str(source) in ("habit_loop", "nudge_ab", "reinforcement_schedule")
            else BehavioralTraceStore.from_csv(source)
        )
        joined, log = join_traces_to_frame(self._df, store.collection, on=on, how=how)
        self._df = joined
        self.join_log.extend(log)
        self.roles = infer_column_roles(self._df)
        self.behavioral_result = store.mine(discover=discover) if discover else None
        self.source = f"{self.source}+behavioral:{store.name}"
        return self

    def insight_loop(
        self,
        *,
        text: str = "",
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        join: Optional[Union[str, list[str]]] = None,
        join_on: Optional[Union[str, list[str]]] = None,
        guide_backends: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run insight suite over this instance → ``InsightReport``.

        Prefer the library API in apps::

            from autocausal.insight import InsightSuite, run_insight_loop
            report = InsightSuite.from_autocausal(ac).run()
        """
        from autocausal.insight import InsightSuite

        resolved_slm = self._assert_slm_allowed(use_slm)
        if "max_rounds" in kwargs:
            kwargs["max_rounds"] = self._limit_rounds(int(kwargs["max_rounds"]))
        suite = InsightSuite.from_autocausal(
            self,
            use_slm=resolved_slm,
            model_name=model_name,
            guide_backends=guide_backends,
        )
        return suite.run(
            text=text,
            use_slm=resolved_slm,
            join=join,
            join_on=join_on,
            **kwargs,
        )

    def agentic_loop(
        self,
        *,
        text: str = "",
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        max_rounds: int = 3,
        persist_dir: Optional[Union[str, Path]] = None,
        vector_backend: str = "auto",
        prefer_langgraph: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Run SLM-guided agentic causal loop → ``AgenticLoopReport``.

        Cyclic FSM: hypothesize → skill/tool → validate → compact → persist → route.
        Soft optional LangGraph / vector backends. Prefer library import::

            from autocausal.agentic import AgenticCausalLoop, run_agentic_loop
        """
        from autocausal.agentic import AgenticCausalLoop

        resolved_slm = self._assert_slm_allowed(use_slm)
        max_rounds = self._limit_rounds(max_rounds)
        loop = AgenticCausalLoop(
            use_slm=resolved_slm,
            model_name=model_name,
            max_rounds=max_rounds,
            persist_dir=persist_dir,
            vector_backend=vector_backend,
            prefer_langgraph=prefer_langgraph,
        )
        return loop.run(
            ac=self,
            text=text,
            max_rounds=max_rounds,
            use_slm=resolved_slm,
            **kwargs,
        )

    def slm_loop(
        self,
        *,
        text: str = "",
        use_slm: bool = True,
        model_name: Optional[str] = None,
        max_rounds: int = 2,
        persist_dir: Optional[Union[str, Path]] = None,
        prefer_langgraph: bool = True,
        ensure_qwen: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run LangGraph/FSM SLM chain → ``SLMChainReport`` (insight + agentic).

        Prefer::

            from autocausal.agentic import run_slm_langgraph_loop
        """
        from autocausal.agentic.langgraph_chain import run_slm_langgraph_loop

        use_slm = self._assert_slm_allowed(use_slm)
        max_rounds = self._limit_rounds(max_rounds)
        return run_slm_langgraph_loop(
            ac=self,
            text=text,
            max_rounds=max_rounds,
            use_slm=use_slm,
            model_name=model_name,
            persist_dir=persist_dir,
            prefer_langgraph=prefer_langgraph,
            ensure_qwen=ensure_qwen,
            **kwargs,
        )

    def report(self, *, as_markdown: bool = True) -> str:
        """Render the last ``DiscoveryResult`` as markdown or JSON.

        After ``discover()``, prefer either::

            print(ac.report())
            print(ac.result.report())   # same markdown via DiscoveryResult.report
            print(result.to_markdown()) # explicit
        """
        if self.result is None:
            self.discover()
        assert self.result is not None
        return self.result.report(as_markdown=as_markdown)

    def generate_report(
        self,
        path: Union[str, Path],
        *,
        format: str = "pdf",
        use_slm: bool = True,
        policy: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a policy-constrained report artifact from this session."""
        from autocausal.reporting import ReportEngine, ReportPolicy

        report_policy = policy or ReportPolicy.production()
        return ReportEngine(
            use_slm=use_slm,
            policy=report_policy,
        ).generate(
            source=self,
            output=path,
            format=format,
            **kwargs,
        )

    def tabular_ml(
        self,
        *,
        target: str,
        features: Optional[Sequence[str]] = None,
        task: Optional[str] = None,
        group_column: Optional[str] = None,
        time_column: Optional[str] = None,
        calibrate: bool = False,
        enforce_gates: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Run leakage-safe AutoTabularML on the working frame.

        Predictive metrics are not causal effects. Production mode prefers
        explicit targets and fail-closed gates.
        """
        from autocausal.automl import AutoTabularML

        production = self.mode == "production"
        suite = AutoTabularML(
            self._df,
            policy=self.policy,
            mode=self.mode,
            random_state=self.random_state,
        )
        report = suite.run(
            target=target,
            task=task,
            feature_columns=list(features) if features is not None else None,
            group_column=group_column,
            time_column=time_column or (
                self.panel_spec.time if self.panel_spec is not None else None
            ),
            calibrate=calibrate,
            enforce_gates=(
                production if enforce_gates is None else bool(enforce_gates)
            ),
            **kwargs,
        )
        self.run_manifest.config.setdefault("tabular_ml", []).append(
            {
                "target": target,
                "selected": getattr(report, "selected_name", None),
                "mode": self.mode,
            }
        )
        return report

    def autoviz(
        self,
        *,
        use_slm: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Plan analysis-aware visualizations from the current session."""
        from autocausal.autoviz import AutoVizSuite

        use_slm = self._assert_slm_allowed(use_slm)
        suite = AutoVizSuite(
            self,
            mode=self.mode,
            roles=self.roles,
            panel=self.panel_spec,
            candidates=self.result.candidates if self.result is not None else None,
            edges=self.result.edges if self.result is not None else None,
            gate_results=getattr(self.run_manifest, "gates", None),
            **kwargs,
        )
        return suite.run(use_slm=use_slm)

    def run(
        self,
        *,
        impute_method: ImputeMethod = "auto",
        **discover_kwargs: Any,
    ) -> DiscoveryResult:
        self.impute(method=impute_method)
        return self.discover(**discover_kwargs)

    @classmethod
    def auto(
        cls,
        path_or_url: str,
        *,
        table: Optional[str] = None,
        query: Optional[str] = None,
        text: Optional[str] = None,
        use_slm: Optional[bool] = None,
        guide_backends: Optional[list[str]] = None,
        join: Optional[Union[str, list[str]]] = None,
        join_on: Optional[Union[str, list[str]]] = None,
        use_web_ground: bool = False,
        impute_method: ImputeMethod = "auto",
        second_pass: bool = True,
        physics: bool = False,
        physics_horizon: int = 5,
        physics_system: str = "damped_oscillator",
        physics_domain: Union[str, list[str]] = "auto",
        cleanse: bool = True,
        eda: bool = False,
        mode: AnalysisMode = "exploratory",
        strict: bool = False,
        policy: Optional[ProductionPolicy | dict[str, Any]] = None,
        random_state: Optional[int] = None,
        **discover_kwargs: Any,
    ) -> AutoResult:
        """Orchestrated flow: load → [cleanse] → [eda] → join? → mine → impute → discover → guide → ground.

        ``use_slm`` defaults to enabled in every mode (SLM guides behavior;
        soft-falls to deterministic rules when a model is unavailable). Pass
        ``use_slm=False`` to force rule-only guidance.

        ``mode="production"`` / ``strict=True`` propagates to discover (no synthetic
        IV, QC block, prefer ensemble+stability).
        """
        from autocausal.db import ping
        from autocausal.ingest import dialect_from_url
        from autocausal.suites.director import resolve_suite_slm

        notes: list[str] = []
        ping_info = None
        path = str(path_or_url)
        lower = path.lower()
        resolved_mode = resolve_mode(mode, strict=strict if strict else None)
        effective_policy = resolve_policy(
            resolved_mode,
            policy,
            random_state=random_state,
        )
        if use_slm is False:
            requested_slm = False
        elif use_slm is True:
            requested_slm = True
        else:
            requested_slm = True  # SLM guides by default in all modes
        if requested_slm and not effective_policy.allow_slm:
            raise UnsafePayloadError(
                "AutoCausal.auto(use_slm=True) is forbidden by production policy.",
                code="slm_forbidden",
                recommendations=[
                    "Pass use_slm=False or set policy.allow_slm=True (default)."
                ],
            )
        use_slm = resolve_suite_slm(requested_slm)
        discover_kwargs.setdefault("mode", resolved_mode)

        if lower.endswith(".csv"):
            ac = cls.from_csv(
                path,
                mode=resolved_mode,
                policy=effective_policy,
                random_state=effective_policy.random_state,
            )
        elif lower.endswith(".parquet"):
            ac = cls.from_parquet(
                path,
                mode=resolved_mode,
                policy=effective_policy,
                random_state=effective_policy.random_state,
            )
        elif "://" in path or path.startswith("sqlite:"):
            ping_info = ping(path, timeout=5.0)
            notes.append(f"ping ok={ping_info.ok} latency_ms={ping_info.latency_ms}")
            if not table and not query:
                # try bundled demo table name if sqlite demo
                raise ValueError("Database URL requires table= or query=")
            ac = cls.from_sqlalchemy(
                path,
                table=table,
                query=query,
                mode=resolved_mode,
                policy=effective_policy,
                random_state=effective_policy.random_state,
            )
            notes.append(f"dialect={dialect_from_url(path)}")
        else:
            # treat as CSV path fallback
            ac = cls.from_csv(
                path,
                mode=resolved_mode,
                policy=effective_policy,
                random_state=effective_policy.random_state,
            )

        ac._suite_use_slm = use_slm
        notes.append(f"use_slm={use_slm} (soft rule fallback)")
        notes.append(f"mode={ac.mode}")

        if cleanse:
            ac.cleanse(use_slm=use_slm, text=text or "", impute=impute_method)
            notes.append("cleanse=True (SLM-directed AutoCleanseSuite)")
        if eda:
            ac.eda(use_slm=use_slm, text=text or "")
            notes.append("eda=True (SLM-directed AutoEDASuite)")

        if join:
            ac.join_public(join, on=join_on, allow_network=False)
            notes.append(f"joined public: {join}")

        physics_payload: Optional[dict[str, Any]] = None
        if physics:
            # Dedicated physics loop owns mine/impute/discover/rollout/ground/guide
            phys = ac.physics_loop(
                horizon=physics_horizon,
                text=text,
                domain=physics_domain,
                system=physics_system,
                use_slm=use_slm,
                second_pass=second_pass,
                use_web_ground=use_web_ground,
                impute_method=impute_method,
                **discover_kwargs,
            )
            physics_payload = phys.to_dict()
            notes.extend(list(phys.notes or []))
            notes.append(f"physics=True horizon={physics_horizon} system={physics_system}")
            mining = ac.mining
            result = ac.result
            assert result is not None
            guide = ac.guide_result
            grounding = ac.grounding
            direction = (
                ac.direction_plan.to_dict()
                if ac.direction_plan is not None and hasattr(ac.direction_plan, "to_dict")
                else None
            )
            return AutoResult(
                discovery=result,
                mining=mining.to_dict() if mining is not None and hasattr(mining, "to_dict") else mining,
                guide=guide.to_dict() if guide is not None else None,
                direction_plan=direction,
                grounding=grounding.to_dict() if grounding is not None else None,
                physics=physics_payload,
                join_log=list(ac.join_log),
                ping=ping_info.to_dict() if ping_info is not None else None,
                source=ac.source,
                notes=notes,
            )

        ac.automine(use_slm=use_slm, text=text or "")
        mining = ac.mining
        if ac.imputation is None:
            ac.impute(method=impute_method)
        result = ac.discover(**discover_kwargs)

        if guide_backends:
            plan = ac.direct(
                text=text,
                backends=guide_backends,
                use_slm=use_slm,
                second_pass=second_pass,
                **discover_kwargs,
            )
            guide = ac.guide_result
            notes.append(f"guide_backends={guide_backends}")
            if plan.focus_columns:
                notes.append(f"direction focus: {plan.focus_columns[:12]}")
            result = ac.result or result
            direction = plan.to_dict()
        else:
            guide = ac.guide(text=text, use_slm=use_slm)
            direction = None
            if second_pass and guide.focus_columns:
                focus = [c for c in guide.focus_columns if c in ac.df.columns]
                if len(focus) >= 2:
                    notes.append(f"second-pass focus: {focus[:12]}")
                    result = ac.discover(focus_columns=focus, **discover_kwargs)
                    guide = ac.guide(text=text, use_slm=use_slm)

        grounding = ac.ground(use_web=use_web_ground)
        sens = ac.sensitivity(text=text or "")
        notes.append(f"sensitivity domain={sens.domain_hint}")
        auto_result = AutoResult(
            discovery=result,
            mining=mining.to_dict() if mining is not None and hasattr(mining, "to_dict") else mining,
            guide=guide.to_dict() if guide is not None else None,
            direction_plan=direction,
            grounding=grounding.to_dict() if grounding is not None else None,
            physics=physics_payload,
            join_log=list(ac.join_log),
            ping=ping_info.to_dict() if ping_info is not None else None,
            source=ac.source,
            notes=notes,
            sensitivity_report=sens.to_dict(),
            qc=ac.qc_report.to_dict() if ac.qc_report is not None else None,
            nlp_hints=ac.nlp_hints.to_dict() if ac.nlp_hints is not None and hasattr(ac.nlp_hints, "to_dict") else None,
        )
        return auto_result
