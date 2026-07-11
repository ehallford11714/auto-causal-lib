"""Public AutoCausal API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import pandas as pd

from autocausal.discovery import discover_ensemble, discover_relationships
from autocausal.impute import ImputationReport, impute_dataframe
from autocausal.ingest import load_csv, load_parquet, load_sqlalchemy
from autocausal.results import AutoResult, DiscoveryResult
from autocausal.roles import ColumnRole, infer_column_roles


ImputeMethod = Literal["auto", "median_mode", "knn"]
QCMode = Literal["off", "warn", "block"]

__all__ = ["AutoCausal", "DiscoveryResult", "AutoResult"]


class AutoCausal:
    """Load tabular data, impute missing fields, discover exploratory causal edges."""

    def __init__(self, df: pd.DataFrame, *, source: str = "memory") -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self._raw = df.copy()
        self._df = df.copy()
        self.source = source
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
        # First-class suite reports (AutoCleanse / AutoEDA / AutoMine)
        self.cleanse_report: Any = None
        self.eda_report: Any = None
        self.mine_report: Any = None
        self._suite_use_slm: Optional[bool] = None
        self.grail_report: Any = None


    @classmethod
    def from_csv(cls, path: str | Path, **read_csv_kwargs: Any) -> "AutoCausal":
        df = load_csv(path, **read_csv_kwargs)
        return cls(df, source=f"csv:{path}")

    @classmethod
    def from_parquet(cls, path: str | Path, **kwargs: Any) -> "AutoCausal":
        df = load_parquet(path, **kwargs)
        return cls(df, source=f"parquet:{path}")

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
        **engine_kwargs: Any,
    ) -> "AutoCausal":
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
        label = table or (query[:40] + "…" if query and len(query) > 40 else query) or "sql"
        return cls(df, source=f"sql:{label}")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, source: str = "dataframe") -> "AutoCausal":
        return cls(df, source=source)

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
            raise ValueError(f"QC blocked discovery: {codes}. See ac.qc_report.")
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
        from autocausal.suites.director import resolve_suite_slm

        slm = resolve_suite_slm(use_slm if use_slm is not None else self._suite_use_slm)
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
        from autocausal.suites.director import resolve_suite_slm

        slm = resolve_suite_slm(use_slm if use_slm is not None else self._suite_use_slm)
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
        from autocausal.suites.director import resolve_suite_slm

        slm = resolve_suite_slm(use_slm if use_slm is not None else self._suite_use_slm)
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


    def discover(
        self,
        *,
        alpha: float = 0.05,
        max_cond_size: int = 2,
        min_abs_corr: float = 0.15,
        use_iv: bool = True,
        focus_columns: Optional[list[str]] = None,
        stability: bool = False,
        bootstrap_n: int = 20,
        ensemble: bool = False,
        methods: Optional[list[str]] = None,
        method: Optional[str] = None,
        min_methods: int = 2,
        qc: QCMode = "warn",
        drop_id_columns: bool = True,
        seed: int = 0,
        include_optional: bool = True,
    ) -> DiscoveryResult:
        if qc != "off":
            self.validate_qc(mode=qc)

        if self.imputation is None and self._df.isna().any().any():
            self.impute(method="auto")
        work = self._df
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
        if ensemble or methods:
            result = discover_ensemble(
                work,
                roles=self.roles,
                methods=methods,  # type: ignore[arg-type]
                alpha=alpha,
                max_cond_size=max_cond_size,
                min_abs_corr=min_abs_corr,
                use_iv=use_iv,
                stability=stability,
                bootstrap_n=bootstrap_n,
                min_methods=min_methods,
                seed=seed,
                include_optional=include_optional if methods is None else False,
            )
        else:
            result = discover_relationships(
                work,
                roles=self.roles,
                alpha=alpha,
                max_cond_size=max_cond_size,
                min_abs_corr=min_abs_corr,
                use_iv=use_iv,
                method=method or "score_pc_lite",  # type: ignore[arg-type]
                stability=stability,
                bootstrap_n=bootstrap_n,
                seed=seed,
            )
        result.imputation = self.imputation
        if self.mining is not None:
            result.mining = self.mining.to_dict() if hasattr(self.mining, "to_dict") else self.mining
        if self.qc_report is not None:
            result.notes = list(result.notes) + [
                f"QC ok={self.qc_report.ok} issues={len(self.qc_report.issues)}"
            ]
        if self.sensitivity_report is not None:
            result.sensitivity_report = (
                self.sensitivity_report.to_dict()
                if hasattr(self.sensitivity_report, "to_dict")
                else self.sensitivity_report
            )
        # Bind session + working frame so result.estimate / refute / fabric work standalone
        result.bind_session(self, frame=work.copy(), source=self.source)
        self.result = result
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
        use_slm: bool = False,
        model_name: Optional[str] = None,
        backends: Optional[list[str]] = None,
    ) -> Any:
        """Run guide backends; multi-backend when ``backends`` is set."""
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
        use_slm: bool = False,
        model_name: Optional[str] = None,
        second_pass: bool = True,
        **discover_kwargs: Any,
    ) -> Any:
        """
        Steer causal direction with one or more guide backends.

        Merges outputs into a ``DirectionPlan``, optionally re-runs discover
        focused on plan columns (second pass).
        """
        from autocausal.guides import direct as run_direct

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
            "source": self.source,
            "shape": [int(self._df.shape[0]), int(self._df.shape[1])],
            "n_rows": int(self._df.shape[0]),
            "n_cols": int(self._df.shape[1]),
            "columns": [str(c) for c in self._df.columns],
            "has_result": self.result is not None,
            "n_edges": n_edges,
            "methods": methods,
            "has_grail": self.grail_report is not None,
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
        seed: int = 0,
    ) -> Any:
        """Compute sensitivity metrics and attach to discovery / AutoResult."""
        from autocausal.sensitivity import compute_sensitivity

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
            seed=seed,
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
        cands = self.result.candidates if self.result is not None else {}
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
        **kwargs: Any,
    ) -> Any:
        """Estimate ATE/CATE via builtin / DoubleML / EconML (soft-optional).

        Primary session API. Equivalent chaining on the discover handle::

            result = ac.discover()
            result.estimate(backend="builtin_ols")
        """
        from autocausal.engines import estimate as eng_estimate

        candidates = self.result.candidates if self.result is not None else None
        result = eng_estimate(
            self._df,
            backend=backend,
            y=y,
            d=d,
            x=list(x) if x is not None else None,
            z=z,
            candidates=candidates,
            **kwargs,
        )
        if not hasattr(self, "estimate_results") or self.estimate_results is None:
            self.estimate_results = []
        self.estimate_results.append(result)
        if self.result is not None:
            self.result.estimate_results.append(result)
        return result

    def refute(
        self,
        edge: Optional[dict[str, Any]] = None,
        *,
        method: str = "placebo",
        **kwargs: Any,
    ) -> Any:
        """Soft refute hook (DoWhy real path / builtin placebo) — no-ops gracefully.

        Primary session API. Equivalent chaining on the discover handle::

            result = ac.discover()
            result.refute(method="placebo")
        """
        from autocausal.suite_tools import refute as suite_refute

        if edge is None and self.result is not None and self.result.edges:
            edge = self.result.edges[0]
        candidates = None
        if self.result is not None:
            candidates = self.result.candidates
        result = suite_refute(
            edge or {},
            method=method,
            df=self._df,
            candidates=candidates,
            **kwargs,
        )
        self.refute_results.append(result)
        if self.result is not None:
            self.result.refute_results.append(result)
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
            },
        )
    def create(
        self,
        *,
        text: Optional[str] = None,
        use_slm: bool = False,
        model_name: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> Any:
        """SLM/rule *creation*: propose questions, instruments, morphemes."""
        from autocausal.slm import create_from_context

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
        return self.creation_result

    def interpret(
        self,
        *,
        text: Optional[str] = None,
        use_slm: bool = False,
        model_name: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> Any:
        """SLM/rule *inference* narrative + caveats over discovery/IV results."""
        from autocausal.slm import infer_from_results

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
        use_slm: bool = False,
        second_pass: bool = True,
        use_web_ground: bool = False,
        impute_method: ImputeMethod = "auto",
        **discover_kwargs: Any,
    ) -> Any:
        """Run autocausal physics suite: mine → discover → rollout → physical ground → guide."""
        from autocausal.physics import PhysicsCausalSuite

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
        use_slm: bool = False,
        use_torch: Optional[bool] = None,
        guides: Optional[list[str]] = None,
        horizon: int = 5,
        physics: bool = True,
        **kwargs: Any,
    ) -> Any:
        """KPI-mined loop: mine → SLM ModelConstructPlan → impute → discover → FitReport."""
        from autocausal.ml import KPIMinedCausalLoop

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
        use_slm: bool = False,
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

            ac.guide_result = guide_pipeline(
                hints.to_guide_context(), use_slm=use_slm
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
            report = InsightSuite.from_autocausal(ac).run(use_slm=False)
        """
        from autocausal.insight import InsightSuite

        suite = InsightSuite.from_autocausal(
            self,
            use_slm=bool(use_slm) if use_slm is not None else False,
            model_name=model_name,
            guide_backends=guide_backends,
        )
        return suite.run(
            text=text,
            use_slm=use_slm,
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

        loop = AgenticCausalLoop(
            use_slm=bool(use_slm) if use_slm is not None else False,
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
            use_slm=use_slm,
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
        use_slm: bool = True,
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
        **discover_kwargs: Any,
    ) -> AutoResult:
        """Orchestrated flow: load → [cleanse] → [eda] → join? → mine → impute → discover → guide → ground.

        ``use_slm`` defaults to ``True`` (soft-fail to rules). Auto* means
        SLM-directed when HuggingFace is available — never hard-crashes offline.
        """
        from autocausal.db import ping
        from autocausal.ingest import dialect_from_url
        from autocausal.suites.director import resolve_suite_slm

        notes: list[str] = []
        ping_info = None
        path = str(path_or_url)
        lower = path.lower()
        use_slm = resolve_suite_slm(use_slm)

        if lower.endswith(".csv"):
            ac = cls.from_csv(path)
        elif lower.endswith(".parquet"):
            ac = cls.from_parquet(path)
        elif "://" in path or path.startswith("sqlite:"):
            ping_info = ping(path, timeout=5.0)
            notes.append(f"ping ok={ping_info.ok} latency_ms={ping_info.latency_ms}")
            if not table and not query:
                # try bundled demo table name if sqlite demo
                raise ValueError("Database URL requires table= or query=")
            ac = cls.from_sqlalchemy(path, table=table, query=query)
            notes.append(f"dialect={dialect_from_url(path)}")
        else:
            # treat as CSV path fallback
            ac = cls.from_csv(path)

        ac._suite_use_slm = use_slm
        notes.append(f"use_slm={use_slm} (soft rule fallback)")

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
