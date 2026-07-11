"""Result dataclasses (kept separate to avoid circular imports)."""

from __future__ import annotations

import json
import weakref
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from autocausal.impute import ImputationReport
from autocausal.roles import ColumnRole


@dataclass
class DiscoveryResult:
    """Structured output from causal relationship discovery.

    After ``ac.discover()``, this object is a usable post-discover handle::

        result = ac.discover()
        result.estimate(backend="builtin_ols")
        result.refute(method="placebo")
        result.to_fabric_bundle()

    ``AutoCausal.estimate`` / ``refute`` remain the primary session API; these
    methods prefer a live parent session (weakref) and otherwise run standalone
    against an attached ``frame``.
    """

    edges: list[dict[str, Any]]
    graph: dict[str, Any]
    roles: dict[str, ColumnRole]
    candidates: dict[str, list[str]]
    imputation: Optional[ImputationReport] = None
    method: str = "score_pc_lite"
    notes: list[str] = field(default_factory=list)
    mining: Optional[dict[str, Any]] = None
    guide: Optional[dict[str, Any]] = None
    grounding: Optional[dict[str, Any]] = None
    stability_enabled: bool = False
    bootstrap_n: int = 0
    ensemble_methods: list[str] = field(default_factory=list)
    method_edges: Optional[dict[str, list[dict[str, Any]]]] = None
    sensitivity_report: Optional[dict[str, Any]] = None
    mode: str = "exploratory"
    run_id: str = ""
    policy: dict[str, Any] = field(default_factory=dict)
    evidence_gates: list[dict[str, Any]] = field(default_factory=list)
    rejected_edges: list[dict[str, Any]] = field(default_factory=list)
    manifest: Optional[Any] = field(default=None, repr=False, compare=False)
    # Attached at discover-time for standalone estimate/refute/fabric exports
    frame: Optional[Any] = field(default=None, repr=False, compare=False)
    source: str = ""
    estimate_results: list[Any] = field(default_factory=list, repr=False, compare=False)
    refute_results: list[Any] = field(default_factory=list, repr=False, compare=False)
    _owner_ref: Any = field(default=None, init=False, repr=False, compare=False)

    def bind_session(self, owner: Any, *, frame: Any = None, source: str = "") -> "DiscoveryResult":
        """Attach a weakref to the parent ``AutoCausal`` (+ optional working frame)."""
        try:
            self._owner_ref = weakref.ref(owner)
        except TypeError:
            self._owner_ref = None
        if frame is not None:
            self.frame = frame
        if source:
            self.source = source
        elif owner is not None and getattr(owner, "source", None):
            self.source = str(owner.source)
        return self

    def session(self) -> Any:
        """Return the live parent ``AutoCausal`` if still reachable, else ``None``."""
        if self._owner_ref is None:
            return None
        try:
            return self._owner_ref()
        except Exception:
            return None

    def dataframe(self) -> Any:
        """Working frame: prefer live session ``_df``, else attached ``frame``."""
        owner = self.session()
        if owner is not None and getattr(owner, "_df", None) is not None:
            return owner._df
        return self.frame

    def _frame_shape(self) -> tuple[int, int]:
        df = self.dataframe()
        if df is None:
            return (0, 0)
        try:
            return (int(len(df)), int(len(df.columns)))
        except Exception:
            return (0, 0)

    def to_dict(self) -> dict[str, Any]:
        roles = {k: (v.value if hasattr(v, "value") else str(v)) for k, v in self.roles.items()}
        out: dict[str, Any] = {
            "method": self.method,
            "mode": self.mode,
            "run_id": self.run_id,
            "edges": self.edges,
            "rejected_edges": self.rejected_edges,
            "evidence_gates": self.evidence_gates,
            "graph": self.graph,
            "roles": roles,
            "candidates": self.candidates,
            "notes": self.notes,
            "stability_enabled": self.stability_enabled,
            "bootstrap_n": self.bootstrap_n,
            "policy": dict(self.policy),
        }
        if self.manifest is not None:
            out["manifest"] = (
                self.manifest.to_dict()
                if hasattr(self.manifest, "to_dict")
                else self.manifest
            )
        if self.source:
            out["source"] = self.source
        if self.ensemble_methods:
            out["ensemble_methods"] = list(self.ensemble_methods)
        if self.method_edges is not None:
            out["method_edges"] = self.method_edges
        if self.imputation is not None:
            imputation = (
                self.imputation.to_dict()
                if hasattr(self.imputation, "to_dict")
                else asdict(self.imputation)
            )
            redact_values = (
                self.mode == "production"
                or bool(self.policy.get("redact_sample_values"))
            )
            if redact_values:
                for column in imputation.get("columns") or []:
                    if isinstance(column, dict) and "fill_value" in column:
                        column["fill_value"] = "<redacted>"
            out["imputation"] = imputation
        if self.mining is not None:
            out["mining"] = self.mining
        if self.guide is not None:
            out["guide"] = self.guide
        if self.grounding is not None:
            out["grounding"] = self.grounding
        if self.sensitivity_report is not None:
            out["sensitivity"] = self.sensitivity_report
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DiscoveryResult":
        """Restore the portable result surface (without raw frame/session refs)."""
        from autocausal.production import RunManifest

        payload = dict(value)
        role_values: dict[str, ColumnRole] = {}
        for column, role in (payload.get("roles") or {}).items():
            try:
                role_values[str(column)] = ColumnRole(str(role))
            except ValueError:
                role_values[str(column)] = ColumnRole.UNKNOWN
        manifest_raw = payload.get("manifest")
        manifest = (
            RunManifest.from_dict(manifest_raw)
            if isinstance(manifest_raw, dict)
            else manifest_raw
        )
        return cls(
            edges=list(payload.get("edges") or []),
            graph=dict(payload.get("graph") or {}),
            roles=role_values,
            candidates={
                str(key): [str(item) for item in items]
                for key, items in (payload.get("candidates") or {}).items()
            },
            method=str(payload.get("method") or "score_pc_lite"),
            notes=list(payload.get("notes") or []),
            mining=payload.get("mining"),
            guide=payload.get("guide"),
            grounding=payload.get("grounding"),
            stability_enabled=bool(payload.get("stability_enabled")),
            bootstrap_n=int(payload.get("bootstrap_n") or 0),
            ensemble_methods=list(payload.get("ensemble_methods") or []),
            method_edges=payload.get("method_edges"),
            sensitivity_report=payload.get("sensitivity"),
            mode=str(payload.get("mode") or "exploratory"),
            run_id=str(payload.get("run_id") or ""),
            policy=dict(payload.get("policy") or {}),
            evidence_gates=list(payload.get("evidence_gates") or []),
            rejected_edges=list(payload.get("rejected_edges") or []),
            source=str(payload.get("source") or ""),
            manifest=manifest,
        )

    @classmethod
    def from_json(cls, value: str) -> "DiscoveryResult":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise TypeError("DiscoveryResult JSON must contain an object")
        return cls.from_dict(payload)

    def replay_config(self) -> dict[str, Any]:
        """Export replayable configuration without embedding raw data."""
        if self.manifest is not None and hasattr(self.manifest, "replay_config"):
            return self.manifest.replay_config()
        return {
            "schema": "AutoCausalReplayConfig.v1",
            "mode": self.mode,
            "policy": dict(self.policy),
            "discover": {
                "method": self.method,
                "stability": self.stability_enabled,
                "bootstrap_n": self.bootstrap_n,
                "ensemble": bool(self.ensemble_methods),
                "methods": list(self.ensemble_methods),
            },
        }

    def reproduce(
        self,
        frame: Any = None,
        *,
        overrides: Optional[dict[str, Any]] = None,
        verify_fingerprint: bool = True,
    ) -> "DiscoveryResult":
        """Replay discovery deterministically against an attached/supplied frame."""
        from autocausal.api import AutoCausal
        from autocausal.production import (
            ProductionGateError,
            ProductionPolicy,
            build_data_fingerprint,
        )

        data = frame if frame is not None else self.dataframe()
        if data is None:
            raise RuntimeError("reproduce() requires an attached or supplied DataFrame")
        config = self.replay_config()
        expected = config.get("expected_data_fingerprint") or {}
        actual = build_data_fingerprint(data)
        if (
            verify_fingerprint
            and expected.get("sha256")
            and actual.get("sha256") != expected.get("sha256")
        ):
            raise ProductionGateError(
                "Replay data fingerprint does not match the original run.",
                code="replay_fingerprint_mismatch",
                recommendations=[
                    "Supply the original frame or set verify_fingerprint=False "
                    "only after deliberate review."
                ],
            )
        policy_raw = config.get("policy") or self.policy
        policy = ProductionPolicy.from_dict(policy_raw)
        ac = AutoCausal.from_dataframe(
            data,
            source=self.source or "replay",
            mode=str(config.get("mode") or self.mode),  # type: ignore[arg-type]
            policy=policy,
            random_state=int(config.get("random_state") or policy.random_state),
        )
        discover = dict(config.get("discover") or {})
        for key in ("methods", "candidates", "focus_columns"):
            if not discover.get(key):
                discover.pop(key, None)
        if discover.get("method") is None:
            discover.pop("method", None)
        discover.update(overrides or {})
        return ac.discover(**discover)

    def to_markdown(self) -> str:
        from autocausal.report import render_markdown_report

        return render_markdown_report(self)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``.

        Prefer ``result.report()`` or ``ac.report()`` after ``discover()``.
        """
        if as_markdown:
            return self.to_markdown()
        return self.to_json()

    def generate_report(
        self,
        path: str | Path,
        *,
        format: str = "pdf",
        use_slm: bool = True,
        policy: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a policy-constrained report artifact from this result."""
        from autocausal.reporting import ReportEngine, ReportPolicy

        return ReportEngine(
            use_slm=use_slm,
            policy=policy or ReportPolicy.production(),
        ).generate(source=self, output=path, format=format, **kwargs)

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
        """Estimate ATE/CATE from this discovery (delegates to session or engines).

        Primary path remains ``ac.estimate(...)``; this lets callers chain::

            result = ac.discover()
            est = result.estimate(backend="builtin_ols")
        """
        owner = self.session()
        if owner is not None and hasattr(owner, "estimate"):
            # Keep session.result aligned so candidates/edges stay consistent
            if getattr(owner, "result", None) is not self:
                owner.result = self
            # Session estimate appends to self.estimate_results
            return owner.estimate(backend=backend, y=y, d=d, x=x, z=z, **kwargs)

        df = self.dataframe()
        if df is None:
            raise RuntimeError(
                "DiscoveryResult.estimate requires an attached frame or a live "
                "AutoCausal session. Call ac.discover() (which binds both), or "
                "pass frame=... via bind_session()."
            )
        from autocausal.engines import estimate as eng_estimate

        result = eng_estimate(
            df,
            backend=backend,
            y=y,
            d=d,
            x=list(x) if x is not None else None,
            z=z,
            candidates=self.candidates,
            **kwargs,
        )
        self.estimate_results.append(result)
        return result

    def refute(
        self,
        edge: Optional[dict[str, Any]] = None,
        *,
        method: str = "placebo",
        **kwargs: Any,
    ) -> Any:
        """Soft refute hook from this discovery (delegates to session or suite_tools).

        Primary path remains ``ac.refute(...)``; this lets callers chain::

            result = ac.discover()
            ref = result.refute(method="placebo")
        """
        owner = self.session()
        if owner is not None and hasattr(owner, "refute"):
            if getattr(owner, "result", None) is not self:
                owner.result = self
            return owner.refute(edge=edge, method=method, **kwargs)

        df = self.dataframe()
        if df is None:
            raise RuntimeError(
                "DiscoveryResult.refute requires an attached frame or a live "
                "AutoCausal session. Call ac.discover() (which binds both), or "
                "pass frame=... via bind_session()."
            )
        from autocausal.suite_tools import refute as suite_refute

        if edge is None and self.edges:
            edge = self.edges[0]
        result = suite_refute(
            edge or {},
            method=method,
            df=df,
            candidates=self.candidates,
            **kwargs,
        )
        self.refute_results.append(result)
        return result

    def run_sensitivity(
        self,
        *,
        text: str = "",
        domain: Optional[str] = None,
        n_boot: int = 8,
        seed: int = 0,
    ) -> Any:
        """Compute sensitivity metrics and attach to ``self.sensitivity_report``.

        Prefer ``ac.sensitivity(...)`` when the session is still available.
        """
        owner = self.session()
        if owner is not None and hasattr(owner, "sensitivity"):
            if getattr(owner, "result", None) is not self:
                owner.result = self
            return owner.sensitivity(text=text, domain=domain, n_boot=n_boot, seed=seed)

        df = self.dataframe()
        if df is None:
            raise RuntimeError(
                "DiscoveryResult.run_sensitivity requires an attached frame or "
                "a live AutoCausal session."
            )
        from autocausal.sensitivity import compute_sensitivity

        report = compute_sensitivity(
            df,
            edges=self.edges,
            trajectory=None,
            text=text,
            domain=domain,
            n_boot=n_boot,
            seed=seed,
        )
        self.sensitivity_report = report.to_dict() if hasattr(report, "to_dict") else report
        if hasattr(report, "to_mine_notes"):
            self.notes = list(self.notes) + list(report.to_mine_notes())
        return report

    def sensitivity(
        self,
        *,
        text: str = "",
        domain: Optional[str] = None,
        n_boot: int = 8,
        seed: int = 0,
    ) -> Any:
        """Compute sensitivity — matches ``AutoCausal.sensitivity`` naming.

        Stored payload is available as ``result.sensitivity_report``.
        """
        return self.run_sensitivity(text=text, domain=domain, n_boot=n_boot, seed=seed)

    def to_causaliv_request(
        self,
        *,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        instrument: Optional[str] = None,
        confounders: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        """Structured CausalIV handoff spec (soft if ``causaliv`` missing).

        Prefer ``ac.to_causaliv_request()`` when the session is live; this
        builds the same envelope from candidates + attached frame.
        """
        owner = self.session()
        if owner is not None and hasattr(owner, "to_causaliv_request"):
            if getattr(owner, "result", None) is not self:
                owner.result = self
            return owner.to_causaliv_request(
                treatment=treatment,
                outcome=outcome,
                instrument=instrument,
                confounders=confounders,
            )

        cands = self.candidates or {}
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

        df = self.dataframe()
        n_rows = int(len(df)) if df is not None else 0
        columns = [str(c) for c in df.columns] if df is not None else []
        return {
            "schema": "CausalIVRequest.v1",
            "produced_by": "autocausal",
            "y": y,
            "d": d,
            "z": z,
            "w": w,
            "n_rows": n_rows,
            "columns": columns,
            "edges": list(self.edges[:20]),
            "causaliv_available": causaliv_available,
            "notes": iv_handoff_notes(
                treatment=d, outcome=y, instrument=z, confounders=w
            ),
            "soft": True,
        }

    def engines_status(self) -> dict[str, Any]:
        """Unified discovery/estimate/refute/package engine status (stateless)."""
        from autocausal.engines import engine_status

        return engine_status()

    def to_causal_edges(self) -> list[dict[str, Any]]:
        """Export edges as CausalEdge.v1 envelopes (shared Fabric contract)."""
        from autocausal.contracts import edges_to_causal_edge_envelopes

        return edges_to_causal_edge_envelopes(self.edges)

    def to_mine_report(
        self,
        *,
        n_rows: int = 0,
        n_cols: int = 0,
        backend: str = "autocausal.mine",
    ) -> dict[str, Any]:
        """Export attached mining (if any) as a MineReport.v1 envelope."""
        from autocausal.contracts import mining_to_mine_report

        if n_rows <= 0 or n_cols <= 0:
            fr, fc = self._frame_shape()
            if n_rows <= 0:
                n_rows = fr
            if n_cols <= 0:
                n_cols = fc
        return mining_to_mine_report(
            self.mining,
            n_rows=n_rows,
            n_cols=n_cols,
            backend=backend,
        )

    def to_search_dag(self, *, soft: bool = True) -> dict[str, Any]:
        """Soft-optional CausalSearch DAG export (SearchDAG.v1 envelope)."""
        from autocausal.contracts import discovery_to_search_dag

        return discovery_to_search_dag(self, soft=soft)

    def to_fabric_bundle(
        self,
        *,
        n_rows: int = 0,
        n_cols: int = 0,
        insight: Any = None,
        source: str = "",
        notes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Assemble FabricBundle.v1 from this discovery (+ attached mining).

        Prefer ``ac.to_fabric_bundle()`` when you still have the ``AutoCausal``
        session (includes frame shape / QC). This method lets callers export
        directly from a ``DiscoveryResult``::

            result = ac.discover()
            bundle = result.to_fabric_bundle()
        """
        from autocausal.contracts import fabric_bundle

        if n_rows <= 0 or n_cols <= 0:
            fr, fc = self._frame_shape()
            if n_rows <= 0:
                n_rows = fr
            if n_cols <= 0:
                n_cols = fc
        owner = self.session()
        extra: Optional[dict[str, Any]] = None
        if owner is not None:
            extra = {
                "qc": (
                    owner.qc_report.to_dict()
                    if getattr(owner, "qc_report", None) is not None
                    and hasattr(owner.qc_report, "to_dict")
                    else None
                ),
                "nlp_hints": (
                    owner.nlp_hints.to_dict()
                    if getattr(owner, "nlp_hints", None) is not None
                    and hasattr(owner.nlp_hints, "to_dict")
                    else None
                ),
            }
        return fabric_bundle(
            mining=self.mining,
            discovery=self,
            insight=insight,
            n_rows=n_rows,
            n_cols=n_cols,
            source=source or self.source,
            notes=list(notes or []) + list(self.notes or []),
            sensitivity=self.sensitivity_report,
            extra=extra,
        )


@dataclass
class AutoResult:
    """Full orchestrated auto() pipeline output."""

    discovery: DiscoveryResult
    mining: Optional[dict[str, Any]] = None
    guide: Optional[dict[str, Any]] = None
    direction_plan: Optional[dict[str, Any]] = None
    grounding: Optional[dict[str, Any]] = None
    physics: Optional[dict[str, Any]] = None
    join_log: list[dict[str, Any]] = field(default_factory=list)
    ping: Optional[dict[str, Any]] = None
    source: str = ""
    notes: list[str] = field(default_factory=list)
    sensitivity_report: Optional[dict[str, Any]] = None
    qc: Optional[dict[str, Any]] = None
    nlp_hints: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ping": self.ping,
            "join_log": self.join_log,
            "mining": self.mining,
            "discovery": self.discovery.to_dict(),
            "guide": self.guide,
            "direction_plan": self.direction_plan,
            "grounding": self.grounding,
            "physics": self.physics,
            "sensitivity": self.sensitivity_report,
            "qc": self.qc,
            "nlp_hints": self.nlp_hints,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        from autocausal.report import render_auto_markdown

        return render_auto_markdown(self)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()

    def generate_report(
        self,
        path: str | Path,
        *,
        format: str = "pdf",
        use_slm: bool = True,
        policy: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a policy-constrained report artifact from this auto result."""
        from autocausal.reporting import ReportEngine, ReportPolicy

        return ReportEngine(
            use_slm=use_slm,
            policy=policy or ReportPolicy.production(),
        ).generate(source=self, output=path, format=format, **kwargs)

    def to_causal_edges(self) -> list[dict[str, Any]]:
        """Export discovery edges as CausalEdge.v1 envelopes."""
        return self.discovery.to_causal_edges()

    def to_mine_report(
        self,
        *,
        n_rows: int = 0,
        n_cols: int = 0,
        backend: str = "autocausal.mine",
    ) -> dict[str, Any]:
        """Export mining as a MineReport.v1 envelope."""
        from autocausal.contracts import mining_to_mine_report

        return mining_to_mine_report(
            self.mining if self.mining is not None else self.discovery.mining,
            n_rows=n_rows,
            n_cols=n_cols,
            backend=backend,
        )

    def to_search_dag(self, *, soft: bool = True) -> dict[str, Any]:
        """Soft-optional CausalSearch DAG export (SearchDAG.v1 envelope)."""
        return self.discovery.to_search_dag(soft=soft)

    def estimate(self, **kwargs: Any) -> Any:
        """Delegate to ``discovery.estimate`` (session or attached frame)."""
        return self.discovery.estimate(**kwargs)

    def refute(self, edge: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        """Delegate to ``discovery.refute`` (session or attached frame)."""
        return self.discovery.refute(edge, **kwargs)

    def run_sensitivity(self, **kwargs: Any) -> Any:
        """Delegate to ``discovery.run_sensitivity``."""
        return self.discovery.run_sensitivity(**kwargs)

    def sensitivity(self, **kwargs: Any) -> Any:
        """Alias for ``run_sensitivity`` — matches ``AutoCausal.sensitivity``."""
        return self.discovery.sensitivity(**kwargs)

    def to_causaliv_request(self, **kwargs: Any) -> dict[str, Any]:
        """Delegate to ``discovery.to_causaliv_request``."""
        return self.discovery.to_causaliv_request(**kwargs)

    def engines_status(self) -> dict[str, Any]:
        """Unified engine status (stateless)."""
        return self.discovery.engines_status()

    def to_fabric_bundle(
        self,
        *,
        n_rows: int = 0,
        n_cols: int = 0,
        insight: Any = None,
    ) -> dict[str, Any]:
        """Assemble FabricBundle.v1 (MineReport + CausalEdges + optional InsightPack)."""
        from autocausal.contracts import fabric_bundle

        if n_rows <= 0 or n_cols <= 0:
            fr, fc = self.discovery._frame_shape()
            if n_rows <= 0:
                n_rows = fr
            if n_cols <= 0:
                n_cols = fc
        return fabric_bundle(
            mining=self.mining if self.mining is not None else self.discovery.mining,
            discovery=self.discovery,
            insight=insight,
            n_rows=n_rows,
            n_cols=n_cols,
            source=self.source,
            notes=list(self.notes),
            sensitivity=self.sensitivity_report
            if self.sensitivity_report is not None
            else self.discovery.sensitivity_report,
            extra={"qc": self.qc, "nlp_hints": self.nlp_hints},
        )
