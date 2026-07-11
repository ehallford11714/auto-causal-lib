"""MCP / AgentHook tool registry — maps tool names to AutoCausal library APIs.

Distinct from ``autocausal.skilling.ToolSurface`` (suite action tools for SLM).
This registry exposes high-level library entry points for external agents.
Optional suites soft-fail; handlers never crash the process on missing modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from autocausal.mcp.serialize import err_payload, ok_payload, to_jsonable
from autocausal.mcp.session import DEFAULT_SESSION, SessionStore

__all__ = [
    "ToolSpec",
    "ToolRegistry",
    "build_default_registry",
    "EPISTEMIC",
]

EPISTEMIC = (
    "AutoCausal outputs are exploratory assistance — not causal identification. "
    "Edges, roles, and SLM text are candidate hypotheses for human review."
)

Handler = Callable[[dict[str, Any], SessionStore], dict[str, Any]]


@dataclass
class ToolSpec:
    """One agent-callable tool (MCP + AgentHook)."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler = field(repr=False)
    optional_module: str = ""
    epistemic: str = EPISTEMIC

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": dict(self.parameters.get("properties") or self.parameters),
                "required": list(self.parameters.get("required") or []),
                "additionalProperties": True,
            },
            "epistemic": self.epistemic,
            "optional_module": self.optional_module or None,
        }


class ToolRegistry:
    """Name → ToolSpec map with soft invoke."""

    def __init__(self, tools: Optional[list[ToolSpec]] = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}. Known: {self.list_names()}")
        return self._tools[name]

    def list_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def list_tools(self) -> list[ToolSpec]:
        return [self._tools[n] for n in self.list_names()]

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.list_tools()]

    def invoke(
        self,
        name: str,
        args: Optional[dict[str, Any]] = None,
        *,
        store: Optional[SessionStore] = None,
    ) -> dict[str, Any]:
        args = dict(args or {})
        store = store or SessionStore()
        try:
            tool = self.get(name)
        except KeyError as e:
            return err_payload(str(e), tool=name)
        try:
            result = tool.handler(args, store)
            if not isinstance(result, dict):
                return ok_payload(tool=name, result=result)
            if "ok" not in result:
                result = {"ok": True, **result}
            result.setdefault("tool", name)
            result.setdefault("epistemic", tool.epistemic)
            return to_jsonable(result)  # type: ignore[return-value]
        except Exception as e:
            return err_payload(f"{type(e).__name__}: {e}", tool=name)


def _props(props: dict[str, Any], required: Optional[list[str]] = None) -> dict[str, Any]:
    return {"properties": props, "required": list(required or [])}


def _sid(args: dict[str, Any]) -> str:
    return str(args.get("session_id") or DEFAULT_SESSION)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _list_datasets(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    try:
        from autocausal.datasets import list_datasets
    except Exception as e:
        return err_payload(f"datasets module unavailable: {e}", tool="autocausal_list_datasets")
    items = list_datasets()
    datasets = [
        (i.to_dict() if hasattr(i, "to_dict") else to_jsonable(i)) for i in items
    ]
    return ok_payload(
        tool="autocausal_list_datasets",
        datasets=datasets,
        n=len(datasets),
    )


def _load_dataset(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    try:
        from autocausal.api import AutoCausal
        from autocausal.datasets import load_dataset
    except Exception as e:
        return err_payload(f"load_dataset unavailable: {e}", tool="autocausal_load_dataset")
    dataset_id = args.get("dataset_id") or args.get("id") or args.get("name")
    if not dataset_id:
        return err_payload("dataset_id is required", tool="autocausal_load_dataset")
    df = load_dataset(str(dataset_id))
    ac = AutoCausal.from_dataframe(df, source=f"dataset:{dataset_id}")
    sid = store.put(ac, args.get("session_id"))
    store.refresh_meta(sid)
    return ok_payload(
        tool="autocausal_load_dataset",
        session_id=sid,
        dataset_id=str(dataset_id),
        n_rows=len(ac.df),
        n_cols=len(ac.df.columns),
        columns=[str(c) for c in ac.df.columns],
        source=ac.source,
    )


def _from_csv(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    try:
        from autocausal.api import AutoCausal
    except Exception as e:
        return err_payload(f"AutoCausal unavailable: {e}", tool="autocausal_from_csv")
    path = args.get("path") or args.get("csv")
    if not path:
        return err_payload("path is required", tool="autocausal_from_csv")
    ac = AutoCausal.from_csv(str(path))
    sid = store.put(ac, args.get("session_id"))
    store.refresh_meta(sid)
    return ok_payload(
        tool="autocausal_from_csv",
        session_id=sid,
        path=str(path),
        n_rows=len(ac.df),
        n_cols=len(ac.df.columns),
        columns=[str(c) for c in ac.df.columns],
        source=ac.source,
    )


def _cleanse(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    sid = _sid(args)
    try:
        ac = store.get(sid)
    except KeyError as e:
        return err_payload(str(e), tool="autocausal_cleanse")
    try:
        use_slm = args.get("use_slm")
        text = str(args.get("text") or "")
        impute = str(args.get("impute") or "auto")
        ac.cleanse(
            use_slm=bool(use_slm) if use_slm is not None else False,
            text=text,
            impute=impute,
        )
        store.refresh_meta(sid)
        report = None
        if ac.cleanse_report is not None and hasattr(ac.cleanse_report, "to_dict"):
            report = ac.cleanse_report.to_dict()
        return ok_payload(
            tool="autocausal_cleanse",
            session_id=sid,
            n_rows=len(ac.df),
            n_cols=len(ac.df.columns),
            cleanse_report=report,
            source=ac.source,
        )
    except Exception as e:
        # Soft fallback: impute only
        try:
            ac.impute(method=str(args.get("impute") or "auto"))  # type: ignore[arg-type]
            store.refresh_meta(sid)
            return ok_payload(
                tool="autocausal_cleanse",
                session_id=sid,
                soft_fallback="impute",
                warning=f"cleanse suite soft-failed ({e}); ran impute instead",
                n_rows=len(ac.df),
                n_cols=len(ac.df.columns),
            )
        except Exception as e2:
            return err_payload(f"cleanse failed: {e}; impute failed: {e2}", tool="autocausal_cleanse")


def _eda(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    sid = _sid(args)
    try:
        ac = store.get(sid)
    except KeyError as e:
        return err_payload(str(e), tool="autocausal_eda")
    try:
        use_slm = args.get("use_slm")
        text = str(args.get("text") or "")
        ac.eda(use_slm=bool(use_slm) if use_slm is not None else False, text=text)
        store.refresh_meta(sid)
        report = None
        if ac.eda_report is not None and hasattr(ac.eda_report, "to_dict"):
            report = ac.eda_report.to_dict()
        return ok_payload(
            tool="autocausal_eda",
            session_id=sid,
            eda_report=report,
            source=ac.source,
        )
    except Exception as e:
        # Soft fallback: QC snapshot
        try:
            qc = ac.validate_qc(mode="warn")
            return ok_payload(
                tool="autocausal_eda",
                session_id=sid,
                soft_fallback="qc",
                warning=f"eda suite soft-failed ({e}); ran QC instead",
                qc=qc.to_dict() if hasattr(qc, "to_dict") else to_jsonable(qc),
            )
        except Exception as e2:
            return err_payload(f"eda failed: {e}; qc failed: {e2}", tool="autocausal_eda")


def _mine(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    sid = _sid(args)
    try:
        ac = store.get(sid)
    except KeyError as e:
        return err_payload(str(e), tool="autocausal_mine")
    min_score = float(args.get("min_score") or 0.15)
    use_suite = args.get("use_suite", True)
    try:
        if use_suite:
            use_slm = args.get("use_slm")
            text = str(args.get("text") or "")
            try:
                ac.automine(
                    use_slm=bool(use_slm) if use_slm is not None else False,
                    text=text,
                    min_score=min_score,
                )
            except Exception:
                ac.mine(min_score=min_score)
        else:
            ac.mine(min_score=min_score)
        store.refresh_meta(sid)
        mining = None
        if ac.mining is not None:
            mining = ac.mining.to_dict() if hasattr(ac.mining, "to_dict") else to_jsonable(ac.mining)
        mine_report = None
        if ac.mine_report is not None and hasattr(ac.mine_report, "to_dict"):
            mine_report = ac.mine_report.to_dict()
        return ok_payload(
            tool="autocausal_mine",
            session_id=sid,
            mining=mining,
            mine_report=mine_report,
            source=ac.source,
        )
    except Exception as e:
        return err_payload(f"mine failed: {e}", tool="autocausal_mine")


def _discover(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    sid = _sid(args)
    try:
        ac = store.get(sid)
    except KeyError as e:
        return err_payload(str(e), tool="autocausal_discover")
    kwargs: dict[str, Any] = {}
    for key in (
        "alpha",
        "max_cond_size",
        "min_abs_corr",
        "use_iv",
        "auto_instrument",
        "allow_iv_fallback",
        "stability",
        "bootstrap_n",
        "ensemble",
        "min_methods",
        "qc",
        "drop_id_columns",
        "seed",
        "random_state",
        "method",
        "include_optional",
        "mode",
        "strict",
    ):
        if key in args and args[key] is not None:
            kwargs[key] = args[key]
    if args.get("focus_columns"):
        kwargs["focus_columns"] = list(args["focus_columns"])
    if args.get("methods"):
        kwargs["methods"] = list(args["methods"])
    if args.get("candidates"):
        kwargs["candidates"] = dict(args["candidates"])
    try:
        result = ac.discover(**kwargs)
        payload = result.to_dict() if hasattr(result, "to_dict") else to_jsonable(result)
        return ok_payload(
            tool="autocausal_discover",
            session_id=sid,
            discovery=payload,
            n_edges=len(getattr(result, "edges", []) or []),
            source=ac.source,
        )
    except Exception as e:
        gate_error = e.to_dict() if hasattr(e, "to_dict") else None
        return err_payload(
            f"discover failed: {e}",
            tool="autocausal_discover",
            gate_error=gate_error,
        )


def _list_engines(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    try:
        from autocausal.engines import engine_status, list_engines

        kind = args.get("kind")
        if args.get("full") or not kind:
            return ok_payload(tool="autocausal_list_engines", **engine_status())
        engines = [e.to_dict() for e in list_engines(kind=str(kind))]
        return ok_payload(tool="autocausal_list_engines", kind=kind, engines=engines, n=len(engines))
    except Exception as e:
        return err_payload(f"list_engines failed: {e}", tool="autocausal_list_engines")


def _estimate(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    sid = _sid(args)
    try:
        if store.has(sid):
            ac = store.get(sid)
        elif args.get("path"):
            from autocausal import AutoCausal

            ac = AutoCausal.from_csv(str(args["path"]))
            store.put(ac, sid)
        else:
            return err_payload("session_id or path required", tool="autocausal_estimate")
        result = ac.estimate(
            backend=str(args.get("backend") or "builtin_ols"),
            y=args.get("y"),
            d=args.get("d"),
            x=list(args["x"]) if args.get("x") else None,
            z=args.get("z"),
            mode=args.get("mode"),
            strict=args.get("strict"),
            random_state=args.get("random_state", getattr(ac, "random_state", 0)),
        )
        return ok_payload(tool="autocausal_estimate", session_id=sid, **result.to_dict())
    except Exception as e:
        gate_error = e.to_dict() if hasattr(e, "to_dict") else None
        return err_payload(
            f"estimate failed: {e}",
            tool="autocausal_estimate",
            gate_error=gate_error,
        )


def _refute(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    sid = _sid(args)
    try:
        if store.has(sid):
            ac = store.get(sid)
            edge = args.get("edge")
            if edge is None and ac.result is not None and ac.result.edges:
                edge = ac.result.edges[0]
            result = ac.refute(
                edge=edge,
                method=str(args.get("method") or "placebo"),
                y=args.get("y"),
                d=args.get("d"),
                z=args.get("z"),
                mode=args.get("mode"),
                strict=args.get("strict"),
                seed=args.get("random_state", getattr(ac, "random_state", 0)),
            )
        else:
            return err_payload("session_id required (load data first)", tool="autocausal_refute")
        payload = result.to_dict() if hasattr(result, "to_dict") else to_jsonable(result)
        return ok_payload(tool="autocausal_refute", session_id=sid, refute=payload)
    except Exception as e:
        gate_error = e.to_dict() if hasattr(e, "to_dict") else None
        return err_payload(
            f"refute failed: {e}",
            tool="autocausal_refute",
            gate_error=gate_error,
        )


def _session_or_path(args: dict[str, Any], store: SessionStore, *, tool: str):
    sid = _sid(args)
    if store.has(sid):
        return store.get(sid), sid
    if args.get("path"):
        from autocausal import AutoCausal

        ac = AutoCausal.from_csv(str(args["path"]))
        store.put(ac, sid)
        return ac, sid
    raise ValueError(f"{tool}: session_id or path required")


def _correlate(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    try:
        ac, sid = _session_or_path(args, store, tool="autocausal_correlate")
        result = ac.correlate(
            x=args.get("x"),
            y=args.get("y"),
            columns=list(args["columns"]) if args.get("columns") else None,
            method=str(args.get("method") or "auto"),
            controls=list(args["controls"]) if args.get("controls") else None,
            bootstrap_n=int(args.get("bootstrap_n") or 0),
            permutation_n=int(args.get("permutation_n") or 0),
            alpha=float(args.get("alpha") or 0.05),
        )
        payload = result.to_dict() if hasattr(result, "to_dict") else to_jsonable(result)
        return ok_payload(
            tool="autocausal_correlate",
            session_id=sid,
            identification_evidence=False,
            correlation=payload,
        )
    except Exception as e:
        return err_payload(f"correlate failed: {e}", tool="autocausal_correlate")


def _tabular_ml(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    try:
        ac, sid = _session_or_path(args, store, tool="autocausal_tabular_ml")
        target = args.get("target")
        if not target:
            return err_payload("target is required", tool="autocausal_tabular_ml")
        mode = args.get("mode")
        if mode:
            ac.mode = str(mode)
        report = ac.tabular_ml(
            target=str(target),
            features=list(args["features"]) if args.get("features") else None,
            task=args.get("task"),
            group_column=args.get("group_column") or args.get("group"),
            time_column=args.get("time_column") or args.get("time"),
            calibrate=bool(args.get("calibrate", False)),
            enforce_gates=args.get("enforce_gates"),
        )
        payload = report.to_dict() if hasattr(report, "to_dict") else to_jsonable(report)
        return ok_payload(
            tool="autocausal_tabular_ml",
            session_id=sid,
            identification_evidence=False,
            report=payload,
        )
    except Exception as e:
        gate_error = e.to_dict() if hasattr(e, "to_dict") else None
        return err_payload(
            f"tabular_ml failed: {e}",
            tool="autocausal_tabular_ml",
            gate_error=gate_error,
        )


def _autoviz(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    try:
        ac, sid = _session_or_path(args, store, tool="autocausal_autoviz")
        mode = args.get("mode")
        if mode:
            ac.mode = str(mode)
        if args.get("discover"):
            ac.mine()
            ac.impute()
            ac.discover(use_iv=False, qc="off")
        report = ac.autoviz(use_slm=bool(args.get("use_slm", False)))
        payload = report.to_dict() if hasattr(report, "to_dict") else to_jsonable(report)
        return ok_payload(
            tool="autocausal_autoviz",
            session_id=sid,
            identification_evidence=False,
            report=payload,
        )
    except Exception as e:
        return err_payload(f"autoviz failed: {e}", tool="autocausal_autoviz")


def _insight_loop(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    sid = _sid(args)
    text = str(args.get("text") or "")
    use_slm = bool(args.get("use_slm")) if args.get("use_slm") is not None else False
    max_rounds = int(args.get("max_rounds") or args.get("rounds") or 1)
    try:
        from autocausal.insight import InsightSuite, run_insight_loop
    except Exception as e:
        return err_payload(
            f"insight module unavailable: {e}",
            tool="autocausal_insight_loop",
            soft=True,
        )
    try:
        if store.has(sid):
            ac = store.get(sid)
            suite = InsightSuite.from_autocausal(ac, use_slm=use_slm)
            if max_rounds > 1 and hasattr(suite, "run_loop"):
                # run_loop typically wants a path; use single-pass from instance
                report = suite.run(text=text, use_slm=use_slm)
            else:
                report = suite.run(text=text, use_slm=use_slm)
        else:
            path = args.get("path") or args.get("csv")
            dataset_id = args.get("dataset_id")
            if dataset_id:
                from autocausal.api import AutoCausal
                from autocausal.datasets import load_dataset

                df = load_dataset(str(dataset_id))
                ac = AutoCausal.from_dataframe(df, source=f"dataset:{dataset_id}")
                store.put(ac, sid)
                report = InsightSuite.from_autocausal(ac, use_slm=use_slm).run(
                    text=text, use_slm=use_slm
                )
            elif path:
                report = run_insight_loop(str(path), text=text, use_slm=use_slm)
            else:
                return err_payload(
                    "Need session_id with loaded data, or path/dataset_id",
                    tool="autocausal_insight_loop",
                )

        payload = report.to_dict() if hasattr(report, "to_dict") else to_jsonable(report)
        md = report.to_markdown() if hasattr(report, "to_markdown") else None
        store.set_insight(sid, payload)
        return ok_payload(
            tool="autocausal_insight_loop",
            session_id=sid,
            insight=payload,
            markdown=md,
            max_rounds=max_rounds,
        )
    except Exception as e:
        return err_payload(f"insight_loop failed: {e}", tool="autocausal_insight_loop", soft=True)


def _agentic_loop(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    """Run AgenticCausalLoop (hypothesize→skill→validate→compact→persist→route)."""
    sid = _sid(args)
    text = str(args.get("text") or "")
    use_slm = bool(args.get("use_slm")) if args.get("use_slm") is not None else False
    max_rounds = int(args.get("max_rounds") or args.get("rounds") or 2)
    persist_dir = args.get("persist_dir")
    prefer_langgraph = (
        bool(args.get("prefer_langgraph"))
        if args.get("prefer_langgraph") is not None
        else True
    )
    try:
        from autocausal.agentic import AgenticCausalLoop, run_agentic_loop
    except Exception as e:
        return err_payload(
            f"agentic module unavailable: {e}",
            tool="autocausal_agentic_loop",
            soft=True,
        )
    try:
        if store.has(sid):
            ac = store.get(sid)
            loop = AgenticCausalLoop(
                use_slm=use_slm,
                max_rounds=max_rounds,
                persist_dir=persist_dir,
                prefer_langgraph=prefer_langgraph,
            )
            report = loop.run(ac=ac, text=text, max_rounds=max_rounds, use_slm=use_slm)
        else:
            path = args.get("path") or args.get("csv")
            dataset_id = args.get("dataset_id")
            if dataset_id:
                from autocausal.api import AutoCausal
                from autocausal.datasets import load_dataset

                df = load_dataset(str(dataset_id))
                ac = AutoCausal.from_dataframe(df, source=f"dataset:{dataset_id}")
                store.put(ac, sid)
                report = AgenticCausalLoop(
                    use_slm=use_slm,
                    max_rounds=max_rounds,
                    persist_dir=persist_dir,
                    prefer_langgraph=prefer_langgraph,
                ).run(ac=ac, text=text, max_rounds=max_rounds, use_slm=use_slm)
            elif path:
                report = run_agentic_loop(
                    str(path),
                    text=text,
                    use_slm=use_slm,
                    max_rounds=max_rounds,
                    persist_dir=persist_dir,
                    prefer_langgraph=prefer_langgraph,
                )
            else:
                return err_payload(
                    "Need session_id with loaded data, or path/dataset_id",
                    tool="autocausal_agentic_loop",
                )

        payload = report.to_dict() if hasattr(report, "to_dict") else to_jsonable(report)
        md = report.to_markdown() if hasattr(report, "to_markdown") else None
        if hasattr(store, "set_insight"):
            try:
                store.set_insight(sid, payload)
            except Exception:
                pass
        return ok_payload(
            tool="autocausal_agentic_loop",
            session_id=sid,
            agentic=payload,
            markdown=md,
            max_rounds=max_rounds,
            n_rounds=getattr(report, "n_rounds", None),
            runtime_backend=getattr(report, "runtime_backend", None),
        )
    except Exception as e:
        return err_payload(
            f"agentic_loop failed: {e}", tool="autocausal_agentic_loop", soft=True
        )


def _recommend_experiments(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    sid = _sid(args)
    try:
        from autocausal.insight import ExperimentRecommender
    except Exception as e:
        return err_payload(
            f"ExperimentRecommender unavailable: {e}",
            tool="autocausal_recommend_experiments",
            soft=True,
        )
    try:
        use_slm = bool(args.get("use_slm")) if args.get("use_slm") is not None else False
        recommender = ExperimentRecommender(use_slm=use_slm)
        edges: list[dict[str, Any]] = []
        candidates: dict[str, Any] = {}
        mining: dict[str, Any] = {}
        if store.has(sid):
            ac = store.get(sid)
            if ac.result is not None:
                edges = list(ac.result.edges or [])
                candidates = dict(ac.result.candidates or {})
            if ac.mining is not None and hasattr(ac.mining, "to_dict"):
                mining = ac.mining.to_dict()
        plan = recommender.recommend(
            edges=edges,
            mining=mining,
            candidates=candidates,
            text=str(args.get("text") or ""),
        )
        payload = plan.to_dict() if hasattr(plan, "to_dict") else to_jsonable(plan)
        store.set_experiments(sid, payload)
        return ok_payload(
            tool="autocausal_recommend_experiments",
            session_id=sid,
            experiments=payload,
        )
    except Exception as e:
        return err_payload(
            f"recommend_experiments failed: {e}",
            tool="autocausal_recommend_experiments",
            soft=True,
        )


def _public_mine(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    try:
        from autocausal.api import AutoCausal
    except Exception as e:
        return err_payload(f"AutoCausal unavailable: {e}", tool="autocausal_public_mine")
    sources = args.get("sources")
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",") if s.strip()]
    try:
        report = AutoCausal.mine_public(
            sources,
            join_on=args.get("join_on"),
            how=str(args.get("how") or "outer"),
            allow_network=bool(args.get("allow_network") or False),
            discover=bool(args.get("discover") if args.get("discover") is not None else True),
            use_iv=bool(args.get("use_iv") if args.get("use_iv") is not None else True),
            min_score=float(args.get("min_score") or 0.15),
            min_abs_corr=float(args.get("min_abs_corr") or 0.12),
            validate=bool(args.get("validate") or False),
        )
        sid = _sid(args)
        payload = report.to_dict() if hasattr(report, "to_dict") else to_jsonable(report)
        md = report.to_markdown() if hasattr(report, "to_markdown") else None
        store.set_public(sid, payload)
        return ok_payload(
            tool="autocausal_public_mine",
            session_id=sid,
            public_report=payload,
            markdown=md,
            sources=sources,
            n_edges=len(getattr(report, "edges", []) or []),
        )
    except Exception as e:
        return err_payload(f"public_mine failed: {e}", tool="autocausal_public_mine", soft=True)


def _report(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    sid = _sid(args)
    fmt = str(args.get("format") or ("markdown" if args.get("as_markdown", True) else "json")).lower()
    try:
        ac = store.get(sid)
    except KeyError as e:
        # Fall back to last insight / public
        insight = store.get_insight(sid)
        public = store.get_public(sid)
        if insight or public:
            return ok_payload(
                tool="autocausal_report",
                session_id=sid,
                format=fmt,
                insight=insight,
                public_report=public,
                note="No live session frame; returning cached insight/public payloads.",
            )
        return err_payload(str(e), tool="autocausal_report")
    try:
        as_md = fmt in ("md", "markdown", "text")
        body = ac.report(as_markdown=as_md)
        out: dict[str, Any] = {
            "ok": True,
            "tool": "autocausal_report",
            "session_id": sid,
            "format": "markdown" if as_md else "json",
            "source": ac.source,
        }
        if as_md:
            out["markdown"] = body
        else:
            try:
                import json

                out["report"] = json.loads(body) if isinstance(body, str) else body
            except Exception:
                out["report"] = body
        # Attach suite reports if present
        if ac.cleanse_report is not None and hasattr(ac.cleanse_report, "to_dict"):
            out["cleanse_report"] = ac.cleanse_report.to_dict()
        if ac.eda_report is not None and hasattr(ac.eda_report, "to_dict"):
            out["eda_report"] = ac.eda_report.to_dict()
        if ac.mine_report is not None and hasattr(ac.mine_report, "to_dict"):
            out["mine_report"] = ac.mine_report.to_dict()
        return to_jsonable(out)  # type: ignore[return-value]
    except Exception as e:
        return err_payload(f"report failed: {e}", tool="autocausal_report")


def _skilling_list(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    try:
        from autocausal.skilling import skill_catalog
    except Exception as e:
        return err_payload(
            f"skilling module unavailable: {e}",
            tool="autocausal_skilling_list",
            soft=True,
        )
    try:
        cat = skill_catalog()
        return ok_payload(tool="autocausal_skilling_list", catalog=cat)
    except Exception as e:
        return err_payload(f"skilling_list failed: {e}", tool="autocausal_skilling_list", soft=True)


def _session_status(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    return ok_payload(
        tool="autocausal_session_status",
        sessions=store.list_sessions(),
        has_default=store.has(DEFAULT_SESSION),
    )


def _list_mcp_tools(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    # Filled after registry build via closure — placeholder replaced in build_default_registry
    return ok_payload(tool="autocausal_list_tools", tools=[])


def _list_integrations(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    from autocausal.integrations import list_integrations

    items = list_integrations(
        category=args.get("category"),
        deep=bool(args.get("deep", False)),
    )
    return ok_payload(
        tool="autocausal_list_integrations",
        integrations=[item.to_dict() for item in items],
        n=len(items),
        telemetry_enabled=False,
    )


def _integration_status(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    from autocausal.integrations import integration_status

    integration_id = args.get("integration_id") or args.get("id")
    if not integration_id:
        return err_payload(
            "integration_id is required",
            tool="autocausal_integration_status",
        )
    status = integration_status(
        str(integration_id),
        deep=bool(args.get("deep", False)),
    )
    return ok_payload(
        tool="autocausal_integration_status",
        integration=status.to_dict(),
    )


def _route_capability(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
    from autocausal.integrations import (
        CapabilityRouter,
        ResourceBudget,
        get_default_registry,
    )

    capability = args.get("capability")
    if not capability:
        return err_payload(
            "capability is required",
            tool="autocausal_route_capability",
        )
    budget_value = args.get("budget") or {}
    budget = (
        budget_value
        if isinstance(budget_value, ResourceBudget)
        else ResourceBudget(**dict(budget_value))
    )
    decision = CapabilityRouter(get_default_registry()).route(
        str(capability),
        policy=args.get("policy"),
        budget=budget,
        data_type=args.get("data_type"),
        n_rows=args.get("n_rows"),
        estimated_memory_mb=args.get("estimated_memory_mb"),
        deep_health=bool(args.get("deep", False)),
        context=args.get("context"),
    )
    return ok_payload(
        tool="autocausal_route_capability",
        decision=decision.to_dict(),
        invoked=False,
    )


def build_default_registry() -> ToolRegistry:
    """Construct the default agent-facing tool surface."""
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="autocausal_list_integrations",
            description=(
                "List maintained optional integrations without importing heavy packages."
            ),
            parameters=_props(
                {
                    "category": {"type": "string"},
                    "deep": {
                        "type": "boolean",
                        "default": False,
                        "description": "Explicitly import registered adapters for health probes.",
                    },
                }
            ),
            handler=_list_integrations,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_integration_status",
            description="Inspect one integration's install, version, policy, and health state.",
            parameters=_props(
                {
                    "integration_id": {"type": "string"},
                    "deep": {"type": "boolean", "default": False},
                },
                required=["integration_id"],
            ),
            handler=_integration_status,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_route_capability",
            description=(
                "Return a policy/resource routing decision without invoking an adapter."
            ),
            parameters=_props(
                {
                    "capability": {"type": "string"},
                    "policy": {"type": "object"},
                    "budget": {"type": "object"},
                    "data_type": {"type": "string"},
                    "n_rows": {"type": "integer"},
                    "estimated_memory_mb": {"type": "integer"},
                    "context": {"type": "object"},
                    "deep": {"type": "boolean", "default": False},
                },
                required=["capability"],
            ),
            handler=_route_capability,
        )
    )

    registry.register(
        ToolSpec(
            name="autocausal_list_datasets",
            description="List bundled example dataset IDs (iris, wine, titanic, …).",
            parameters=_props({}),
            handler=_list_datasets,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_load_dataset",
            description="Load a bundled example dataset into a session as AutoCausal.",
            parameters=_props(
                {
                    "dataset_id": {
                        "type": "string",
                        "description": "Dataset id (e.g. iris, wine, titanic)",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session key (default: 'default')",
                    },
                },
                required=["dataset_id"],
            ),
            handler=_load_dataset,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_from_csv",
            description="Load a CSV path into a session as AutoCausal.",
            parameters=_props(
                {
                    "path": {"type": "string", "description": "Path to CSV file"},
                    "session_id": {"type": "string"},
                },
                required=["path"],
            ),
            handler=_from_csv,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_cleanse",
            description="Run AutoCleanseSuite (or soft-fallback impute) on the session frame.",
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "use_slm": {"type": "boolean", "default": False},
                    "text": {"type": "string"},
                    "impute": {
                        "type": "string",
                        "enum": ["auto", "median_mode", "knn"],
                        "default": "auto",
                    },
                }
            ),
            handler=_cleanse,
            optional_module="autocausal.suites",
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_eda",
            description="Run AutoEDASuite (or soft-fallback QC) on the session frame.",
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "use_slm": {"type": "boolean", "default": False},
                    "text": {"type": "string"},
                }
            ),
            handler=_eda,
            optional_module="autocausal.suites",
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_mine",
            description="Run AutoMineSuite or core mine() on the session frame.",
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "min_score": {"type": "number", "default": 0.15},
                    "use_suite": {"type": "boolean", "default": True},
                    "use_slm": {"type": "boolean", "default": False},
                    "text": {"type": "string"},
                }
            ),
            handler=_mine,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_discover",
            description="Run exploratory causal discovery on the session frame.",
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "alpha": {"type": "number", "default": 0.05},
                    "min_abs_corr": {"type": "number", "default": 0.15},
                    "use_iv": {"type": "boolean", "default": True},
                    "auto_instrument": {
                        "type": "boolean",
                        "default": False,
                        "description": "Opt-in: synthesize exploratory auto_instrument_z (demo only; identification=none). Production mode refuses this.",
                    },
                    "allow_iv_fallback": {
                        "type": "boolean",
                        "default": False,
                        "description": "Propose weak correlate instrument candidates (not identification)",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["exploratory", "production"],
                        "default": "exploratory",
                        "description": "production refuses synthetic IV and applies policy ensemble+stability+QC gates",
                    },
                    "candidates": {
                        "type": "object",
                        "description": "Optional role injection: treatment/outcome/instrument/confounder lists",
                    },
                    "stability": {"type": "boolean"},
                    "ensemble": {"type": "boolean"},
                    "random_state": {
                        "type": "integer",
                        "description": "Unified deterministic seed for the run",
                    },
                    "method": {
                        "type": "string",
                        "description": "Single method e.g. score_pc_lite, causal_learn_pc, lingam",
                    },
                    "methods": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ensemble methods including soft causal-learn/lingam/gcastle",
                    },
                    "qc": {"type": "string", "enum": ["off", "warn", "block"], "default": "warn"},
                    "focus_columns": {"type": "array", "items": {"type": "string"}},
                }
            ),
            handler=_discover,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_list_engines",
            description="List discovery/estimate/refute/package engines and connectivity map.",
            parameters=_props(
                {
                    "kind": {
                        "type": "string",
                        "enum": ["discovery", "estimate", "refute", "package"],
                    },
                    "full": {"type": "boolean", "default": True},
                }
            ),
            handler=_list_engines,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_estimate",
            description="Estimate ATE/CATE via builtin_ols / doubleml / econml (soft-optional).",
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "path": {"type": "string"},
                    "backend": {
                        "type": "string",
                        "default": "builtin_ols",
                        "description": "builtin_ols | doubleml | econml | econml_causal_forest | builtin_2sls",
                    },
                    "y": {"type": "string"},
                    "d": {"type": "string"},
                    "x": {"type": "array", "items": {"type": "string"}},
                    "z": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["exploratory", "production"],
                    },
                    "random_state": {"type": "integer"},
                }
            ),
            handler=_estimate,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_correlate",
            description=(
                "Descriptive association / correlation analysis. "
                "Never causal identification evidence."
            ),
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "path": {"type": "string"},
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "method": {"type": "string", "default": "auto"},
                    "controls": {"type": "array", "items": {"type": "string"}},
                    "bootstrap_n": {"type": "integer", "default": 0},
                    "permutation_n": {"type": "integer", "default": 0},
                    "alpha": {"type": "number", "default": 0.05},
                }
            ),
            handler=_correlate,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_tabular_ml",
            description=(
                "Run leakage-safe AutoTabularML. Predictive metrics are not "
                "causal effects."
            ),
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "path": {"type": "string"},
                    "target": {"type": "string"},
                    "features": {"type": "array", "items": {"type": "string"}},
                    "task": {"type": "string"},
                    "group_column": {"type": "string"},
                    "time_column": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["exploratory", "production"],
                    },
                    "calibrate": {"type": "boolean", "default": False},
                    "enforce_gates": {"type": "boolean"},
                },
                required=["target"],
            ),
            handler=_tabular_ml,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_autoviz",
            description=(
                "Plan analysis-aware visualizations. Charts are descriptive "
                "and do not prove causal effects."
            ),
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "path": {"type": "string"},
                    "discover": {"type": "boolean", "default": False},
                    "use_slm": {"type": "boolean", "default": False},
                    "mode": {
                        "type": "string",
                        "enum": ["exploratory", "production"],
                    },
                }
            ),
            handler=_autoviz,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_refute",
            description="Refute an edge via placebo builtin or DoWhy refute_estimate (soft).",
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "method": {
                        "type": "string",
                        "default": "placebo",
                        "description": "placebo | random_common_cause | dowhy | dowhy_data_subset | …",
                    },
                    "edge": {"type": "object"},
                    "y": {"type": "string"},
                    "d": {"type": "string"},
                    "z": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["exploratory", "production"],
                    },
                    "random_state": {"type": "integer"},
                }
            ),
            handler=_refute,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_insight_loop",
            description="Run InsightSuite / insight loop (soft if insight module missing).",
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "path": {"type": "string", "description": "CSV path if no session"},
                    "dataset_id": {"type": "string"},
                    "text": {"type": "string"},
                    "use_slm": {"type": "boolean", "default": False},
                    "max_rounds": {"type": "integer", "default": 1},
                }
            ),
            handler=_insight_loop,
            optional_module="autocausal.insight",
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_agentic_loop",
            description=(
                "Run SLM-guided agentic causal loop "
                "(hypothesize→skill→validate→compact→persist→route). Soft LangGraph/FSM."
            ),
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "path": {"type": "string", "description": "CSV path if no session"},
                    "dataset_id": {"type": "string"},
                    "text": {"type": "string"},
                    "use_slm": {"type": "boolean", "default": False},
                    "max_rounds": {"type": "integer", "default": 2},
                    "persist_dir": {
                        "type": "string",
                        "description": "Optional JSONL episode directory",
                    },
                    "prefer_langgraph": {
                        "type": "boolean",
                        "default": True,
                        "description": "Try LangGraph if installed; else offline FSM",
                    },
                }
            ),
            handler=_agentic_loop,
            optional_module="autocausal.agentic",
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_recommend_experiments",
            description="Recommend next experiments / mining steps from session context.",
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "text": {"type": "string"},
                    "use_slm": {"type": "boolean", "default": False},
                }
            ),
            handler=_recommend_experiments,
            optional_module="autocausal.insight",
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_public_mine",
            description="Mine + discover across public suite sources (offline demos by default).",
            parameters=_props(
                {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Public source ids, e.g. finance_demo, demographics_demo",
                    },
                    "join_on": {"type": "string"},
                    "discover": {"type": "boolean", "default": True},
                    "use_iv": {"type": "boolean", "default": True},
                    "allow_network": {"type": "boolean", "default": False},
                    "session_id": {"type": "string"},
                }
            ),
            handler=_public_mine,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_report",
            description="Emit markdown or JSON discovery report for the session.",
            parameters=_props(
                {
                    "session_id": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "default": "markdown",
                    },
                    "as_markdown": {"type": "boolean", "default": True},
                }
            ),
            handler=_report,
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_skilling_list",
            description="List SLM skilling catalog (skills + suite ToolSurface schemas).",
            parameters=_props({}),
            handler=_skilling_list,
            optional_module="autocausal.skilling",
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_session_status",
            description="List active AutoCausal MCP/AgentHook sessions.",
            parameters=_props({}),
            handler=_session_status,
        )
    )

    # Soft GRAIL tools (Kineteq GRAIL adaptation — stub always available)
    def _register_grail_tools() -> None:
        try:
            from autocausal.grail.mcp_tools import MCP_TOOL_SCHEMAS, dispatch_grail_tool
            from autocausal.grail.types import EPISTEMIC as GRAIL_EPISTEMIC
        except Exception:
            return

        def _make_handler(tool_name: str) -> Handler:
            def _handler(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
                # Enrich from session frame when present
                enriched = dict(args)
                sid = _sid(args)
                if store.has(sid):
                    try:
                        ac = store.get(sid)
                        if "columns" not in enriched:
                            enriched["columns"] = [str(c) for c in ac.df.columns]
                        if "edges" not in enriched and ac.result is not None:
                            enriched["edges"] = list(ac.result.edges or [])
                        if not enriched.get("text") and not enriched.get("goal"):
                            enriched["text"] = str(getattr(ac, "source", "") or "")
                    except Exception:
                        pass
                out = dispatch_grail_tool(tool_name, enriched)
                if not out.get("ok", True):
                    return err_payload(
                        str(out.get("error") or "grail tool failed"),
                        tool=tool_name,
                        soft=True,
                    )
                return ok_payload(tool=tool_name, session_id=sid, **{
                    k: v for k, v in out.items() if k != "ok"
                })

            return _handler

        for schema in MCP_TOOL_SCHEMAS:
            name = schema["name"]
            params = schema.get("parameters") or {}
            props = dict(params.get("properties") or {})
            # Allow session_id on all GRAIL tools
            props.setdefault("session_id", {"type": "string"})
            registry.register(
                ToolSpec(
                    name=name,
                    description=schema["description"],
                    parameters=_props(props, required=list(params.get("required") or [])),
                    handler=_make_handler(name),
                    optional_module="autocausal.grail",
                    epistemic=GRAIL_EPISTEMIC,
                )
            )

    _register_grail_tools()

    # Deep research stays in its own package; this narrow adapter avoids
    # coupling the core registry to provider/SLM implementations.
    try:
        from autocausal.research.mcp import register_research_tools

        register_research_tools(
            registry,
            tool_spec_cls=ToolSpec,
            props_fn=_props,
        )
    except Exception:
        # Optional surface: importing core AutoCausal must remain offline-safe.
        pass

    try:
        from autocausal.reporting.tools import register_reporting_mcp_tools

        register_reporting_mcp_tools(registry)
    except Exception:
        # Reporting stays lazy so MCP remains usable without PDF dependencies.
        pass

    def _list_tools(args: dict[str, Any], store: SessionStore) -> dict[str, Any]:
        return ok_payload(
            tool="autocausal_list_tools",
            tools=registry.schemas(),
            n=len(registry.list_names()),
        )

    registry.register(
        ToolSpec(
            name="autocausal_list_tools",
            description="List all AutoCausal MCP / AgentHook tools and JSON schemas.",
            parameters=_props({}),
            handler=_list_tools,
        )
    )
    return registry
