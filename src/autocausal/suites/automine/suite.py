"""AutoMineSuite — SLM-directed orchestration over MineActions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pandas as pd

from autocausal.suites.automine.actions import MINE_REGISTRY, MineActions
from autocausal.suites.automine.report import MineReport
from autocausal.suites.base import resolve_frame
from autocausal.suites.director import SLMAutoDirector, SLMDirectives, resolve_suite_slm

__all__ = ["AutoMineSuite", "auto_mine"]


class AutoMineSuite:
    """Library-first mining facade (SLM picks action sequence when available).

    Example::

        from autocausal.suites.automine import AutoMineSuite, MineActions
        MineActions.mine_associations(df)
        mine = AutoMineSuite(df, use_slm=True).run()
    """

    def __init__(
        self,
        source: Any = None,
        *,
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        actions: Optional[Sequence[str]] = None,
        text: str = "",
        min_score: float = 0.15,
        join_public: Optional[Union[str, list[str]]] = None,
        allow_network: bool = False,
        include_behavioral: bool = False,
        try_datamine: bool = True,
        table: Optional[str] = None,
        query: Optional[str] = None,
    ) -> None:
        self.source = source
        self.use_slm = resolve_suite_slm(use_slm)
        self.model_name = model_name
        self.actions_override = list(actions) if actions else None
        self.text = text
        self.min_score = min_score
        self.join_public = join_public
        self.allow_network = allow_network
        self.include_behavioral = include_behavioral
        self.try_datamine = try_datamine
        self.table = table
        self.query = query
        self.frame: Optional[pd.DataFrame] = None
        self.report: Optional[MineReport] = None
        self.directives: Optional[SLMDirectives] = None
        self._mining: Any = None

    def run(self, source: Any = None, *, text: Optional[str] = None) -> "AutoMineSuite":
        src = self.source if source is None else source
        if src is None:
            raise ValueError("AutoMineSuite requires a DataFrame, path, or AutoCausal")
        df, label, ac_in = resolve_frame(src, table=self.table, query=self.query)
        txt = self.text if text is None else text

        skip_join = self.join_public is None
        director = SLMAutoDirector(use_slm=self.use_slm, model_name=self.model_name)
        directives = director.direct(
            "mine",
            df,
            text=txt,
            context={
                "skip_join": skip_join,
                "include_behavioral": self.include_behavioral,
            },
        )
        self.directives = directives

        sequence = self.actions_override or list(directives.actions) or MineActions.default_sequence()
        # Ensure join only when requested
        if skip_join:
            sequence = [a for a in sequence if a != "join_public_sources"]
        elif self.join_public is not None and "join_public_sources" not in sequence:
            sequence = ["join_public_sources"] + sequence
        if self.include_behavioral and "mine_behavioral" not in sequence:
            sequence.append("mine_behavioral")
        sequence = [a for a in sequence if a in MINE_REGISTRY]
        if not sequence:
            sequence = MineActions.default_sequence()

        warnings: list[str] = []
        notes = [
            "AutoMine associations are exploratory — not causal effects.",
        ]
        action_results: list[dict[str, Any]] = []
        actions_run: list[str] = []
        tools_invoked: list[dict[str, Any]] = []
        join_log: list[dict[str, Any]] = []
        columns: list[dict[str, Any]] = []
        associations: list[dict[str, Any]] = []
        suggestions: list[dict[str, Any]] = []
        kpis: list[str] = []
        ranked: list[dict[str, Any]] = []
        fabric: Optional[dict[str, Any]] = None
        behavioral: Optional[dict[str, Any]] = None
        datamine_payload: Optional[dict[str, Any]] = None
        typed_associations: list[dict[str, Any]] = []

        out = df
        for name in sequence:
            kwargs: dict[str, Any] = {}
            if name == "mine_associations":
                kwargs["min_score"] = self.min_score
            elif name == "join_public_sources":
                if self.join_public == "auto":
                    kwargs["sources"] = directives.join_sources[:2]
                else:
                    kwargs["sources"] = self.join_public
                kwargs["allow_network"] = self.allow_network
            elif name == "mine_kpi_hints":
                kwargs["kpi_focus"] = directives.kpi_focus
                kwargs["mining"] = self._mining
            elif name == "rank_candidates":
                kwargs["associations"] = associations
                kwargs["suggestions"] = suggestions
                kwargs["focus"] = directives.focus_columns
                kwargs["kpis"] = kpis
            elif name == "to_mine_report":
                kwargs["columns"] = columns
                kwargs["associations"] = associations
                kwargs["suggestions"] = suggestions
                kwargs["kpis"] = kpis
            elif name == "mine_behavioral":
                kwargs["discover"] = False

            try:
                result = MINE_REGISTRY.run(name, out, **kwargs)
            except Exception as e:
                warnings.append(f"Action `{name}` soft-fail: {type(e).__name__}: {e}")
                continue

            actions_run.append(name)
            # Strip non-serializable from payload copy
            pdata = dict(result.payload or {})
            mining_obj = pdata.pop("_mining", None)
            if mining_obj is not None:
                self._mining = mining_obj
            action_results.append(
                {
                    "name": result.name,
                    "payload": {k: v for k, v in pdata.items() if k != "_mining"},
                    "ops": result.ops,
                    "warnings": result.warnings,
                    "notes": result.notes,
                    "n_affected": result.n_affected,
                }
            )
            tools_invoked.append({"tool": f"automine.{name}", "ok": True})
            warnings.extend(result.warnings)
            notes.extend(result.notes)
            if result.frame is not None:
                out = result.frame

            if name == "mine_associations":
                columns = list(pdata.get("columns") or [])
                associations = list(pdata.get("associations") or [])
                suggestions = list(pdata.get("suggestions") or [])
                kpis = list(dict.fromkeys(list(pdata.get("kpis") or []) + kpis))
            elif name == "mine_kpi_hints":
                kpis = list(dict.fromkeys(list(pdata.get("kpis") or []) + kpis))
            elif name == "join_public_sources":
                join_log.extend(list(pdata.get("join_log") or []))
            elif name == "rank_candidates":
                associations = list(pdata.get("associations") or associations)
                suggestions = list(pdata.get("suggestions") or suggestions)
                ranked = list(pdata.get("ranked_candidates") or [])
            elif name == "to_mine_report":
                fabric = pdata.get("fabric_envelope")
            elif name == "mine_behavioral":
                behavioral = pdata.get("behavioral")

        if self.try_datamine:
            try:
                from autocausal.datamine_adapter import available, mine_via_datamine

                if available():
                    datamine_payload = mine_via_datamine(out, min_score=self.min_score)
                    notes.append("DataMineLib adapter used (soft).")
                else:
                    notes.append("DataMineLib not on path — autocausal.mining only.")
            except Exception as e:
                warnings.append(f"DataMine soft-fail: {type(e).__name__}: {e}")

        typed_columns = [
            str(column)
            for column in out.columns
            if pd.api.types.is_numeric_dtype(out[column])
            or out[column].nunique(dropna=True) <= 100
        ][:12]
        if len(typed_columns) >= 2:
            try:
                from autocausal.correlation import correlation_matrix

                typed_scan = correlation_matrix(
                    out,
                    columns=typed_columns,
                    method="auto",
                    random_state=int(
                        getattr(ac_in, "random_state", 0) if ac_in is not None else 0
                    ),
                )
                typed_associations = [
                    {
                        **result.to_dict(),
                        "evidence_type": "descriptive_association",
                        "identification_evidence": False,
                    }
                    for result in typed_scan.results
                ]
            except Exception as e:
                warnings.append(
                    f"Typed association scan soft-fail: {type(e).__name__}: {e}"
                )

        dir_dict = directives.to_dict()
        dir_dict["tools_invoked"] = tools_invoked

        report = MineReport(
            n_rows=len(out),
            n_cols=out.shape[1],
            columns=columns,
            associations=associations,
            suggestions=suggestions,
            kpis=kpis,
            ranked_candidates=ranked,
            join_log=join_log,
            datamine=datamine_payload,
            behavioral=behavioral,
            fabric_envelope=fabric,
            action_results=action_results,
            actions_run=actions_run,
            slm_directives=dir_dict,
            notes=list(dict.fromkeys(notes + list(directives.notes))),
            warnings=warnings,
            source=label,
            backend=directives.backend,
            typed_associations=typed_associations,
        )
        self.frame = out
        self.report = report

        if ac_in is not None:
            try:
                ac_in._df = out
                ac_in.mining = self._mining
                ac_in.mine_report = report
                ac_in.join_log.extend(join_log)
            except Exception:
                pass
        return self

    def to_autocausal(self) -> Any:
        from autocausal.api import AutoCausal

        if self.frame is None:
            self.run()
        assert self.frame is not None
        ac = AutoCausal.from_dataframe(
            self.frame, source=f"mine:{self.report.source if self.report else 'memory'}"
        )
        ac.mining = self._mining
        ac.mine_report = self.report
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

    def to_mine_report(self) -> dict[str, Any]:
        if self.report is None:
            self.run()
        assert self.report is not None
        return self.report.to_mine_report()

    def write(self, path: Union[str, Path], *, fmt: str = "auto") -> Path:
        if self.report is None:
            self.run()
        assert self.report is not None
        return self.report.write(path, fmt=fmt)


def auto_mine(source: Any, *, use_slm: Optional[bool] = None, **kwargs: Any) -> MineReport:
    suite = AutoMineSuite(source, use_slm=use_slm, **kwargs).run()
    assert suite.report is not None
    return suite.report
