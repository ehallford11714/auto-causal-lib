"""AutoMineSuite — SLM-directed automated mining facade."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from autocausal.suites.base import resolve_frame, write_report
from autocausal.suites.director import (
    EPISTEMIC_NOTE,
    SLMAutoDirector,
    SLMDirectives,
    resolve_suite_slm,
)

__all__ = ["MineReport", "AutoMineSuite", "auto_mine"]


@dataclass
class MineReport:
    """Structured mining report with optional Fabric ``to_mine_report`` export."""

    n_rows: int
    n_cols: int
    columns: list[dict[str, Any]] = field(default_factory=list)
    associations: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    kpis: list[str] = field(default_factory=list)
    join_log: list[dict[str, Any]] = field(default_factory=list)
    datamine: Optional[dict[str, Any]] = None
    slm_directives: Optional[dict[str, Any]] = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: str = ""
    backend: str = "rule"
    mining_backend: str = "autocausal.mining"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": list(self.columns),
            "associations": list(self.associations),
            "suggestions": list(self.suggestions),
            "kpis": list(self.kpis),
            "join_log": list(self.join_log),
            "datamine": self.datamine,
            "slm_directives": self.slm_directives,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "source": self.source,
            "backend": self.backend,
            "mining_backend": self.mining_backend,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_mine_report(self, *, backend: Optional[str] = None) -> dict[str, Any]:
        """Export as MineReport.v1 Fabric envelope."""
        from autocausal.contracts import mining_to_mine_report

        return mining_to_mine_report(
            self,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            backend=backend or self.mining_backend,
            extra_meta={"suite": "AutoMineSuite", "director_backend": self.backend},
        )

    def to_markdown(self) -> str:
        lines = [
            "# AutoMine report",
            "",
            f"- rows={self.n_rows}, cols={self.n_cols}",
            f"- director backend: `{self.backend}`",
            f"- mining backend: `{self.mining_backend}`",
            "",
            f"> {EPISTEMIC_NOTE}",
            "",
        ]
        if self.kpis:
            lines += ["## Suggested KPIs", ""]
            for k in self.kpis:
                lines.append(f"- `{k}`")
            lines.append("")
        lines += ["## Top associations", ""]
        if not self.associations:
            lines.append("_None above threshold._")
        else:
            lines += ["| a | b | metric | score |", "|---|---|---|---:|"]
            for a in self.associations[:25]:
                lines.append(
                    f"| `{a.get('a')}` | `{a.get('b')}` | {a.get('metric', '')} | {a.get('score', '')} |"
                )
        lines.append("")
        if self.suggestions:
            lines += ["## Suggested relationships", ""]
            for s in self.suggestions[:20]:
                lines.append(
                    f"- `{s.get('source')}` → `{s.get('target')}` "
                    f"({s.get('reason', '')}; score={s.get('score', '')})"
                )
            lines.append("")
        if self.join_log:
            lines += ["## Joins", ""]
            for j in self.join_log:
                lines.append(f"- {j}")
            lines.append("")
        if self.warnings:
            lines += ["## Warnings", ""]
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""]
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")
        return "\n".join(lines)

    def write(self, path: Union[str, Path], *, fmt: str = "auto") -> Path:
        return write_report(self, path, fmt=fmt)


class AutoMineSuite:
    """Library-first mining facade (SLM-directed).

    Wraps ``autocausal.mining`` + optional public joins + soft DataMineLib adapter.

    Example::

        from autocausal import AutoMineSuite
        mine = AutoMineSuite(df, use_slm=True).run()
        print(mine.report.to_mine_report()["schema"])
    """

    def __init__(
        self,
        source: Any = None,
        *,
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        text: str = "",
        min_score: float = 0.15,
        join_public: Optional[Union[str, list[str]]] = None,
        allow_network: bool = False,
        try_datamine: bool = True,
        table: Optional[str] = None,
        query: Optional[str] = None,
    ) -> None:
        self.source = source
        self.use_slm = resolve_suite_slm(use_slm)
        self.model_name = model_name
        self.text = text
        self.min_score = min_score
        self.join_public = join_public
        self.allow_network = allow_network
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

        director = SLMAutoDirector(use_slm=self.use_slm, model_name=self.model_name)
        skip_join = self.join_public is None
        directives = director.direct(
            "mine",
            df,
            text=txt,
            context={"skip_join": skip_join},
        )
        self.directives = directives

        join_log: list[dict[str, Any]] = []
        warnings: list[str] = []
        notes = [
            "AutoMine associations are exploratory — not causal effects.",
            EPISTEMIC_NOTE,
        ]

        # Joins: explicit ids / "auto" (director); None = no join
        join_ids: list[str] = []
        if self.join_public == "auto":
            join_ids = list(directives.join_sources)[:2]
        elif isinstance(self.join_public, str):
            join_ids = [x.strip() for x in self.join_public.split(",") if x.strip()]
        elif self.join_public is not None:
            join_ids = list(self.join_public)

        if join_ids:
            try:
                from autocausal.public_suite import join_public_frames

                df, log = join_public_frames(
                    df,
                    join_ids if len(join_ids) > 1 else join_ids[0],
                    allow_network=self.allow_network,
                )
                join_log.extend(log if isinstance(log, list) else [log])
                notes.append(f"Joined public: {join_ids}")
            except Exception as e:
                warnings.append(f"Public join soft-fail: {type(e).__name__}: {e}")

        # Core mining via autocausal.mining
        from autocausal.mining import mine

        mining = mine(df, min_score=self.min_score)
        self._mining = mining

        # Soft DataMineLib adapter
        datamine_payload: Optional[dict[str, Any]] = None
        if self.try_datamine:
            try:
                from autocausal.datamine_adapter import available, mine_via_datamine

                if available():
                    datamine_payload = mine_via_datamine(df, min_score=self.min_score)
                    notes.append("DataMineLib adapter used (soft).")
                else:
                    notes.append("DataMineLib not on path — autocausal.mining only.")
            except Exception as e:
                warnings.append(f"DataMine soft-fail: {type(e).__name__}: {e}")

        kpis = list(getattr(mining, "kpis", None) or [])
        # Director KPI focus first
        for k in directives.kpi_focus:
            if k not in kpis and k in df.columns:
                kpis.insert(0, k)

        # Prioritize associations involving focus / KPI columns
        assocs = list(getattr(mining, "associations", None) or [])
        focus = set(directives.focus_columns) | set(kpis)
        if focus:
            assocs = sorted(
                assocs,
                key=lambda a: (
                    0 if {a.get("a"), a.get("b")} & focus else 1,
                    -float(a.get("score") or 0),
                ),
            )

        report = MineReport(
            n_rows=len(df),
            n_cols=df.shape[1],
            columns=list(getattr(mining, "columns", None) or []),
            associations=assocs,
            suggestions=list(getattr(mining, "suggestions", None) or []),
            kpis=kpis,
            join_log=join_log,
            datamine=datamine_payload,
            slm_directives=directives.to_dict(),
            notes=list(dict.fromkeys(notes + list(directives.notes))),
            warnings=warnings,
            source=label,
            backend=directives.backend,
            mining_backend="autocausal.mining",
        )
        self.frame = df
        self.report = report

        # Sync onto incoming AutoCausal if present
        if ac_in is not None:
            try:
                ac_in._df = df
                ac_in.mining = mining
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


def auto_mine(
    source: Any,
    *,
    use_slm: Optional[bool] = None,
    **kwargs: Any,
) -> MineReport:
    suite = AutoMineSuite(source, use_slm=use_slm, **kwargs).run()
    assert suite.report is not None
    return suite.report
