"""Public multi-source causal mining: join → mine → impute → discover → report.

Exploratory only — correlation is not causation. Edges are candidate relationships
from heuristic discovery, not identified causal effects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union

import pandas as pd

from autocausal.public_suite import (
    get_public,
    join_public_corpus,
    list_public,
)


__all__ = [
    "PublicCausalReport",
    "PublicCausalMiner",
    "mine_public",
]


CAVEATS = [
    "Exploratory heuristics only — edges are candidate relationships, not identified causal effects.",
    "Correlation is not causation; public joins can introduce selection bias and ecological fallacy.",
    "Bundled fixtures are synthetic MIT demos, not real market/PII/clinical data.",
    "Optional IV (causaliv / 2SLS lite) is soft and does not validate exclusion restrictions.",
    "Network downloads soft-fail offline; prefer bundled sources for reproducible runs.",
]


@dataclass
class PublicCausalReport:
    """End-to-end public corpus mining + causal discovery report."""

    sources: list[str]
    source_meta: list[dict[str, Any]] = field(default_factory=list)
    join_log: list[dict[str, Any]] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    mining: Optional[dict[str, Any]] = None
    discovery: Optional[dict[str, Any]] = None
    edges: list[dict[str, Any]] = field(default_factory=list)
    candidates: dict[str, list[str]] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    iv_notes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=lambda: list(CAVEATS))
    source: str = "public_corpus"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# Public causal mining report",
            "",
            f"**Corpus:** `{self.source}` | {self.n_rows} rows x {self.n_cols} cols",
            "",
            "> Exploratory DAG candidates only -- not identified causal effects. "
            "Correlation is not causation.",
            "",
            "## Public sources",
            "",
        ]
        for s in self.source_meta:
            lines.append(
                f"- `{s.get('id')}` ({s.get('access')}, {s.get('domain')}): "
                f"{s.get('name')} — {s.get('license_note', '')}"
            )
        if not self.source_meta:
            for sid in self.sources:
                lines.append(f"- `{sid}`")
        lines.append("")

        if self.join_log:
            lines.append("## Joins")
            lines.append("")
            for j in self.join_log:
                if j.get("ok"):
                    lines.append(
                        f"- `{j.get('id')}` on `{j.get('keys')}` "
                        f"({j.get('how')}) -> {j.get('rows')} rows, "
                        f"+{j.get('cols_added', 0)} cols"
                    )
                else:
                    lines.append(f"- `{j.get('id')}` **failed**: {j.get('error')}")
            lines.append("")

        if self.mining:
            lines.append("## Mining")
            lines.append("")
            lines.append(f"- Associations: {len(self.mining.get('associations') or [])}")
            kpis = self.mining.get("kpis") or []
            if kpis:
                lines.append("- KPIs: " + ", ".join(f"`{k}`" for k in kpis[:12]))
            lines.append("")

        if self.candidates:
            lines.append("## Role candidates (X/Y/Z/W)")
            lines.append("")
            for kind, cols in self.candidates.items():
                joined = ", ".join(f"`{c}`" for c in cols) if cols else "-"
                lines.append(f"- **{kind}:** {joined}")
            lines.append("")

        lines.append("## Causal edges (exploratory)")
        lines.append("")
        if not self.edges:
            lines.append("_No edges above threshold._")
        else:
            lines.append("| source | target | type | score | confidence | p-value |")
            lines.append("|---|---|---|---:|---:|---:|")
            for e in self.edges:
                lines.append(
                    f"| `{e.get('source')}` | `{e.get('target')}` | "
                    f"{e.get('type', '')} | {e.get('score', '')} | "
                    f"{e.get('confidence', '')} | {e.get('pvalue', '')} |"
                )
        lines.append("")

        if self.iv_notes:
            lines.append("## IV / validation notes")
            lines.append("")
            for n in self.iv_notes:
                lines.append(f"- {n}")
            lines.append("")

        if self.notes:
            lines.append("## Pipeline notes")
            lines.append("")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")

        lines.append("## Caveats")
        lines.append("")
        for c in self.caveats:
            lines.append(f"- {c}")
        lines.append("")
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


class PublicCausalMiner:
    """Facade: multi-source public join → mine → impute → discover → report."""

    def __init__(
        self,
        sources: Optional[list[str]] = None,
        *,
        join_on: Optional[Union[str, list[str]]] = None,
        how: str = "outer",
        allow_network: bool = False,
        base: Optional[pd.DataFrame] = None,
        base_source: str = "user",
    ) -> None:
        if sources is None:
            sources = [s.id for s in list_public(offline_only=True) if s.access == "bundled"]
        self.sources = list(sources)
        self.join_on = join_on
        self.how = how
        self.allow_network = allow_network
        self.base = base
        self.base_source = base_source
        self.df: Optional[pd.DataFrame] = None
        self.join_log: list[dict[str, Any]] = []
        self.report: Optional[PublicCausalReport] = None
        self._ac: Any = None

    def load_join(self) -> pd.DataFrame:
        """Load and align public sources into one frame."""
        df, log = join_public_corpus(
            self.sources,
            on=self.join_on,
            how=self.how,
            allow_network=self.allow_network,
            base=self.base,
            base_label=self.base_source if self.base is not None else None,
        )
        self.df = df
        self.join_log = log
        return df

    def run(
        self,
        *,
        discover: bool = True,
        use_iv: bool = True,
        min_score: float = 0.15,
        min_abs_corr: float = 0.12,
        alpha: float = 0.05,
        impute_method: str = "auto",
        validate: bool = False,
    ) -> PublicCausalReport:
        """Join → mine → impute → discover (optional) → PublicCausalReport."""
        from autocausal.api import AutoCausal

        notes: list[str] = []
        if self.df is None:
            self.load_join()
        assert self.df is not None

        ok_ids = [j["id"] for j in self.join_log if j.get("ok")]
        fail = [j for j in self.join_log if not j.get("ok")]
        if fail:
            notes.append(f"soft-failed sources: {[f.get('id') for f in fail]}")
        if self.df.empty or len(self.df.columns) < 2:
            report = PublicCausalReport(
                sources=self.sources,
                join_log=list(self.join_log),
                n_rows=len(self.df),
                n_cols=len(self.df.columns),
                notes=notes + ["corpus too small for mining"],
                source="public:" + ",".join(ok_ids or self.sources),
            )
            self.report = report
            return report

        meta: list[dict[str, Any]] = []
        for sid in ok_ids or self.sources:
            try:
                meta.append(get_public(sid).to_dict())
            except KeyError:
                meta.append({"id": sid, "name": sid})

        ac = AutoCausal.from_dataframe(
            self.df,
            source="public:" + ",".join(ok_ids or self.sources),
        )
        ac.join_log = list(self.join_log)
        ac.mine(min_score=min_score)
        mining = ac.mining
        notes.append(f"mined {len(mining.associations) if mining else 0} associations")

        ac.impute(method=impute_method)  # type: ignore[arg-type]
        edges: list[dict[str, Any]] = []
        candidates: dict[str, list[str]] = {}
        roles: dict[str, str] = {}
        discovery_dict: Optional[dict[str, Any]] = None
        iv_notes: list[str] = []

        if discover:
            result = ac.discover(
                alpha=alpha,
                min_abs_corr=min_abs_corr,
                use_iv=use_iv,
            )
            edges = list(result.edges)
            candidates = dict(result.candidates)
            roles = {
                k: (v.value if hasattr(v, "value") else str(v))
                for k, v in result.roles.items()
            }
            discovery_dict = result.to_dict()
            notes.extend(list(result.notes or []))
            # surface IV-related edge notes
            for e in edges:
                if e.get("type") in ("iv", "iv_candidate", "instrument"):
                    iv_notes.append(
                        f"IV-ish edge {e.get('source')}→{e.get('target')} "
                        f"(score={e.get('score')})"
                    )
            if use_iv and not iv_notes:
                iv_notes.append(
                    "IV enabled (soft causaliv / 2SLS lite); no IV edges above threshold."
                )

        if validate and discover and ac.result is not None:
            # light consistency: edges should appear among top associations when numeric
            assoc = {
                frozenset((a["a"], a["b"]))
                for a in (mining.associations if mining else [])
            }
            supported = 0
            for e in edges:
                key = frozenset((e.get("source"), e.get("target")))
                if key in assoc:
                    supported += 1
            notes.append(
                f"validate: {supported}/{len(edges)} edges also in mining associations"
            )

        report = PublicCausalReport(
            sources=list(self.sources),
            source_meta=meta,
            join_log=list(self.join_log),
            n_rows=len(ac.df),
            n_cols=len(ac.df.columns),
            mining=mining.to_dict() if mining is not None else None,
            discovery=discovery_dict,
            edges=edges,
            candidates=candidates,
            roles=roles,
            iv_notes=iv_notes,
            notes=notes,
            source=ac.source,
        )
        self._ac = ac
        self.df = ac.df
        self.report = report
        return report


def mine_public(
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
) -> PublicCausalReport:
    """Convenience: PublicCausalMiner(...).run(...)."""
    if isinstance(sources, str):
        sources = [x.strip() for x in sources.split(",") if x.strip()]
    miner = PublicCausalMiner(
        sources,
        join_on=join_on,
        how=how,
        allow_network=allow_network,
        base=base,
    )
    return miner.run(
        discover=discover,
        use_iv=use_iv,
        min_score=min_score,
        min_abs_corr=min_abs_corr,
        validate=validate,
    )
