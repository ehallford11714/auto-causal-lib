"""Auto data mining: column profiling, associations, KPI-like suggestions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import log, sqrt
from typing import Any, Optional

import numpy as np
import pandas as pd

from autocausal.roles import ColumnRole, infer_column_roles


__all__ = [
    "MiningReport",
    "profile_dataframe",
    "association_matrix",
    "suggest_relationships",
    "mine",
]


@dataclass
class MiningReport:
    columns: list[dict[str, Any]]
    associations: list[dict[str, Any]]
    suggestions: list[dict[str, Any]]
    kpis: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = ["# AutoCausal mining report", ""]
        lines.append(f"**Columns profiled:** {len(self.columns)}")
        lines.append(f"**Associations:** {len(self.associations)}")
        lines.append("")
        if self.kpis:
            lines.append("## Suggested KPIs")
            lines.append("")
            for k in self.kpis:
                lines.append(f"- `{k}`")
            lines.append("")
        lines.append("## Column profiles")
        lines.append("")
        lines.append("| column | dtype | null% | unique | skew | top |")
        lines.append("|---|---|---:|---:|---:|---|")
        for c in self.columns:
            top = ", ".join(str(x) for x in (c.get("top_values") or [])[:3])
            skew = c.get("skew")
            skew_s = f"{skew:.3f}" if isinstance(skew, (int, float)) and skew == skew else "—"
            lines.append(
                f"| `{c['name']}` | {c.get('dtype', '')} | {c.get('null_pct', 0):.1f} | "
                f"{c.get('nunique', '')} | {skew_s} | {top or '—'} |"
            )
        lines.append("")
        lines.append("## Top associations")
        lines.append("")
        if not self.associations:
            lines.append("_None above threshold._")
        else:
            lines.append("| a | b | metric | score |")
            lines.append("|---|---|---|---:|")
            for a in self.associations[:25]:
                lines.append(
                    f"| `{a['a']}` | `{a['b']}` | {a.get('metric', '')} | {a.get('score', '')} |"
                )
        lines.append("")
        lines.append("## Suggested relationships")
        lines.append("")
        if not self.suggestions:
            lines.append("_No suggestions._")
        else:
            for s in self.suggestions[:20]:
                lines.append(
                    f"- `{s.get('source')}` → `{s.get('target')}` "
                    f"({s.get('reason', '')}; score={s.get('score', '')})"
                )
        lines.append("")
        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")
        return "\n".join(lines)


_KPI_HINTS = (
    "revenue",
    "sales",
    "conversion",
    "churn",
    "retention",
    "ltv",
    "arpu",
    "ctr",
    "cpa",
    "roi",
    "profit",
    "margin",
    "outcome",
    "y",
    "target",
    "score",
    "nps",
    "engagement",
)


def profile_dataframe(df: pd.DataFrame, *, top_k: int = 5) -> dict[str, Any]:
    roles = infer_column_roles(df)
    columns: list[dict[str, Any]] = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        null_pct = float(s.isna().mean() * 100.0)
        nunique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        skew: Optional[float] = None
        top_values: list[Any] = []
        role = roles.get(col, ColumnRole.UNKNOWN)
        if pd.api.types.is_numeric_dtype(s) and role == ColumnRole.NUMERIC:
            clean = pd.to_numeric(s, errors="coerce").dropna()
            if len(clean) >= 3:
                skew = float(clean.skew())
            vc = clean.round(4).value_counts().head(top_k)
            top_values = [float(v) if isinstance(v, (np.floating, float)) else v for v in vc.index.tolist()]
        else:
            vc = s.astype(str).value_counts(dropna=True).head(top_k)
            top_values = vc.index.tolist()
        columns.append(
            {
                "name": str(col),
                "dtype": dtype,
                "role": role.value if hasattr(role, "value") else str(role),
                "null_pct": round(null_pct, 2),
                "nunique": nunique,
                "skew": round(skew, 4) if skew is not None and skew == skew else None,
                "top_values": top_values,
            }
        )
    return {"columns": columns, "n_rows": len(df), "n_cols": len(df.columns)}


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V (bias-corrected lite) for categorical association."""
    tbl = pd.crosstab(x.astype(str), y.astype(str))
    if tbl.size == 0 or tbl.shape[0] < 2 or tbl.shape[1] < 2:
        return 0.0
    n = tbl.to_numpy().sum()
    if n < 2:
        return 0.0
    # chi-square
    expected = np.outer(tbl.sum(axis=1), tbl.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((tbl.to_numpy() - expected) ** 2 / np.where(expected == 0, np.nan, expected))
    r, k = tbl.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
    kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return float(sqrt(phi2corr / denom))


def _mutual_info_discrete(x: pd.Series, y: pd.Series) -> float:
    """Simple discrete mutual information (nats → normalized-ish score)."""
    a = x.astype(str)
    b = y.astype(str)
    n = len(a)
    if n == 0:
        return 0.0
    joint = pd.crosstab(a, b)
    px = joint.sum(axis=1).to_numpy(dtype=float) / n
    py = joint.sum(axis=0).to_numpy(dtype=float) / n
    pxy = joint.to_numpy(dtype=float) / n
    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            p = pxy[i, j]
            if p <= 0:
                continue
            mi += p * log(p / (px[i] * py[j] + 1e-15) + 1e-15)
    # normalize roughly by mean entropy
    hx = -sum(p * log(p + 1e-15) for p in px if p > 0)
    hy = -sum(p * log(p + 1e-15) for p in py if p > 0)
    denom = max((hx + hy) / 2.0, 1e-12)
    return float(min(1.0, mi / denom))


def association_matrix(
    df: pd.DataFrame,
    *,
    roles: Optional[dict[str, ColumnRole]] = None,
    min_score: float = 0.15,
    max_pairs: int = 200,
) -> list[dict[str, Any]]:
    roles = roles or infer_column_roles(df)
    cols = [c for c in df.columns if roles.get(c) != ColumnRole.ID]
    assocs: list[dict[str, Any]] = []
    for a, b in combinations(cols, 2):
        ra, rb = roles.get(a), roles.get(b)
        score = 0.0
        metric = "none"
        if ra == ColumnRole.NUMERIC and rb == ColumnRole.NUMERIC:
            sa = pd.to_numeric(df[a], errors="coerce")
            sb = pd.to_numeric(df[b], errors="coerce")
            mask = sa.notna() & sb.notna()
            if mask.sum() >= 5:
                corr = float(sa[mask].corr(sb[mask]))
                if corr == corr:
                    score = abs(corr)
                    metric = "pearson"
        elif ra in (ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN) and rb in (
            ColumnRole.CATEGORICAL,
            ColumnRole.BOOLEAN,
        ):
            score = _cramers_v(df[a], df[b])
            metric = "cramers_v"
            # also MI as secondary
            mi = _mutual_info_discrete(df[a], df[b])
            if mi > score:
                score, metric = mi, "mutual_info"
        else:
            # mixed: bin numeric lightly and use Cramér / MI
            xa, xb = df[a], df[b]
            if ra == ColumnRole.NUMERIC:
                xa = pd.qcut(pd.to_numeric(df[a], errors="coerce"), q=4, duplicates="drop").astype(str)
            else:
                xa = df[a].astype(str)
            if rb == ColumnRole.NUMERIC:
                xb = pd.qcut(pd.to_numeric(df[b], errors="coerce"), q=4, duplicates="drop").astype(str)
            else:
                xb = df[b].astype(str)
            score = max(_cramers_v(xa, xb), _mutual_info_discrete(xa, xb))
            metric = "mixed_assoc"
        if score >= min_score:
            assocs.append(
                {
                    "a": str(a),
                    "b": str(b),
                    "metric": metric,
                    "score": round(float(score), 4),
                }
            )
    assocs.sort(key=lambda x: x["score"], reverse=True)
    return assocs[:max_pairs]


def suggest_relationships(
    df: pd.DataFrame,
    associations: list[dict[str, Any]],
    *,
    roles: Optional[dict[str, ColumnRole]] = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    roles = roles or infer_column_roles(df)
    kpis = [
        c
        for c in df.columns
        if any(h in str(c).lower() for h in _KPI_HINTS)
        and roles.get(c) in (ColumnRole.NUMERIC, ColumnRole.BOOLEAN, ColumnRole.CATEGORICAL)
    ]
    if not kpis:
        # fallback: high-variance numeric
        nums = [c for c, r in roles.items() if r == ColumnRole.NUMERIC]
        var_rank = sorted(
            nums,
            key=lambda c: float(pd.to_numeric(df[c], errors="coerce").var() or 0.0),
            reverse=True,
        )
        kpis = var_rank[:3]

    suggestions: list[dict[str, Any]] = []
    kpi_set = set(kpis)
    for assoc in associations:
        a, b = assoc["a"], assoc["b"]
        if a in kpi_set or b in kpi_set:
            target = a if a in kpi_set else b
            source = b if target == a else a
            suggestions.append(
                {
                    "source": source,
                    "target": target,
                    "score": assoc["score"],
                    "reason": f"associated with KPI-like `{target}` via {assoc['metric']}",
                }
            )
        else:
            # interesting high-score pairs
            if assoc["score"] >= 0.35:
                suggestions.append(
                    {
                        "source": a,
                        "target": b,
                        "score": assoc["score"],
                        "reason": f"strong {assoc['metric']} association",
                    }
                )
    suggestions.sort(key=lambda s: s["score"], reverse=True)
    # dedupe
    seen: set[tuple[str, str]] = set()
    uniq: list[dict[str, Any]] = []
    for s in suggestions:
        key = (s["source"], s["target"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq[:30], [str(k) for k in kpis]


def mine(
    df: pd.DataFrame,
    *,
    min_score: float = 0.15,
) -> MiningReport:
    roles = infer_column_roles(df)
    profile = profile_dataframe(df)
    assocs = association_matrix(df, roles=roles, min_score=min_score)
    suggestions, kpis = suggest_relationships(df, assocs, roles=roles)
    notes = [
        "Mining is exploratory association analysis, not causal identification.",
        "Feed suggestions into impute() → discover() for DAG candidates.",
    ]
    return MiningReport(
        columns=profile["columns"],
        associations=assocs,
        suggestions=suggestions,
        kpis=kpis,
        notes=notes,
    )
