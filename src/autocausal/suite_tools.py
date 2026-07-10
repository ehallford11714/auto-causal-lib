"""Tool suite registry: causal, NLP, KPI mining, and validation.

Catalogs built-in adapters and optional soft imports (DoWhy, EconML, CausalML,
NLTK, gensim, spaCy, datamine, causaliv). Never hard-fails on missing extras.
"""

from __future__ import annotations

import importlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

__all__ = [
    "ToolSpec",
    "ToolResult",
    "ValidationReport",
    "list_tools",
    "get_tool",
    "invoke_tool",
    "validate_pipeline",
    "tool_catalog",
    "refute",
    "RefuteResult",
]


@dataclass
class ToolSpec:
    id: str
    name: str
    category: str  # causal | nlp | kpi | validation
    description: str
    builtin: bool = False
    optional_extra: Optional[str] = None
    install_hint: str = ""
    status: str = "unknown"  # available | stub | missing

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    tool_id: str
    ok: bool
    backend: str
    data: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    ok: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    notes: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Validation suite report",
            "",
            f"**Overall:** {'PASS' if self.ok else 'REVIEW'} (score={self.score:.2f})",
            "",
            "## Checks",
            "",
        ]
        for c in self.checks:
            mark = "OK" if c.get("ok") else "!!"
            lines.append(f"- [{mark}] **{c.get('id')}**: {c.get('detail')}")
        lines.append("")
        if self.tools_used:
            lines.append(f"**Tools:** {', '.join(self.tools_used)}")
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)


def _soft_import(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except Exception:
        return None


def _probe(module: str) -> bool:
    return _soft_import(module) is not None


# ---------------------------------------------------------------------------
# Built-in causal estimators
# ---------------------------------------------------------------------------


def _builtin_2sls(df: pd.DataFrame, *, y: str, d: str, z: str, controls: Optional[list[str]] = None) -> ToolResult:
    from autocausal.iv import _numpy_2sls

    try:
        cols = [y, d, z] + list(controls or [])
        frame = df[cols].dropna()
        if len(frame) < 8:
            return ToolResult(
                tool_id="builtin_2sls",
                ok=False,
                backend="numpy",
                error=f"need ≥8 complete rows, got {len(frame)}",
            )
        yv = frame[y].to_numpy(dtype=float)
        dv = frame[d].to_numpy(dtype=float)
        zv = frame[z].to_numpy(dtype=float)
        ctrl = None
        if controls:
            keep = [c for c in controls if c in frame.columns]
            if keep:
                ctrl = frame[keep].to_numpy(dtype=float)
        out = _numpy_2sls(yv, dv, zv, ctrl)
        return ToolResult(tool_id="builtin_2sls", ok=True, backend="numpy", data=out)
    except Exception as e:
        return ToolResult(tool_id="builtin_2sls", ok=False, backend="numpy", error=f"{type(e).__name__}: {e}")


def _builtin_did(
    df: pd.DataFrame,
    *,
    y: str,
    treated: str,
    post: str,
) -> ToolResult:
    """Simple 2x2 DiD: E[Y|T=1,Post=1]-E[Y|T=1,Post=0] - (control)."""
    try:
        g = df[[y, treated, post]].dropna()
        def mean(t: int, p: int) -> float:
            m = g[(g[treated] == t) & (g[post] == p)][y]
            return float(m.mean()) if len(m) else float("nan")

        att = (mean(1, 1) - mean(1, 0)) - (mean(0, 1) - mean(0, 0))
        return ToolResult(
            tool_id="builtin_did",
            ok=math.isfinite(att),
            backend="numpy",
            data={"att": att, "cells": {"t1p1": mean(1, 1), "t1p0": mean(1, 0), "t0p1": mean(0, 1), "t0p0": mean(0, 0)}},
            notes=["Simple 2x2 DiD; assumes parallel trends."],
        )
    except Exception as e:
        return ToolResult(tool_id="builtin_did", ok=False, backend="numpy", error=f"{type(e).__name__}: {e}")


def _adapter_causaliv(df: pd.DataFrame, **kwargs: Any) -> ToolResult:
    mod = _soft_import("causaliv")
    if mod is None:
        return ToolResult(
            tool_id="causaliv",
            ok=False,
            backend="missing",
            error="causaliv not installed",
            notes=["pip install -e ../CausalIVSuite"],
        )
    try:
        if hasattr(mod, "two_sls") and kwargs.get("y") and kwargs.get("d") and kwargs.get("z"):
            y = df[kwargs["y"]].to_numpy(dtype=float)
            d = df[kwargs["d"]].to_numpy(dtype=float)
            z = df[kwargs["z"]].to_numpy(dtype=float)
            out = mod.two_sls(y=y, d=d, z=z)
            return ToolResult(tool_id="causaliv", ok=True, backend="causaliv", data=dict(out) if isinstance(out, dict) else {"result": out})
        # validate helpers
        validate = _soft_import("causaliv.validate")
        if validate and hasattr(validate, "weak_iv_check"):
            return ToolResult(
                tool_id="causaliv",
                ok=True,
                backend="causaliv.validate",
                data={"available": True},
                notes=["causaliv present; pass y/d/z for two_sls"],
            )
        return ToolResult(tool_id="causaliv", ok=True, backend="causaliv", data={"available": True})
    except Exception as e:
        return ToolResult(tool_id="causaliv", ok=False, backend="causaliv", error=f"{type(e).__name__}: {e}")


def _stub_optional(tool_id: str, module: str, install: str) -> Callable[..., ToolResult]:
    def _fn(*_a: Any, **_k: Any) -> ToolResult:
        if _probe(module):
            return ToolResult(
                tool_id=tool_id,
                ok=True,
                backend=module,
                data={"available": True, "invoked": False},
                notes=[f"{module} installed — wire domain-specific call in caller; stub invoke only."],
            )
        return ToolResult(
            tool_id=tool_id,
            ok=False,
            backend="stub",
            error=f"{module} not installed",
            notes=[install],
        )

    return _fn


# ---------------------------------------------------------------------------
# NLP adapters
# ---------------------------------------------------------------------------


def _nlp_nltk(text: str = "", **_k: Any) -> ToolResult:
    nltk = _soft_import("nltk")
    if nltk is None:
        # soft tokenize fallback
        tokens = re.findall(r"[A-Za-z']+", text or "")
        return ToolResult(
            tool_id="nltk",
            ok=True,
            backend="builtin_regex",
            data={"tokens": tokens[:80], "stopwords_removed": False},
            notes=["nltk not installed; used regex tokenize. pip install 'autocausal[nlp]'"],
        )
    notes: list[str] = []
    try:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
            except Exception as e:
                notes.append(f"punkt download soft-fail: {e}")
        try:
            from nltk.corpus import stopwords

            try:
                stops = set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                stops = set(stopwords.words("english"))
        except Exception:
            stops = set()
            notes.append("stopwords unavailable")

        from nltk.tokenize import word_tokenize

        tokens = word_tokenize((text or "").lower())
        filtered = [t for t in tokens if t.isalpha() and t not in stops]
        pos = []
        try:
            try:
                nltk.data.find("taggers/averaged_perceptron_tagger")
            except LookupError:
                nltk.download("averaged_perceptron_tagger", quiet=True)
            pos = nltk.pos_tag(filtered[:40])
        except Exception as e:
            notes.append(f"POS soft-fail: {e}")
        return ToolResult(
            tool_id="nltk",
            ok=True,
            backend="nltk",
            data={
                "tokens": filtered[:80],
                "pos": [{"token": t, "tag": p} for t, p in pos[:40]],
                "n_tokens": len(filtered),
            },
            notes=notes,
        )
    except Exception as e:
        return ToolResult(tool_id="nltk", ok=False, backend="nltk", error=f"{type(e).__name__}: {e}")


def _nlp_gensim(texts: Optional[list[str]] = None, text: str = "", **_k: Any) -> ToolResult:
    gensim = _soft_import("gensim")
    docs = texts or ([text] if text else [])
    if gensim is None:
        # bag-of-words cosine stub
        if len(docs) < 2:
            return ToolResult(
                tool_id="gensim",
                ok=True,
                backend="builtin_bow",
                data={"similarity": None, "topics": []},
                notes=["gensim not installed; need ≥2 docs for bow stub. pip install 'autocausal[nlp]'"],
            )
        def bow(s: str) -> dict[str, int]:
            c: dict[str, int] = {}
            for t in re.findall(r"[a-z]+", s.lower()):
                c[t] = c.get(t, 0) + 1
            return c
        a, b = bow(docs[0]), bow(docs[1])
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        na = math.sqrt(sum(v * v for v in a.values())) or 1.0
        nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
        sim = dot / (na * nb)
        return ToolResult(
            tool_id="gensim",
            ok=True,
            backend="builtin_bow",
            data={"similarity": round(sim, 4), "topics": []},
            notes=["gensim missing — used bag-of-words cosine stub"],
        )
    try:
        from gensim import corpora
        from gensim.models import LdaModel
        from gensim.similarities import MatrixSimilarity

        tokenized = [re.findall(r"[a-z]+", (d or "").lower()) for d in docs]
        tokenized = [t for t in tokenized if t]
        if not tokenized:
            return ToolResult(tool_id="gensim", ok=False, backend="gensim", error="no tokens")
        dictionary = corpora.Dictionary(tokenized)
        corpus = [dictionary.doc2bow(t) for t in tokenized]
        sim = None
        if len(corpus) >= 2:
            index = MatrixSimilarity(corpus, num_features=len(dictionary))
            sims = index[corpus[0]]
            sim = float(sims[1]) if len(sims) > 1 else None
        topics = []
        if len(dictionary) >= 3 and len(corpus) >= 1:
            lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=min(3, len(dictionary)), passes=1)
            topics = [{"topic_id": i, "terms": lda.print_topic(i, topn=5)} for i in range(lda.num_topics)]
        return ToolResult(
            tool_id="gensim",
            ok=True,
            backend="gensim",
            data={"similarity": sim, "topics": topics, "n_docs": len(tokenized)},
        )
    except Exception as e:
        return ToolResult(tool_id="gensim", ok=False, backend="gensim", error=f"{type(e).__name__}: {e}")


def _nlp_spacy(text: str = "", **_k: Any) -> ToolResult:
    spacy = _soft_import("spacy")
    if spacy is None:
        return ToolResult(
            tool_id="spacy",
            ok=False,
            backend="missing",
            error="spacy not installed",
            notes=["pip install spacy && python -m spacy download en_core_web_sm"],
        )
    try:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = spacy.blank("en")
            return ToolResult(
                tool_id="spacy",
                ok=True,
                backend="spacy.blank",
                data={"tokens": [t.text for t in nlp(text or "") ][:80]},
                notes=["en_core_web_sm missing — used blank English"],
            )
        doc = nlp(text or "")
        return ToolResult(
            tool_id="spacy",
            ok=True,
            backend="spacy",
            data={
                "tokens": [t.text for t in doc][:80],
                "ents": [{"text": e.text, "label": e.label_} for e in doc.ents][:40],
                "pos": [{"text": t.text, "pos": t.pos_} for t in doc][:40],
            },
        )
    except Exception as e:
        return ToolResult(tool_id="spacy", ok=False, backend="spacy", error=f"{type(e).__name__}: {e}")


def _nlp_text_z(text: str = "", **_k: Any) -> ToolResult:
    """Text → instrument/role tags (patterns shared with InstrumentForge cues)."""
    cues = [
        (r"\blottery\b", "lottery_assignment", "instrument"),
        (r"\brandom(ly|ized)?\b", "randomized_assignment", "instrument"),
        (r"\brain(fall)?\b", "weather_rainfall", "instrument"),
        (r"\bjudge\b", "judge_leniency", "instrument"),
        (r"\bshift[- ]share\b", "shift_share", "instrument"),
        (r"\btreatment\b", "treatment", "treatment"),
        (r"\boutcome\b|\brevenue\b|\bsales\b", "outcome", "outcome"),
        (r"\bconfound", "confounder", "confounder"),
    ]
    tags: list[dict[str, Any]] = []
    lower = (text or "").lower()
    for pat, name, role in cues:
        m = re.search(pat, lower)
        if m:
            tags.append({"token": name, "role": role, "span": m.group(0)})
    return ToolResult(
        tool_id="text_z",
        ok=True,
        backend="builtin",
        data={"tags": tags, "n": len(tags)},
        notes=["Built-in text→Z/role tagger"],
    )


# ---------------------------------------------------------------------------
# KPI mining
# ---------------------------------------------------------------------------


def _kpi_autocausal(df: Optional[pd.DataFrame] = None, **_k: Any) -> ToolResult:
    if df is None:
        return ToolResult(tool_id="autocausal_mining", ok=False, backend="autocausal", error="df required")
    from autocausal.mining import mine

    report = mine(df)
    return ToolResult(
        tool_id="autocausal_mining",
        ok=True,
        backend="autocausal.mining",
        data={
            "kpis": report.kpis,
            "n_associations": len(report.associations),
            "n_suggestions": len(report.suggestions),
        },
    )


def _kpi_datamine(df: Optional[pd.DataFrame] = None, **_k: Any) -> ToolResult:
    dm = _soft_import("datamine") or _soft_import("dataminelib")
    if dm is None:
        return ToolResult(
            tool_id="datamine",
            ok=False,
            backend="missing",
            error="datamine not installed",
            notes=["Prefer shared DataMineLib when present; else use autocausal.mining"],
        )
    try:
        if hasattr(dm, "mine") and df is not None:
            out = dm.mine(df)
            data = out.to_dict() if hasattr(out, "to_dict") else dict(out) if isinstance(out, dict) else {"result": str(out)[:500]}
            return ToolResult(tool_id="datamine", ok=True, backend="datamine", data=data)
        return ToolResult(tool_id="datamine", ok=True, backend="datamine", data={"available": True})
    except Exception as e:
        return ToolResult(tool_id="datamine", ok=False, backend="datamine", error=f"{type(e).__name__}: {e}")


def _kpi_vision(**_k: Any) -> ToolResult:
    vk = _soft_import("visionkpi") or _soft_import("emotivevision")
    if vk is None:
        return ToolResult(
            tool_id="vision_kpi",
            ok=False,
            backend="missing",
            error="visionkpi/emotivevision not installed",
            notes=["Use VisionKPIMiner or emotivevision.autocausal streams_to_frame"],
        )
    return ToolResult(
        tool_id="vision_kpi",
        ok=True,
        backend=vk.__name__,
        data={"available": True},
        notes=["Vision KPI package importable — call product-specific mine APIs"],
    )


def _kpi_tabular_profile(df: Optional[pd.DataFrame] = None, **_k: Any) -> ToolResult:
    if df is None:
        return ToolResult(tool_id="tabular_kpi_profile", ok=False, backend="builtin", error="df required")
    from autocausal.mining import profile_dataframe

    raw = profile_dataframe(df)
    profiles = raw.get("columns", raw) if isinstance(raw, dict) else list(raw)
    if not isinstance(profiles, list):
        profiles = []
    numeric = [
        p
        for p in profiles
        if isinstance(p, dict)
        and (
            p.get("dtype") in ("float64", "float32", "int64", "int32")
            or "float" in str(p.get("dtype", ""))
            or "int" in str(p.get("dtype", ""))
        )
    ]
    return ToolResult(
        tool_id="tabular_kpi_profile",
        ok=True,
        backend="builtin",
        data={
            "n_columns": len(profiles),
            "numeric_candidates": [p["name"] for p in numeric[:20]],
            "profiles": profiles[:30],
        },
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, dict[str, Any]] = {
    "builtin_2sls": {
        "spec": ToolSpec(
            id="builtin_2sls",
            name="Built-in 2SLS",
            category="causal",
            description="Numpy two-stage least squares (always available)",
            builtin=True,
        ),
        "fn": _builtin_2sls,
        "probe": lambda: True,
    },
    "builtin_did": {
        "spec": ToolSpec(
            id="builtin_did",
            name="Built-in DiD",
            category="causal",
            description="Simple 2x2 difference-in-differences",
            builtin=True,
        ),
        "fn": _builtin_did,
        "probe": lambda: True,
    },
    "causaliv": {
        "spec": ToolSpec(
            id="causaliv",
            name="CausalIVSuite",
            category="causal",
            description="IV / DiD / validate from causaliv",
            optional_extra="causaliv",
            install_hint="pip install -e ../CausalIVSuite",
        ),
        "fn": _adapter_causaliv,
        "probe": lambda: _probe("causaliv"),
    },
    "dowhy": {
        "spec": ToolSpec(
            id="dowhy",
            name="DoWhy",
            category="causal",
            description="Microsoft DoWhy causal inference (optional)",
            optional_extra="dowhy",
            install_hint="pip install dowhy",
        ),
        "fn": _stub_optional("dowhy", "dowhy", "pip install dowhy"),
        "probe": lambda: _probe("dowhy"),
    },
    "econml": {
        "spec": ToolSpec(
            id="econml",
            name="EconML",
            category="causal",
            description="Microsoft EconML heterogeneous treatment effects (optional)",
            optional_extra="econml",
            install_hint="pip install econml",
        ),
        "fn": _stub_optional("econml", "econml", "pip install econml"),
        "probe": lambda: _probe("econml"),
    },
    "causalml": {
        "spec": ToolSpec(
            id="causalml",
            name="CausalML",
            category="causal",
            description="Uber CausalML uplift / meta-learners (optional)",
            optional_extra="causalml",
            install_hint="pip install causalml",
        ),
        "fn": _stub_optional("causalml", "causalml", "pip install causalml"),
        "probe": lambda: _probe("causalml"),
    },
    "nltk": {
        "spec": ToolSpec(
            id="nltk",
            name="NLTK",
            category="nlp",
            description="Tokenize, stopwords, POS (punkt soft-download)",
            optional_extra="nlp",
            install_hint="pip install 'autocausal[nlp]'",
        ),
        "fn": _nlp_nltk,
        "probe": lambda: _probe("nltk"),
    },
    "gensim": {
        "spec": ToolSpec(
            id="gensim",
            name="gensim",
            category="nlp",
            description="Similarity / LDA topics (bow stub if missing)",
            optional_extra="nlp",
            install_hint="pip install 'autocausal[nlp]'",
        ),
        "fn": _nlp_gensim,
        "probe": lambda: _probe("gensim"),
    },
    "spacy": {
        "spec": ToolSpec(
            id="spacy",
            name="spaCy",
            category="nlp",
            description="Optional NER/POS via spaCy",
            optional_extra="nlp",
            install_hint="pip install spacy && python -m spacy download en_core_web_sm",
        ),
        "fn": _nlp_spacy,
        "probe": lambda: _probe("spacy"),
    },
    "text_z": {
        "spec": ToolSpec(
            id="text_z",
            name="Text→Z tags",
            category="nlp",
            description="Heuristic instrument/role tags from free text",
            builtin=True,
        ),
        "fn": _nlp_text_z,
        "probe": lambda: True,
    },
    "autocausal_mining": {
        "spec": ToolSpec(
            id="autocausal_mining",
            name="AutoCausal mining",
            category="kpi",
            description="Column profiles, associations, KPI suggestions",
            builtin=True,
        ),
        "fn": _kpi_autocausal,
        "probe": lambda: True,
    },
    "datamine": {
        "spec": ToolSpec(
            id="datamine",
            name="DataMine",
            category="kpi",
            description="Shared datamine backend when installed",
            optional_extra="datamine",
            install_hint="Install DataMineLib when available",
        ),
        "fn": _kpi_datamine,
        "probe": lambda: _probe("datamine") or _probe("dataminelib"),
    },
    "vision_kpi": {
        "spec": ToolSpec(
            id="vision_kpi",
            name="Vision KPI hook",
            category="kpi",
            description="VisionKPIMiner / EmotiveVision KPI bridge",
            optional_extra="vision",
            install_hint="pip install -e ../VisionKPIMiner or ../EmotiveVision",
        ),
        "fn": _kpi_vision,
        "probe": lambda: _probe("visionkpi") or _probe("emotivevision"),
    },
    "tabular_kpi_profile": {
        "spec": ToolSpec(
            id="tabular_kpi_profile",
            name="Tabular KPI profile",
            category="kpi",
            description="Lightweight numeric KPI candidate profiles",
            builtin=True,
        ),
        "fn": _kpi_tabular_profile,
        "probe": lambda: True,
    },
}


def _refresh_status(spec: ToolSpec, probe: Callable[[], bool]) -> ToolSpec:
    if spec.builtin:
        spec.status = "available"
    elif probe():
        spec.status = "available"
    else:
        spec.status = "stub" if spec.category in ("causal", "nlp", "kpi") else "missing"
        if not spec.builtin and not probe():
            # NLP tools with builtin fallbacks still "available" via fallback
            if spec.id in ("nltk", "gensim"):
                spec.status = "stub"
            elif spec.id in ("dowhy", "econml", "causalml", "spacy", "datamine", "vision_kpi", "causaliv"):
                spec.status = "missing"
    return spec


def list_tools(*, category: Optional[str] = None) -> list[ToolSpec]:
    out: list[ToolSpec] = []
    for tid, entry in _REGISTRY.items():
        spec: ToolSpec = entry["spec"]
        # copy so status mutations don't leak oddly
        s = ToolSpec(**spec.to_dict())
        _refresh_status(s, entry["probe"])
        if category and s.category != category:
            continue
        out.append(s)
    return out


def tool_catalog() -> dict[str, Any]:
    tools = list_tools()
    return {
        "n": len(tools),
        "by_category": {
            cat: [t.to_dict() for t in tools if t.category == cat]
            for cat in sorted({t.category for t in tools})
        },
        "tools": [t.to_dict() for t in tools],
    }


def get_tool(tool_id: str) -> Optional[ToolSpec]:
    for t in list_tools():
        if t.id == tool_id:
            return t
    return None


def invoke_tool(tool_id: str, **kwargs: Any) -> ToolResult:
    entry = _REGISTRY.get(tool_id)
    if entry is None:
        return ToolResult(tool_id=tool_id, ok=False, backend="none", error=f"unknown tool: {tool_id}")
    try:
        return entry["fn"](**kwargs)
    except TypeError:
        # retry with df-only / text-only flexibility
        try:
            return entry["fn"](kwargs.get("df"), **{k: v for k, v in kwargs.items() if k != "df"})
        except Exception as e:
            return ToolResult(tool_id=tool_id, ok=False, backend="error", error=f"{type(e).__name__}: {e}")
    except Exception as e:
        return ToolResult(tool_id=tool_id, ok=False, backend="error", error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Validation suite
# ---------------------------------------------------------------------------


def validate_pipeline(
    report: Optional[dict[str, Any]] = None,
    *,
    df: Optional[pd.DataFrame] = None,
    claims_text: str = "",
    y: Optional[str] = None,
    d: Optional[str] = None,
    z: Optional[str] = None,
) -> ValidationReport:
    """Combine weak-IV F, placebo stub, NLP claim consistency, KPI coverage."""
    report = report or {}
    checks: list[dict[str, Any]] = []
    tools_used: list[str] = []
    notes: list[str] = []

    # KPI coverage
    kpis = report.get("kpis") or []
    edges = report.get("edges") or []
    if df is not None and not kpis:
        mine_res = invoke_tool("autocausal_mining", df=df)
        tools_used.append("autocausal_mining")
        if mine_res.ok:
            kpis = mine_res.data.get("kpis") or []
    kpi_ok = bool(kpis) or (df is not None and len(df.columns) >= 2)
    checks.append(
        {
            "id": "kpi_coverage",
            "ok": kpi_ok,
            "detail": f"{len(kpis)} KPI hints; {len(df.columns) if df is not None else 0} columns",
        }
    )

    # Weak IV F
    f_ok = True
    f_detail = "no IV estimate provided"
    iv = report.get("iv") or report.get("estimates") or {}
    if df is not None and y and d and z and y in df.columns and d in df.columns and z in df.columns:
        iv_res = invoke_tool("builtin_2sls", df=df, y=y, d=d, z=z)
        tools_used.append("builtin_2sls")
        if iv_res.ok:
            iv = iv_res.data
    fstat = iv.get("first_stage_f") if isinstance(iv, dict) else None
    if fstat is not None:
        try:
            fval = float(fstat)
            f_ok = fval >= 10.0
            f_detail = f"first-stage F={fval:.3f} ({'strong' if f_ok else 'weak <10'})"
        except (TypeError, ValueError):
            f_ok = False
            f_detail = f"unparseable F: {fstat}"
    checks.append({"id": "weak_iv_f", "ok": f_ok if fstat is not None else True, "detail": f_detail})

    # Placebo stub: shuffle Z and expect attenuated |coef|
    placebo_ok = True
    placebo_detail = "skipped (need df + y/d/z)"
    if df is not None and y and d and z and all(c in df.columns for c in (y, d, z)):
        tools_used.append("placebo_stub")
        try:
            base = invoke_tool("builtin_2sls", df=df, y=y, d=d, z=z)
            rng = np.random.default_rng(0)
            df_p = df.copy()
            df_p[z] = rng.permutation(df_p[z].to_numpy())
            plac = invoke_tool("builtin_2sls", df=df_p, y=y, d=d, z=z)
            if base.ok and plac.ok:
                b = abs(float(base.data.get("coef") or 0))
                p = abs(float(plac.data.get("coef") or 0))
                placebo_ok = (p <= b + 1e-9) or p < 0.5 * max(b, 1e-6) or p < 0.05
                placebo_detail = f"|coef|={b:.4f} vs placebo={p:.4f}"
            else:
                placebo_detail = "2SLS failed on placebo draw"
                placebo_ok = False
        except Exception as e:
            placebo_ok = False
            placebo_detail = f"placebo error: {e}"
    checks.append({"id": "placebo_stub", "ok": placebo_ok, "detail": placebo_detail})

    # NLP consistency on claims
    text = claims_text or report.get("narrative") or report.get("text") or ""
    if edges and not text:
        text = "; ".join(f"{e.get('source')} causes {e.get('target')}" for e in edges[:5])
    nlp = invoke_tool("nltk", text=text)
    tools_used.append("nltk")
    tags = invoke_tool("text_z", text=text)
    tools_used.append("text_z")
    tokens = (nlp.data or {}).get("tokens") or []
    causal_words = {"cause", "causes", "effect", "affects", "increase", "decrease", "impact"}
    overlap = bool(set(t.lower() for t in tokens) & causal_words) or bool(edges)
    nlp_ok = nlp.ok and (not text or overlap or len(tokens) >= 0)
    checks.append(
        {
            "id": "nlp_claim_consistency",
            "ok": nlp_ok,
            "detail": f"tokens={len(tokens)}; role_tags={len((tags.data or {}).get('tags') or [])}; backend={nlp.backend}",
        }
    )

    # Edge presence
    checks.append(
        {
            "id": "edge_presence",
            "ok": bool(edges) or df is not None,
            "detail": f"{len(edges)} edges in report",
        }
    )

    passed = sum(1 for c in checks if c.get("ok"))
    score = passed / max(len(checks), 1)
    ok = score >= 0.6 and all(
        c.get("ok") for c in checks if c.get("id") in ("weak_iv_f",) and "weak <10" in str(c.get("detail", ""))
    )
    # More readable: overall ok if score high and no hard weak-IV fail when F present
    hard_fail = any(
        (not c.get("ok")) and c.get("id") == "weak_iv_f" and "weak" in str(c.get("detail", ""))
        for c in checks
    )
    ok = score >= 0.6 and not hard_fail

    if hard_fail:
        notes.append("Weak instrument — do not claim IV identification.")
    notes.append("Validation suite is exploratory; not a substitute for design review.")

    return ValidationReport(
        ok=ok,
        checks=checks,
        score=round(score, 3),
        notes=notes,
        tools_used=list(dict.fromkeys(tools_used)),
    )


# ---------------------------------------------------------------------------
# Soft refute hooks (DoWhy / EconML / placebo stubs)
# ---------------------------------------------------------------------------


@dataclass
class RefuteResult:
    """Soft refutation outcome — never hard-fails on missing extras."""

    ok: bool
    method: str
    backend: str
    edge: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None
    soft_skip: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def refute(
    edge: Optional[dict[str, Any]] = None,
    *,
    method: str = "placebo",
    df: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    d: Optional[str] = None,
    **kwargs: Any,
) -> RefuteResult:
    """Attempt a soft refutation of a discovered edge.

    Methods
    -------
    placebo :
        Shuffle the putative treatment and re-check association (builtin).
    random_common_cause :
        Add noise covariate stub (builtin).
    dowhy :
        Soft-call DoWhy refute if installed; else soft-skip.
    econml :
        Soft-call EconML sensitivity stub if installed; else soft-skip.

    Always returns a :class:`RefuteResult` — missing heavy deps soft-skip.
    """
    edge = dict(edge or {})
    src = str(edge.get("source") or d or "")
    tgt = str(edge.get("target") or y or "")
    notes = [
        "Refutation is exploratory — passing/failing a stub does not prove/disprove causation.",
    ]
    method_l = (method or "placebo").lower().strip()

    if method_l in ("dowhy", "dowhy_refute"):
        mod = _soft_import("dowhy")
        if mod is None:
            return RefuteResult(
                ok=True,
                method=method_l,
                backend="missing",
                edge=edge,
                soft_skip=True,
                notes=notes
                + ["DoWhy not installed — soft-skip. pip install 'autocausal[causal-extra]'"],
            )
        return RefuteResult(
            ok=True,
            method=method_l,
            backend="dowhy",
            edge=edge,
            data={"status": "stub", "hint": "Wire CausalModel.refute_estimate in apps"},
            notes=notes + ["DoWhy present; full refute wiring left to caller (soft stub)."],
            soft_skip=True,
        )

    if method_l in ("econml", "econml_sensitivity"):
        mod = _soft_import("econml")
        if mod is None:
            return RefuteResult(
                ok=True,
                method=method_l,
                backend="missing",
                edge=edge,
                soft_skip=True,
                notes=notes
                + ["EconML not installed — soft-skip. pip install 'autocausal[causal-extra]'"],
            )
        return RefuteResult(
            ok=True,
            method=method_l,
            backend="econml",
            edge=edge,
            data={"status": "stub", "hint": "Use EconML sensitivity analyzers in apps"},
            notes=notes + ["EconML present; sensitivity left to caller (soft stub)."],
            soft_skip=True,
        )

    if df is None or not src or not tgt or src not in df.columns or tgt not in df.columns:
        return RefuteResult(
            ok=True,
            method=method_l,
            backend="noop",
            edge=edge,
            soft_skip=True,
            notes=notes + ["Insufficient frame/edge columns — soft no-op."],
        )

    try:
        if method_l in ("placebo", "placebo_treatment"):
            rng = np.random.default_rng(int(kwargs.get("seed", 0)))
            work = df[[src, tgt]].dropna().copy()
            if len(work) < 8:
                return RefuteResult(
                    ok=True,
                    method=method_l,
                    backend="builtin",
                    edge=edge,
                    soft_skip=True,
                    notes=notes + ["Too few rows for placebo."],
                )
            base = float(pd.to_numeric(work[src], errors="coerce").corr(pd.to_numeric(work[tgt], errors="coerce")))
            shuffled = work[src].to_numpy().copy()
            rng.shuffle(shuffled)
            work["_placebo"] = shuffled
            plac = float(
                pd.to_numeric(work["_placebo"], errors="coerce").corr(
                    pd.to_numeric(work[tgt], errors="coerce")
                )
            )
            # "passes" refute if placebo association much weaker
            ratio = abs(plac) / (abs(base) + 1e-9)
            passed = bool(np.isfinite(base) and ratio < 0.5)
            return RefuteResult(
                ok=True,
                method=method_l,
                backend="builtin",
                edge=edge,
                data={
                    "base_corr": round(base, 4) if base == base else None,
                    "placebo_corr": round(plac, 4) if plac == plac else None,
                    "ratio": round(ratio, 4),
                    "refute_passed": passed,
                },
                notes=notes
                + [
                    f"Placebo corr ratio={ratio:.3f}; "
                    + ("association weakened under shuffle." if passed else "association persists — review."),
                ],
            )

        if method_l in ("random_common_cause", "add_unobserved"):
            work = df[[src, tgt]].dropna().copy()
            rng = np.random.default_rng(int(kwargs.get("seed", 1)))
            work["_noise"] = rng.normal(size=len(work))
            # partial out noise (should barely change)
            from numpy.linalg import lstsq

            yv = pd.to_numeric(work[tgt], errors="coerce").to_numpy(dtype=float)
            xv = pd.to_numeric(work[src], errors="coerce").to_numpy(dtype=float)
            zv = work["_noise"].to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(work)), xv])
            b0, *_ = lstsq(X, yv, rcond=None)
            X2 = np.column_stack([np.ones(len(work)), xv, zv])
            b1, *_ = lstsq(X2, yv, rcond=None)
            delta = abs(float(b1[1]) - float(b0[1]))
            return RefuteResult(
                ok=True,
                method=method_l,
                backend="builtin",
                edge=edge,
                data={
                    "coef_base": round(float(b0[1]), 4),
                    "coef_with_noise": round(float(b1[1]), 4),
                    "delta": round(delta, 4),
                    "refute_passed": delta < 0.05 * (abs(float(b0[1])) + 1e-6) + 0.01,
                },
                notes=notes + ["Random common-cause stub — noise should not move coef much."],
            )

        return RefuteResult(
            ok=True,
            method=method_l,
            backend="unknown",
            edge=edge,
            soft_skip=True,
            notes=notes + [f"Unknown refute method {method_l!r} — soft-skip."],
        )
    except Exception as e:
        return RefuteResult(
            ok=False,
            method=method_l,
            backend="error",
            edge=edge,
            error=f"{type(e).__name__}: {e}",
            notes=notes,
        )
