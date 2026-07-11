"""SLM / rule backends for causal *creation* and *inference*.

Creation: propose questions, instruments, morphemes from context.
Inference: interpret discovery/IV results with causal caveats.
Guide: search over intermediate pipeline outputs (legacy path).

Never blocks import on torch/transformers — HuggingFace loads lazily.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol

__all__ = [
    "GuideSuggestion",
    "GuideResult",
    "CreationResult",
    "InferenceResult",
    "RuleBackend",
    "RuleGuide",
    "HuggingFaceSLM",
    "get_backend",
    "get_guide",
    "guide_pipeline",
    "create_from_context",
    "infer_from_results",
    "slm_available",
    "slm_status",
    "probe_hardware",
    "recommend_qwen_model",
    "ensure_local_qwen",
    "DEFAULT_QWEN_CPU",
    "DEFAULT_QWEN_SMALL",
    "DEFAULT_QWEN_MID",
    "DEFAULT_QWEN_LARGE",
]

# Conservative Qwen2.5 Instruct picks by hardware class (exploratory SLM only).
DEFAULT_QWEN_CPU = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_QWEN_SMALL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_QWEN_MID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_QWEN_LARGE = "Qwen/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GuideSuggestion:
    action: str  # inspect_columns | drop_edge | validate_edge | instrument | confounder | search_query | focus_table
    detail: str
    priority: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuideResult:
    backend: str
    suggestions: list[GuideSuggestion]
    focus_columns: list[str] = field(default_factory=list)
    drop_edges: list[dict[str, str]] = field(default_factory=list)
    validate_edges: list[dict[str, str]] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    confounders: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    raw_text: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "focus_columns": self.focus_columns,
            "drop_edges": self.drop_edges,
            "validate_edges": self.validate_edges,
            "instruments": self.instruments,
            "confounders": self.confounders,
            "search_queries": self.search_queries,
            "raw_text": self.raw_text,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = ["# Guide suggestions", "", f"**Backend:** `{self.backend}`", ""]
        if self.focus_columns:
            lines.append("## Focus columns")
            lines.append("")
            for c in self.focus_columns:
                lines.append(f"- `{c}`")
            lines.append("")
        if self.validate_edges:
            lines.append("## Validate edges")
            lines.append("")
            for e in self.validate_edges:
                lines.append(f"- `{e.get('source')}` → `{e.get('target')}`")
            lines.append("")
        if self.drop_edges:
            lines.append("## Consider dropping")
            lines.append("")
            for e in self.drop_edges:
                lines.append(f"- `{e.get('source')}` → `{e.get('target')}`")
            lines.append("")
        if self.instruments:
            lines.append(f"**Instruments:** {', '.join(f'`{i}`' for i in self.instruments)}")
            lines.append("")
        if self.confounders:
            lines.append(f"**Confounders:** {', '.join(f'`{c}`' for c in self.confounders)}")
            lines.append("")
        if self.search_queries:
            lines.append("## Search queries")
            lines.append("")
            for q in self.search_queries:
                lines.append(f"- {q}")
            lines.append("")
        if self.suggestions:
            lines.append("## Actions")
            lines.append("")
            for s in self.suggestions:
                lines.append(f"- [{s.priority:.2f}] **{s.action}**: {s.detail}")
            lines.append("")
        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


@dataclass
class CreationResult:
    """SLM/rule proposals for causal *creation* (questions, Z, morphemes)."""

    backend: str
    questions: list[str] = field(default_factory=list)
    instruments: list[dict[str, Any]] = field(default_factory=list)
    morphemes: list[dict[str, Any]] = field(default_factory=list)
    roles: dict[str, list[str]] = field(default_factory=dict)
    raw_text: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = ["# Causal creation proposals", "", f"**Backend:** `{self.backend}`", ""]
        if self.questions:
            lines += ["## Questions", ""]
            for q in self.questions:
                lines.append(f"- {q}")
            lines.append("")
        if self.instruments:
            lines += ["## Instruments", ""]
            for z in self.instruments:
                lines.append(f"- `{z.get('name')}` — {z.get('rationale', '')} (score={z.get('score', '')})")
            lines.append("")
        if self.morphemes:
            lines += ["## Morphemes / role tags", ""]
            for m in self.morphemes:
                lines.append(f"- `{m.get('token')}` → {m.get('role')} ({m.get('detail', '')})")
            lines.append("")
        if self.roles:
            lines += ["## Role buckets", ""]
            for role, cols in self.roles.items():
                if cols:
                    lines.append(f"- **{role}:** {', '.join(f'`{c}`' for c in cols)}")
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


@dataclass
class InferenceResult:
    """SLM/rule interpretation of causal *inference* outputs."""

    backend: str
    narrative: str = ""
    caveats: list[str] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    raw_text: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Causal inference interpretation",
            "",
            f"**Backend:** `{self.backend}`",
            f"**Confidence:** {self.confidence:.2f}",
            "",
            "## Narrative",
            "",
            self.narrative or "_No narrative._",
            "",
        ]
        if self.claims:
            lines += ["## Claims", ""]
            for c in self.claims:
                lines.append(
                    f"- `{c.get('source')}` → `{c.get('target')}` "
                    f"(effect={c.get('effect', '—')}; {c.get('note', '')})"
                )
            lines.append("")
        if self.caveats:
            lines += ["## Caveats", ""] + [f"- {c}" for c in self.caveats] + [""]
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


class GuideBackend(Protocol):
    def guide(self, context: dict[str, Any]) -> GuideResult: ...


class CausalSLMBackend(Protocol):
    def guide(self, context: dict[str, Any]) -> GuideResult: ...
    def create(self, context: dict[str, Any]) -> CreationResult: ...
    def infer(self, context: dict[str, Any]) -> InferenceResult: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_flag(*names: str) -> bool:
    for n in names:
        if os.environ.get(n, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _context_summary(context: dict[str, Any]) -> str:
    parts = []
    if context.get("text"):
        parts.append(f"User question: {context['text']}")
    if context.get("emotion") or context.get("intent"):
        parts.append(
            f"Affect/intent: emotion={context.get('emotion')} intent={context.get('intent')} "
            f"valence={context.get('valence')} arousal={context.get('arousal')}"
        )
    cols = context.get("columns") or []
    if cols:
        names = [c.get("name", c) if isinstance(c, dict) else str(c) for c in cols[:20]]
        parts.append("Columns: " + ", ".join(names))
    assocs = context.get("associations") or []
    if assocs:
        top = assocs[:5]
        parts.append(
            "Top associations: "
            + "; ".join(f"{a.get('a')}~{a.get('b')}({a.get('score')})" for a in top)
        )
    edges = context.get("edges") or []
    if edges:
        parts.append(
            "Draft edges: "
            + "; ".join(f"{e.get('source')}->{e.get('target')}" for e in edges[:8])
        )
    cands = context.get("candidates") or {}
    if cands:
        parts.append(f"Candidates: {json.dumps(cands, default=str)[:400]}")
    iv = context.get("iv") or context.get("estimates")
    if iv:
        parts.append(f"Estimates: {json.dumps(iv, default=str)[:400]}")
    return "\n".join(parts)


def _col_names(context: dict[str, Any]) -> list[str]:
    columns = context.get("columns") or []
    return [c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in columns]


_IV_NAME_HINTS = ("z", "iv", "instrument", "assign", "lottery", "shock", "exog", "random")
_TREAT_HINTS = ("treat", "d_", "exposure", "campaign", "spend", "policy", "dose")
_OUT_HINTS = ("y_", "outcome", "revenue", "sales", "churn", "conversion", "kpi", "score")
_CONF_HINTS = ("age", "income", "region", "segment", "gender", "cohort", "baseline")


def _bucket_roles(col_names: list[str]) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {
        "instrument": [],
        "treatment": [],
        "outcome": [],
        "confounder": [],
    }
    for c in col_names:
        cl = c.lower()
        if any(h in cl for h in _IV_NAME_HINTS):
            roles["instrument"].append(c)
        if any(h in cl for h in _TREAT_HINTS):
            roles["treatment"].append(c)
        if any(h in cl for h in _OUT_HINTS):
            roles["outcome"].append(c)
        if any(h in cl for h in _CONF_HINTS):
            roles["confounder"].append(c)
    return roles


def slm_available() -> bool:
    """True if transformers can be imported (torch may still fail at load)."""
    try:
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def probe_hardware() -> dict[str, Any]:
    """Best-effort local hardware snapshot for Qwen model selection."""
    import shutil
    import subprocess
    import sys

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cuda": False,
        "vram_gb": None,
        "gpu_name": None,
        "ram_gb": None,
        "disk_free_gb": None,
        "torch": None,
        "notes": [],
    }
    try:
        root = os.path.abspath(os.sep)
        info["disk_free_gb"] = round(shutil.disk_usage(root).free / 1e9, 2)
    except Exception as e:
        info["notes"].append(f"disk probe soft-fail: {type(e).__name__}")

    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                info["ram_gb"] = round(m.ullTotalPhys / 1e9, 2)
                info["ram_avail_gb"] = round(m.ullAvailPhys / 1e9, 2)
        else:
            page = os.sysconf("SC_PAGE_SIZE")
            info["ram_gb"] = round(os.sysconf("SC_PHYS_PAGES") * page / 1e9, 2)
    except Exception as e:
        info["notes"].append(f"RAM probe soft-fail: {type(e).__name__}")

    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=12,
        )
        line = (r.stdout or "").strip().splitlines()
        if line and r.returncode == 0 and "failed" not in (r.stdout or "").lower():
            parts = [p.strip() for p in line[0].split(",")]
            if len(parts) >= 2:
                info["gpu_name"] = parts[0]
                try:
                    # nvidia-smi memory is MiB when nounits
                    info["vram_gb"] = round(float(parts[1]) / 1024.0, 2)
                except (TypeError, ValueError):
                    pass
                info["nvidia_smi"] = line[0]
        else:
            info["notes"].append(
                "nvidia-smi unavailable or insufficient permissions — treating as CPU."
            )
    except Exception as e:
        info["notes"].append(f"nvidia-smi soft-fail: {type(e).__name__}")

    try:
        import torch

        info["torch"] = getattr(torch, "__version__", "unknown")
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu_name"] = info["gpu_name"] or torch.cuda.get_device_name(0)
            info["vram_gb"] = info["vram_gb"] or round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
    except Exception as e:
        info["torch"] = None
        info["notes"].append(f"torch soft-fail: {type(e).__name__}")

    return info


def recommend_qwen_model(hardware: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Pick a conservative Qwen2.5 Instruct id for local guide/create/infer.

    Prefers instruct/chat models. Stays small on CPU or tight VRAM to avoid OOM.
    """
    hw = hardware or probe_hardware()
    cuda = bool(hw.get("cuda"))
    vram = hw.get("vram_gb")
    ram = hw.get("ram_gb") or 0
    try:
        vram_f = float(vram) if vram is not None else 0.0
    except (TypeError, ValueError):
        vram_f = 0.0

    load_in_4bit = False
    if cuda and vram_f >= 24:
        model = DEFAULT_QWEN_LARGE
        reason = f"CUDA VRAM≈{vram_f}GB ≥24 — Qwen2.5-7B-Instruct"
    elif cuda and vram_f >= 12:
        model = DEFAULT_QWEN_MID
        reason = f"CUDA VRAM≈{vram_f}GB (~12–16) — Qwen2.5-3B-Instruct"
        load_in_4bit = vram_f < 14
    elif cuda and vram_f >= 6:
        model = DEFAULT_QWEN_SMALL
        reason = f"CUDA VRAM≈{vram_f}GB (≤8–12) — Qwen2.5-1.5B-Instruct"
        load_in_4bit = True
    elif cuda and vram_f > 0:
        model = DEFAULT_QWEN_CPU
        reason = f"CUDA VRAM≈{vram_f}GB tight — Qwen2.5-0.5B-Instruct"
        load_in_4bit = True
    elif ram >= 32:
        model = DEFAULT_QWEN_SMALL
        reason = f"CPU-only with RAM≈{ram}GB — Qwen2.5-1.5B-Instruct (conservative)"
    else:
        model = DEFAULT_QWEN_CPU
        reason = f"CPU-only / limited RAM≈{ram}GB — Qwen2.5-0.5B-Instruct"

    # Respect explicit env override for recommendation reporting
    env_model = (os.environ.get("AUTOCAUSAL_SLM_MODEL") or "").strip()
    return {
        "model_id": env_model or model,
        "recommended_model_id": model,
        "reason": reason,
        "load_in_4bit": load_in_4bit and cuda,
        "hardware": hw,
        "epistemic": "SLM guides analysis; does not identify causation.",
    }


def ensure_local_qwen(
    *,
    model_id: Optional[str] = None,
    download: bool = True,
    set_env: bool = True,
    token: Optional[str] = None,
) -> dict[str, Any]:
    """Probe hardware, pick Qwen Instruct, optionally download into HF cache.

    Soft-fails on network/auth errors. Never prints tokens.
    Token resolution order: ``token`` arg → ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN``
    env → ``.env`` in cwd / package parent (keys only; values not logged).
    """
    rec = recommend_qwen_model()
    mid = (model_id or rec["model_id"] or rec["recommended_model_id"]).strip()
    out: dict[str, Any] = {
        "ok": False,
        "model_id": mid,
        "recommended": rec,
        "cache_dir": None,
        "downloaded": False,
        "env_set": False,
        "notes": [
            "Epistemic: SLM guides AutoCausal analysis; it does not identify causation.",
        ],
    }

    if set_env:
        os.environ["AUTOCAUSAL_SLM_MODEL"] = mid
        os.environ.setdefault("AUTOCAUSAL_SLM", "1")
        out["env_set"] = True

    # Resolve HF token without echoing secrets
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not tok:
        tok = _read_dotenv_token(("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"))

    if not download:
        out["ok"] = True
        out["notes"].append("download=False — model id recorded only.")
        return out

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:
        out["notes"].append(
            f"huggingface_hub missing ({type(e).__name__}); pip install huggingface_hub. "
            "Rule backend still works."
        )
        return out

    try:
        cache = snapshot_download(repo_id=mid, token=tok)
        out["cache_dir"] = str(cache)
        out["downloaded"] = True
        out["ok"] = True
        out["notes"].append(f"Cached locally under HF hub cache for `{mid}`.")
    except Exception as e:
        out["notes"].append(
            f"Download soft-fail ({type(e).__name__}): network/auth may block. "
            "Set HF_TOKEN or retry later; rule backend remains available."
        )
        # Still ok for wiring if transformers can resolve later
        out["ok"] = False
    return out


def _read_dotenv_token(keys: tuple[str, ...]) -> Optional[str]:
    """Load a token from nearby .env files without logging values."""
    from pathlib import Path

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parents[3] / ".env",
    ]
    for path in candidates:
        try:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, _, v = s.partition("=")
                if k.strip() in keys:
                    val = v.strip().strip('"').strip("'")
                    if val:
                        return val
        except Exception:
            continue
    return None


def slm_status() -> dict[str, Any]:
    """Backend availability snapshot for status UIs (e.g. CausalBridge)."""
    env_on = _env_flag("AUTOCAUSAL_SLM", "EMOTIVEVISION_SLM", "CAUSALIV_SLM")
    transformers_ok = False
    torch_ok = False
    err = None
    try:
        import transformers  # noqa: F401

        transformers_ok = True
    except Exception as e:
        err = f"transformers: {type(e).__name__}: {e}"
    try:
        import torch  # noqa: F401

        torch_ok = True
    except Exception as e:
        if err is None:
            err = f"torch: {type(e).__name__}: {e}"
    rec = recommend_qwen_model()
    return {
        "rule_backend": True,
        "env_slm_enabled": env_on,
        "transformers_installed": transformers_ok,
        "torch_installed": torch_ok,
        "huggingface_ready": transformers_ok and torch_ok,
        "default_model": os.environ.get("AUTOCAUSAL_SLM_MODEL")
        or rec.get("recommended_model_id")
        or DEFAULT_QWEN_SMALL,
        "recommended_qwen": rec,
        "recommended_instruct": [
            DEFAULT_QWEN_CPU,
            DEFAULT_QWEN_SMALL,
            DEFAULT_QWEN_MID,
            DEFAULT_QWEN_LARGE,
            "HuggingFaceTB/SmolLM2-360M-Instruct",
        ],
        "hardware": rec.get("hardware"),
        "error": err,
        "epistemic": "SLM guides analysis; does not identify causation.",
    }


# ---------------------------------------------------------------------------
# Rule backend (always available)
# ---------------------------------------------------------------------------


class RuleBackend:
    """Deterministic offline backend — always available (alias: RuleGuide)."""

    name = "rule"

    def guide(self, context: dict[str, Any]) -> GuideResult:
        edges = list(context.get("edges") or [])
        assocs = list(context.get("associations") or [])
        candidates = dict(context.get("candidates") or {})
        text = (context.get("text") or "").lower()
        col_names = _col_names(context)

        suggestions: list[GuideSuggestion] = []
        focus: list[str] = []
        validate: list[dict[str, str]] = []
        drop: list[dict[str, str]] = []
        instruments = list(candidates.get("instrument") or [])
        confounders = list(candidates.get("confounder") or [])
        queries: list[str] = []

        ranked = sorted(
            edges,
            key=lambda e: float(e.get("confidence") or e.get("score") or 0),
            reverse=True,
        )
        for e in ranked[:5]:
            validate.append({"source": str(e["source"]), "target": str(e["target"])})
            focus.extend([str(e["source"]), str(e["target"])])
            suggestions.append(
                GuideSuggestion(
                    action="validate_edge",
                    detail=f"Validate {e['source']} → {e['target']} (conf={e.get('confidence', e.get('score'))})",
                    priority=float(e.get("confidence") or 0.5),
                    meta=e,
                )
            )

        for e in edges:
            names = f"{e.get('source', '')} {e.get('target', '')}".lower()
            score = float(e.get("confidence") or e.get("score") or 0)
            if "noise" in names or score < 0.12:
                drop.append({"source": str(e["source"]), "target": str(e["target"])})
                suggestions.append(
                    GuideSuggestion(
                        action="drop_edge",
                        detail=f"Consider dropping weak/noise edge {e['source']}→{e['target']}",
                        priority=0.3,
                    )
                )

        for c in col_names:
            cl = c.lower()
            if any(h in cl for h in _IV_NAME_HINTS):
                if c not in instruments:
                    instruments.append(c)
                suggestions.append(
                    GuideSuggestion(
                        action="instrument",
                        detail=f"Column `{c}` looks instrument-like",
                        priority=0.7,
                    )
                )
            if any(h in cl for h in _CONF_HINTS):
                if c not in confounders:
                    confounders.append(c)

        if text:
            for c in col_names:
                if c.lower() in text or any(
                    tok and tok in text for tok in re.split(r"\W+", c.lower()) if len(tok) > 2
                ):
                    focus.append(c)
                    suggestions.append(
                        GuideSuggestion(
                            action="inspect_columns",
                            detail=f"User question mentions `{c}`",
                            priority=0.85,
                        )
                    )
            m = re.search(r"caus(?:e|es|ing)\s+(\w+)", text)
            if m:
                y = m.group(1)
                queries.append(f"what causes {y}")
                for c in col_names:
                    if y in c.lower():
                        focus.append(c)

        for a in assocs[:5]:
            focus.extend([str(a.get("a")), str(a.get("b"))])
            suggestions.append(
                GuideSuggestion(
                    action="inspect_columns",
                    detail=f"Strong association {a.get('a')}~{a.get('b')} ({a.get('metric')}={a.get('score')})",
                    priority=float(a.get("score") or 0.4),
                )
            )

        for e in validate[:3]:
            queries.append(f"causal evidence {e['source']} affects {e['target']}")

        seen: set[str] = set()
        focus_u = []
        for c in focus:
            if c and c not in seen:
                seen.add(c)
                focus_u.append(c)

        return GuideResult(
            backend=self.name,
            suggestions=suggestions[:40],
            focus_columns=focus_u[:20],
            drop_edges=drop[:20],
            validate_edges=validate[:20],
            instruments=instruments[:10],
            confounders=confounders[:10],
            search_queries=queries[:10],
            notes=["RuleBackend is offline and deterministic."],
        )

    def create(self, context: dict[str, Any]) -> CreationResult:
        col_names = _col_names(context)
        roles = _bucket_roles(col_names)
        text = (context.get("text") or "").strip()
        emotion = context.get("emotion")
        intent = context.get("intent")
        candidates = dict(context.get("candidates") or {})

        # Prefer candidate lists from discovery when present
        for key in ("instrument", "treatment", "outcome", "confounder"):
            for c in candidates.get(key) or []:
                if c not in roles[key]:
                    roles[key].append(str(c))

        questions: list[str] = []
        treats = roles["treatment"] or [c for c in col_names if "x" in c.lower()][:2]
        outs = roles["outcome"] or [c for c in col_names if "y" in c.lower()][:2]
        zs = roles["instrument"]

        if treats and outs:
            questions.append(f"Does `{treats[0]}` cause `{outs[0]}`?")
        if zs and treats and outs:
            questions.append(
                f"Using `{zs[0]}` as an instrument, what is the effect of `{treats[0]}` on `{outs[0]}`?"
            )
        if emotion or intent:
            questions.append(
                f"How does affect ({emotion or 'unknown'}) / intent ({intent or 'unknown'}) "
                f"relate to outcomes {', '.join(f'`{o}`' for o in outs[:2]) or 'KPIs'}?"
            )
        if text:
            questions.append(f"Refine: {text}")
        if not questions:
            questions.append("Which columns are plausible treatments vs outcomes?")

        instruments: list[dict[str, Any]] = []
        for z in zs[:8]:
            instruments.append(
                {
                    "name": z,
                    "rationale": "Name/heuristic suggests exogenous or assignment-like variable",
                    "score": 0.7,
                }
            )
        # text cues
        lower = text.lower()
        for cue, name, score in (
            ("lottery", "lottery_assignment", 0.85),
            ("random", "randomized_assignment", 0.9),
            ("rainfall", "weather_rainfall", 0.75),
            ("judge", "judge_leniency", 0.8),
            ("shift-share", "shift_share", 0.8),
            ("bartik", "bartik_shift_share", 0.85),
        ):
            if cue in lower:
                instruments.append(
                    {"name": name, "rationale": f"Text cue '{cue}'", "score": score}
                )

        morphemes: list[dict[str, Any]] = []
        for role, cols in roles.items():
            for c in cols[:6]:
                morphemes.append(
                    {
                        "token": c,
                        "role": role,
                        "detail": f"Heuristic role tag from column name ({role})",
                    }
                )
        if emotion:
            morphemes.append(
                {
                    "token": str(emotion),
                    "role": "affect_context",
                    "detail": "Emotive context — not a causal instrument by itself",
                }
            )
        if intent:
            morphemes.append(
                {
                    "token": str(intent),
                    "role": "intent_context",
                    "detail": "Communicative intent — use as covariate/context, not Z",
                }
            )

        return CreationResult(
            backend=self.name,
            questions=questions[:12],
            instruments=instruments[:12],
            morphemes=morphemes[:30],
            roles=roles,
            notes=[
                "RuleBackend creation is heuristic; validate exclusion/relevance before IV.",
                "Affect/intent are contextual — do not treat as instruments.",
            ],
        )

    def infer(self, context: dict[str, Any]) -> InferenceResult:
        edges = list(context.get("edges") or [])
        iv = context.get("iv") or context.get("estimates") or {}
        emotion = context.get("emotion")
        intent = context.get("intent")
        caveats = [
            "Exploratory associations are not identified causal effects.",
            "Check instrument relevance (first-stage F) and exclusion before claiming IV effects.",
            "Placebo and sensitivity checks are recommended before decisions.",
        ]
        claims: list[dict[str, Any]] = []
        for e in edges[:8]:
            claims.append(
                {
                    "source": e.get("source"),
                    "target": e.get("target"),
                    "effect": e.get("confidence", e.get("score")),
                    "note": "exploratory edge",
                }
            )

        narrative_parts = []
        if claims:
            top = claims[0]
            narrative_parts.append(
                f"Top exploratory link: `{top['source']}` → `{top['target']}` "
                f"(score={top['effect']}). Treat as a hypothesis, not proof."
            )
        if isinstance(iv, dict) and iv:
            fstat = iv.get("first_stage_f") or iv.get("f")
            coef = iv.get("coef") or iv.get("ate") or iv.get("effect")
            if coef is not None:
                narrative_parts.append(f"Point estimate ≈ {coef}.")
            if fstat is not None:
                try:
                    fval = float(fstat)
                    if fval < 10:
                        caveats.append(f"Weak instrument warning: first-stage F≈{fval:.2f} (<10).")
                    narrative_parts.append(f"First-stage F≈{fval:.2f}.")
                except (TypeError, ValueError):
                    pass
        if emotion or intent:
            narrative_parts.append(
                f"Affect/intent context ({emotion}/{intent}) may correlate with outcomes; "
                "do not interpret as a causal mechanism without design."
            )
        if not narrative_parts:
            narrative_parts.append("Insufficient structure for a strong causal narrative.")

        conf = 0.35
        if claims:
            try:
                conf = min(0.75, 0.35 + float(claims[0].get("effect") or 0) * 0.4)
            except (TypeError, ValueError):
                pass

        return InferenceResult(
            backend=self.name,
            narrative=" ".join(narrative_parts),
            caveats=caveats,
            claims=claims,
            confidence=round(conf, 3),
            notes=["RuleBackend inference is template-based."],
        )


# Back-compat alias
RuleGuide = RuleBackend


# ---------------------------------------------------------------------------
# HuggingFace SLM (lazy)
# ---------------------------------------------------------------------------


class HuggingFaceSLM:
    """Lazy Hugging Face transformers backend (optional heavy deps).

    Prefer instruct/chat models via ``AUTOCAUSAL_SLM_MODEL`` (e.g. Qwen2.5-*-Instruct).
    Soft-fails to ``RuleBackend`` enrichment when load/generate fails.
    """

    name = "huggingface"

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        load_in_4bit: Optional[bool] = None,
    ) -> None:
        env_model = os.environ.get("AUTOCAUSAL_SLM_MODEL")
        if model_name:
            self.model_name = model_name
        elif env_model:
            self.model_name = env_model
        else:
            # Safe default for CI; call ensure_local_qwen() / set AUTOCAUSAL_SLM_MODEL for Qwen.
            self.model_name = "sshleifer/tiny-gpt2"
        self.load_in_4bit = load_in_4bit
        self._pipe = None
        self._tokenizer = None
        self._error: Optional[str] = None
        self._rule = RuleBackend()
        self._is_chat = self._detect_chat(self.model_name)

    @staticmethod
    def _detect_chat(name: str) -> bool:
        low = (name or "").lower()
        return any(k in low for k in ("instruct", "chat", "qwen", "phi-3", "smollm"))

    def _ensure(self) -> bool:
        if self._pipe is not None:
            return True
        if self._error:
            return False
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
        except ImportError as e:
            self._error = (
                "transformers not installed; pip install 'auto-causal-lib' "
                f"(base now includes transformers). ({e})"
            )
            return False
        try:
            import torch
        except Exception as e:
            self._error = f"torch required for HF SLM: {type(e).__name__}: {e}"
            return False

        try:
            tok_kwargs: dict[str, Any] = {"trust_remote_code": True}
            model_kwargs: dict[str, Any] = {"trust_remote_code": True}
            device = -1
            use_4bit = self.load_in_4bit
            if use_4bit is None:
                use_4bit = bool(torch.cuda.is_available()) and _env_flag(
                    "AUTOCAUSAL_SLM_4BIT"
                )
            if torch.cuda.is_available():
                device = 0
                if use_4bit:
                    try:
                        import bitsandbytes  # noqa: F401

                        from transformers import BitsAndBytesConfig  # type: ignore

                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                        )
                        model_kwargs["device_map"] = "auto"
                        device = None  # type: ignore[assignment]
                    except Exception:
                        use_4bit = False

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tok_kwargs)
            if getattr(tokenizer, "pad_token", None) is None and getattr(
                tokenizer, "eos_token", None
            ):
                tokenizer.pad_token = tokenizer.eos_token

            # Prefer pipeline for simplicity; fall back to manual if needed
            pipe_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "tokenizer": tokenizer,
                "trust_remote_code": True,
                "max_new_tokens": 160,
            }
            if device is not None:
                pipe_kwargs["device"] = device
            if "quantization_config" in model_kwargs:
                # Load model explicitly for 4bit
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **model_kwargs
                )
                pipe_kwargs["model"] = model
                pipe_kwargs.pop("device", None)

            self._pipe = pipeline("text-generation", **pipe_kwargs)
            self._tokenizer = tokenizer
            return True
        except Exception as e:
            self._error = f"SLM load failed (soft-fail): {type(e).__name__}: {e}"
            return False

    def _format_prompt(self, system: str, user: str) -> str:
        if self._is_chat and self._tokenizer is not None:
            try:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                apply = getattr(self._tokenizer, "apply_chat_template", None)
                if callable(apply):
                    return apply(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            except Exception:
                pass
        return f"{system}\n\n{user}\n\nAssistant:"

    def _generate(self, prompt: str, *, system: str = "") -> str:
        if not self._ensure():
            return ""
        try:
            assert self._pipe is not None
            full = (
                self._format_prompt(
                    system
                    or (
                        "You are a cautious causal analysis assistant. "
                        "Guide exploratory analysis only; never claim identification."
                    ),
                    prompt,
                )
                if self._is_chat
                else (
                    (system + "\n\n" if system else "")
                    + prompt
                )
            )
            out = self._pipe(
                full,
                do_sample=False,
                truncation=True,
                return_full_text=False,
            )
            if not out:
                return ""
            text = out[0].get("generated_text") or ""
            if isinstance(text, list):
                # some chat pipelines return message lists
                text = " ".join(str(x) for x in text)
            text = str(text).strip()
            if text.startswith(full):
                text = text[len(full) :]
            return text.strip()[:2000]
        except Exception as e:
            self._error = f"SLM generate soft-fail: {type(e).__name__}: {e}"
            return ""

    def guide(self, context: dict[str, Any]) -> GuideResult:
        base = self._rule.guide(context)
        if not self._ensure():
            base.backend = "rule+hf_unavailable"
            base.notes.append(self._error or "HF SLM unavailable")
            return base

        user = (
            "Given this summary, list next steps as short bullets: columns to inspect, "
            "edges to validate/drop, instruments, confounders, search queries.\n\n"
            + _context_summary(context)
            + "\n\nSuggestions:"
        )
        text = self._generate(
            user,
            system=(
                "You guide AutoCausal exploratory analysis. "
                "Associations are not causation. Be concise."
            ),
        )
        if not text:
            base.backend = "rule+hf_error"
            base.notes.append(self._error or "SLM generate failed")
            return base

        base.raw_text = text
        base.backend = f"huggingface:{self.model_name}"
        for line in re.split(r"[\n;]+", base.raw_text):
            line = line.strip(" -*\t")
            if len(line) < 8:
                continue
            action = "inspect_columns"
            low = line.lower()
            if "drop" in low:
                action = "drop_edge"
            elif "valid" in low or "confirm" in low:
                action = "validate_edge"
            elif "instrument" in low:
                action = "instrument"
            elif "confound" in low:
                action = "confounder"
            elif "search" in low or "query" in low:
                action = "search_query"
                base.search_queries.append(line[:200])
            base.suggestions.append(
                GuideSuggestion(action=action, detail=line[:300], priority=0.55)
            )
        base.notes.append(
            "HuggingFace SLM used; parse is heuristic. Not causal identification."
        )
        return base

    def create(self, context: dict[str, Any]) -> CreationResult:
        base = self._rule.create(context)
        text = self._generate(
            "Propose causal questions, instruments (Z), and role tags from this context. "
            "Short bullets only.\n\n"
            + _context_summary(context)
            + "\n\nProposals:",
            system="Propose exploratory questions only; do not claim effects are identified.",
        )
        if not text:
            base.backend = "rule+hf_unavailable" if self._error else base.backend
            if self._error:
                base.notes.append(self._error)
            return base
        base.raw_text = text
        base.backend = f"huggingface:{self.model_name}"
        for line in re.split(r"[\n;]+", text):
            line = line.strip(" -*\t")
            if len(line) < 8:
                continue
            low = line.lower()
            if "?" in line or "does " in low or "what " in low:
                base.questions.append(line[:240])
            elif "instrument" in low or " z " in f" {low} ":
                base.instruments.append(
                    {"name": line[:80], "rationale": "SLM proposal", "score": 0.55}
                )
            else:
                base.morphemes.append(
                    {"token": line[:60], "role": "slm_tag", "detail": line[:200]}
                )
        base.notes.append("HuggingFace SLM creation; merge with rule heuristics.")
        return base

    def infer(self, context: dict[str, Any]) -> InferenceResult:
        base = self._rule.infer(context)
        text = self._generate(
            "Interpret these causal results cautiously. Give a short narrative and caveats.\n\n"
            + _context_summary(context)
            + "\n\nInterpretation:",
            system=(
                "Interpret exploratory results only. Always include caveats that "
                "associations are not identified causal effects."
            ),
        )
        if not text:
            base.backend = "rule+hf_unavailable" if self._error else base.backend
            if self._error:
                base.notes.append(self._error)
            return base
        base.raw_text = text
        base.backend = f"huggingface:{self.model_name}"
        base.narrative = (text.split("\n")[0][:500] or base.narrative)
        for line in re.split(r"[\n;]+", text):
            line = line.strip(" -*\t")
            if any(
                k in line.lower()
                for k in ("caveat", "warning", "not causal", "confound", "weak")
            ):
                if line not in base.caveats:
                    base.caveats.append(line[:240])
        base.notes.append("HuggingFace SLM inference; caveats still apply.")
        return base


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def get_backend(
    *,
    use_slm: bool = False,
    model_name: Optional[str] = None,
) -> CausalSLMBackend:
    """Return RuleBackend by default; HF when use_slm or AUTOCAUSAL_SLM=1."""
    env_on = _env_flag("AUTOCAUSAL_SLM", "EMOTIVEVISION_SLM", "CAUSALIV_SLM")
    if use_slm or env_on:
        return HuggingFaceSLM(model_name=model_name)
    return RuleBackend()


def get_guide(*, use_slm: bool = False, model_name: Optional[str] = None) -> GuideBackend:
    """Back-compat: same as get_backend."""
    return get_backend(use_slm=use_slm, model_name=model_name)


def guide_pipeline(
    context: dict[str, Any],
    *,
    use_slm: bool = False,
    model_name: Optional[str] = None,
) -> GuideResult:
    return get_backend(use_slm=use_slm, model_name=model_name).guide(context)


def create_from_context(
    context: dict[str, Any],
    *,
    use_slm: bool = False,
    model_name: Optional[str] = None,
) -> CreationResult:
    """Creation API: questions / instruments / morphemes."""
    return get_backend(use_slm=use_slm, model_name=model_name).create(context)


def infer_from_results(
    context: dict[str, Any],
    *,
    use_slm: bool = False,
    model_name: Optional[str] = None,
) -> InferenceResult:
    """Inference API: narrative + caveats over edges/estimates."""
    return get_backend(use_slm=use_slm, model_name=model_name).infer(context)
