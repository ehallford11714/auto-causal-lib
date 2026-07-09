"""KineteqPivotEmbeddingGuide — soft-optional pivot embeddings + local fallback.

On disk there is no standalone `kineteq` Python package. EmotiveVision documents
a Kineteq MCP JSON-RPC bus (`KINETEQ_MCP_URL` / `KINETEQ_AUTH_TOKEN`). This
adapter:

1. Tries optional local modules: `kineteq`, `kineteq_pivot`, `pivot_embeddings`
2. Optionally calls Kineteq MCP `tools/call` for embedding/pivot tools when
   `AUTOCAUSAL_KINETEQ_MCP=1` (or live MCP env) is set
3. Falls back to a **local hashing/TF-IDF-like** column-token embedding labeled
   `pivot_fallback` (NOT Kineteq)
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Optional

from autocausal.guides.types import GuideResult, GuideSuggestion, col_names, uniq

_DIM = 64


def _env_flag(*names: str) -> bool:
    for n in names:
        if os.environ.get(n, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def kineteq_module_importable() -> tuple[bool, str]:
    for mod in ("kineteq", "kineteq_pivot", "pivot_embeddings"):
        try:
            __import__(mod)
            return True, mod
        except Exception:
            continue
    return False, ""


def kineteq_mcp_configured() -> bool:
    url = (
        os.environ.get("AUTOCAUSAL_KINETEQ_MCP_URL")
        or os.environ.get("KINETEQ_MCP_URL")
        or os.environ.get("EMOTIVEVISION_MCP_URL")
        or ""
    ).strip()
    return bool(url) and _env_flag(
        "AUTOCAUSAL_KINETEQ_MCP",
        "EMOTIVEVISION_LIVE_MCP",
        "KINETEQ_LIVE_MCP",
    )


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if len(t) > 1]


def _hash_embed(tokens: list[str], dim: int = _DIM) -> list[float]:
    """Deterministic hashing trick embedding (local fallback, not Kineteq)."""
    vec = [0.0] * dim
    if not tokens:
        return vec
    for t in tokens:
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()
        idx = int(h[:8], 16) % dim
        sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
        # TF weight
        vec[idx] += sign
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _texts_from_context(context: dict[str, Any]) -> dict[str, str]:
    """Map entity id → text to embed (columns, associations, user text)."""
    out: dict[str, str] = {}
    for c in col_names(context):
        out[f"col:{c}"] = c.replace("_", " ")
    for i, a in enumerate(context.get("associations") or []):
        a_n, b_n = a.get("a"), a.get("b")
        if a_n and b_n:
            out[f"assoc:{i}"] = f"{a_n} associated {b_n} score {a.get('score')}"
    text = (context.get("text") or "").strip()
    if text:
        out["user:text"] = text
    return out


class KineteqPivotEmbeddingGuide:
    """Embed columns + associations + user text; nearest neighbors suggest direction."""

    name = "kineteq_pivot"

    def __init__(self, *, top_k: int = 5) -> None:
        self.top_k = top_k

    def available(self) -> bool:
        mod_ok, _ = kineteq_module_importable()
        return mod_ok or kineteq_mcp_configured()

    def guide(self, context: dict[str, Any]) -> GuideResult:
        names = col_names(context)
        text = (context.get("text") or "").strip()
        edges = list(context.get("edges") or [])
        candidates = dict(context.get("candidates") or {})
        notes: list[str] = []
        embeddings: dict[str, list[float]] = {}
        backend_label = "pivot_fallback"
        used_kineteq = False

        mod_ok, mod_name = kineteq_module_importable()
        if mod_ok:
            try:
                embeddings = self._embed_via_module(mod_name, context)
                backend_label = f"kineteq_module:{mod_name}"
                used_kineteq = True
                notes.append(f"Used local module `{mod_name}` for pivot embeddings.")
            except Exception as e:
                notes.append(f"{mod_name} soft-fail: {type(e).__name__}: {e}")

        if not embeddings and kineteq_mcp_configured():
            try:
                embeddings = self._embed_via_mcp(context)
                if embeddings:
                    backend_label = "kineteq_mcp"
                    used_kineteq = True
                    notes.append("Used Kineteq MCP tools/call for embeddings.")
            except Exception as e:
                notes.append(f"Kineteq MCP soft-fail: {type(e).__name__}: {e}")

        if not embeddings:
            embeddings = self._embed_fallback(context)
            backend_label = "pivot_fallback"
            notes.append(
                "Local hashing/TF-IDF-like pivot embedding fallback "
                "(NOT Kineteq). Set KINETEQ_MCP_URL + AUTOCAUSAL_KINETEQ_MCP=1 "
                "or install a kineteq pivot module when available."
            )

        # Pivot around user text (or centroid of columns)
        if "user:text" in embeddings:
            pivot = embeddings["user:text"]
            pivot_name = "user:text"
        else:
            col_vecs = [embeddings[k] for k in embeddings if k.startswith("col:")]
            if col_vecs:
                dim = len(col_vecs[0])
                pivot = [sum(v[i] for v in col_vecs) / len(col_vecs) for i in range(dim)]
                pivot_name = "col_centroid"
            else:
                pivot = _hash_embed(_tokenize(text or "causal"))
                pivot_name = "empty"

        neighbors = []
        for key, vec in embeddings.items():
            if key == "user:text":
                continue
            neighbors.append((key, _cosine(pivot, vec)))
        neighbors.sort(key=lambda x: x[1], reverse=True)

        related: list[str] = []
        focus: list[str] = []
        instruments: list[str] = list(candidates.get("instrument") or [])
        for key, score in neighbors[: self.top_k * 2]:
            if key.startswith("col:"):
                col = key[4:]
                related.append(col)
                focus.append(col)
                cl = col.lower()
                if any(h in cl for h in ("z", "iv", "instrument", "assign", "lottery")):
                    if col not in instruments:
                        instruments.append(col)
            elif key.startswith("assoc:"):
                # pull columns from association text
                pass

        # Boost edges whose endpoints are near the pivot
        focus_set = set(focus[: self.top_k])
        boost: list[dict[str, Any]] = []
        suppress: list[dict[str, Any]] = []
        for e in edges:
            src, tgt = str(e.get("source")), str(e.get("target"))
            if src in focus_set and tgt in focus_set:
                boost.append(
                    {
                        "source": src,
                        "target": tgt,
                        "reason": "pivot_nn_both",
                        "backend": self.name,
                        "score": float(e.get("confidence") or e.get("score") or 0),
                    }
                )
            elif src not in focus_set and tgt not in focus_set and focus_set:
                conf = float(e.get("confidence") or e.get("score") or 0)
                if conf < 0.2:
                    suppress.append(
                        {
                            "source": src,
                            "target": tgt,
                            "reason": "far_from_pivot",
                            "backend": self.name,
                        }
                    )

        # Suggest instruments from neighbors of treatment-like columns
        treatment = [c for c in focus if any(h in c.lower() for h in ("treat", "spend", "campaign", "policy"))]
        outcome = [c for c in focus if any(h in c.lower() for h in ("revenue", "sales", "y_", "outcome", "churn"))]

        queries = []
        if text:
            queries.append(text[:160])
        for col in related[:3]:
            queries.append(f"variables related to {col}")

        suggestions = [
            GuideSuggestion(
                action="inspect_columns",
                detail=f"Pivot-NN `{col}` near {pivot_name}",
                priority=0.7,
            )
            for col in related[:8]
        ]
        for z in instruments[:4]:
            suggestions.append(
                GuideSuggestion(
                    action="instrument",
                    detail=f"Pivot embedding suggests instrument-like `{z}`",
                    priority=0.72,
                )
            )

        return GuideResult(
            backend=backend_label if used_kineteq else "pivot_fallback",
            available=used_kineteq,
            suggestions=suggestions[:40],
            focus_columns=uniq(focus, limit=20),
            instruments=uniq(instruments, limit=10),
            treatment=uniq(treatment, limit=8),
            outcome=uniq(outcome, limit=8),
            related_variables=uniq(related, limit=15),
            boost_edges=boost[:20],
            suppress_edges=suppress[:20],
            drop_edges=[
                {"source": e["source"], "target": e["target"]} for e in suppress[:20]
            ],
            validate_edges=[
                {"source": e["source"], "target": e["target"]} for e in boost[:20]
            ],
            search_queries=uniq(queries, limit=10),
            next_questions=[
                f"Are {', '.join(f'`{c}`' for c in related[:3])} related instruments/confounders?"
            ]
            if related
            else [],
            notes=notes
            + [
                f"pivot={pivot_name}",
                "Nearest neighbors in embedding space steer discovery focus.",
            ],
        )

    def _embed_fallback(self, context: dict[str, Any]) -> dict[str, list[float]]:
        texts = _texts_from_context(context)
        return {k: _hash_embed(_tokenize(v)) for k, v in texts.items()}

    def _embed_via_module(self, mod_name: str, context: dict[str, Any]) -> dict[str, list[float]]:
        """Bind to future kineteq APIs; raise if contract missing so caller falls back."""
        mod = __import__(mod_name)
        texts = _texts_from_context(context)
        # Documented future contracts
        if hasattr(mod, "embed_texts"):
            vectors = mod.embed_texts(list(texts.values()))
            return {k: list(map(float, vectors[i])) for i, k in enumerate(texts)}
        if hasattr(mod, "pivot_embed"):
            return {k: list(map(float, mod.pivot_embed(v))) for k, v in texts.items()}
        if hasattr(mod, "PivotEmbeddings"):
            eng = mod.PivotEmbeddings()
            return {k: list(map(float, eng.embed(v))) for k, v in texts.items()}
        raise AttributeError(
            f"{mod_name} has no embed_texts/pivot_embed/PivotEmbeddings — "
            "adapter waiting for documented API"
        )

    def _embed_via_mcp(self, context: dict[str, Any]) -> dict[str, list[float]]:
        """JSON-RPC tools/call against Kineteq MCP (EmotiveVision-compatible)."""
        try:
            import httpx
        except ImportError as e:
            raise ImportError("Kineteq MCP client needs httpx (pip install autocausal[web])") from e

        url = (
            os.environ.get("AUTOCAUSAL_KINETEQ_MCP_URL")
            or os.environ.get("KINETEQ_MCP_URL")
            or os.environ.get("EMOTIVEVISION_MCP_URL")
            or ""
        ).strip()
        token = (
            os.environ.get("AUTOCAUSAL_KINETEQ_TOKEN")
            or os.environ.get("KINETEQ_AUTH_TOKEN")
            or os.environ.get("EMOTIVEVISION_MCP_TOKEN")
            or ""
        ).strip()
        texts = _texts_from_context(context)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if token:
            headers["x-api-key"] = token

        tool_names = [
            os.environ.get("AUTOCAUSAL_KINETEQ_EMBED_TOOL", "").strip() or "embed_texts",
            "pivot_embed",
            "embedding_pivot",
            "ml_embed",
        ]

        with httpx.Client(timeout=30.0) as client:
            # initialize (best-effort)
            try:
                client.post(
                    url,
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "autocausal", "version": "0.3.0"},
                        },
                    },
                )
            except Exception:
                pass

            last_err: Optional[Exception] = None
            for tool in tool_names:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": tool,
                        "arguments": {"texts": list(texts.values()), "items": texts},
                    },
                }
                try:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    if "error" in data:
                        last_err = RuntimeError(str(data["error"]))
                        continue
                    result = data.get("result")
                    parsed = self._parse_mcp_embeddings(result, list(texts.keys()))
                    if parsed:
                        return parsed
                except Exception as e:
                    last_err = e
                    continue
            if last_err:
                raise last_err
        return {}

    @staticmethod
    def _parse_mcp_embeddings(result: Any, keys: list[str]) -> dict[str, list[float]]:
        if result is None:
            return {}
        if isinstance(result, dict) and "content" in result:
            parts = result["content"]
            if parts and isinstance(parts[0], dict) and "text" in parts[0]:
                import json

                try:
                    result = json.loads(parts[0]["text"])
                except Exception:
                    return {}
        if isinstance(result, dict):
            if "embeddings" in result:
                emb = result["embeddings"]
                if isinstance(emb, dict):
                    return {k: list(map(float, emb[k])) for k in emb}
                if isinstance(emb, list) and len(emb) == len(keys):
                    return {keys[i]: list(map(float, emb[i])) for i in range(len(keys))}
            # key → vector map
            if all(isinstance(v, (list, tuple)) for v in result.values()):
                return {k: list(map(float, result[k])) for k in result}
        if isinstance(result, list) and len(result) == len(keys):
            return {keys[i]: list(map(float, result[i])) for i in range(len(keys))}
        return {}
