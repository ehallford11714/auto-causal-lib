"""VectorStoreMemory — hybrid LTM soft store (HippoRAG/Mem0-inspired).

Design inspiration (not a reimplementation):
- HippoRAG (arXiv:2405.14831) — retrieval over structured + vector memory
- Mem0 (arXiv:2504.19413) — persistent agent memory layer

Default backend is an in-memory TF-IDF / bag-of-words cosine store using only
numpy (always available). Optional soft backends: chromadb, faiss — never
hard-crash if missing.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

__all__ = ["VectorRecord", "VectorStoreMemory", "make_vector_memory"]


_TOKEN = re.compile(r"[a-z0-9_]+", re.I)


@dataclass
class VectorRecord:
    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    kind: str = "insight"  # insight | experiment | episode | edge

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if len(t) > 1]


class VectorStoreMemory:
    """Soft vector memory with numpy TF-IDF fallback.

    ``backend`` is one of: ``numpy`` (default), ``chromadb``, ``faiss``.
    Unavailable backends silently fall back to numpy.
    """

    def __init__(
        self,
        *,
        backend: str = "auto",
        collection: str = "autocausal_agentic",
        max_records: int = 500,
    ) -> None:
        self.collection = collection
        self.max_records = max(10, int(max_records))
        self.records: list[VectorRecord] = []
        self._df: dict[str, int] = {}
        self._backend_name = "numpy"
        self._chroma = None
        self._faiss_index = None
        self._faiss_vectors: list[list[float]] = []
        self._resolve_backend(backend)

    def _resolve_backend(self, backend: str) -> None:
        want = (backend or "auto").lower().strip()
        if want in ("auto", "chromadb", "chroma"):
            if self._try_chromadb():
                return
            if want in ("chromadb", "chroma"):
                pass  # fall through
        if want in ("auto", "faiss"):
            if self._try_faiss():
                return
        self._backend_name = "numpy"

    def _try_chromadb(self) -> bool:
        try:
            import chromadb  # type: ignore

            client = chromadb.Client()
            self._chroma = client.get_or_create_collection(self.collection)
            self._backend_name = "chromadb"
            return True
        except Exception:
            self._chroma = None
            return False

    def _try_faiss(self) -> bool:
        try:
            import faiss  # type: ignore
            import numpy as np

            # Deferred index — built on first add with known dim
            self._faiss = faiss
            self._np = np
            self._backend_name = "faiss"
            return True
        except Exception:
            self._faiss = None
            return False

    @property
    def backend(self) -> str:
        return self._backend_name

    def __len__(self) -> int:
        return len(self.records)

    def add(
        self,
        text: str,
        *,
        kind: str = "insight",
        meta: Optional[dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> VectorRecord:
        rid = id or f"vec-{len(self.records):04d}"
        rec = VectorRecord(id=rid, text=text, meta=dict(meta or {}), kind=kind)
        self.records.append(rec)
        self._update_df(rec.text)
        if self._backend_name == "chromadb" and self._chroma is not None:
            try:
                self._chroma.add(
                    ids=[rid],
                    documents=[text],
                    metadatas=[{"kind": kind, **{k: str(v) for k, v in (meta or {}).items()}}],
                )
            except Exception:
                pass
        elif self._backend_name == "faiss":
            self._faiss_add(rec)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]
        return rec

    def _update_df(self, text: str) -> None:
        seen = set(_tokenize(text))
        for t in seen:
            self._df[t] = self._df.get(t, 0) + 1

    def _tfidf(self, text: str) -> dict[str, float]:
        toks = _tokenize(text)
        if not toks:
            return {}
        tf: dict[str, float] = {}
        for t in toks:
            tf[t] = tf.get(t, 0.0) + 1.0
        n = float(len(toks))
        N = max(1, len(self.records))
        vec: dict[str, float] = {}
        for t, c in tf.items():
            idf = math.log((N + 1) / (1 + self._df.get(t, 0))) + 1.0
            vec[t] = (c / n) * idf
        return vec

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na <= 0 or nb <= 0:
            return 0.0
        return float(dot / (na * nb))

    def _faiss_add(self, rec: VectorRecord) -> None:
        # Dense bag projection into fixed dim via hashing
        try:
            import numpy as np

            dim = 64
            v = np.zeros(dim, dtype="float32")
            for t, w in self._tfidf(rec.text).items():
                v[hash(t) % dim] += float(w)
            n = float(np.linalg.norm(v))
            if n > 0:
                v /= n
            self._faiss_vectors.append(v.tolist())
            if self._faiss_index is None:
                index = self._faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
                self._faiss_index = index
            self._faiss_index.add(np.asarray([v], dtype="float32"))  # type: ignore[union-attr]
        except Exception:
            pass

    def query(self, text: str, *, k: int = 5, kind: Optional[str] = None) -> list[dict[str, Any]]:
        k = max(1, int(k))
        if self._backend_name == "chromadb" and self._chroma is not None:
            try:
                res = self._chroma.query(query_texts=[text], n_results=k)
                out = []
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                ids = (res.get("ids") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                for i, doc in enumerate(docs):
                    out.append(
                        {
                            "id": ids[i] if i < len(ids) else f"c{i}",
                            "text": doc,
                            "score": 1.0 / (1.0 + float(dists[i] if i < len(dists) else 1.0)),
                            "meta": metas[i] if i < len(metas) else {},
                            "backend": "chromadb",
                        }
                    )
                return out
            except Exception:
                pass

        q = self._tfidf(text)
        scored: list[tuple[float, VectorRecord]] = []
        for rec in self.records:
            if kind and rec.kind != kind:
                continue
            s = self._cosine(q, self._tfidf(rec.text))
            scored.append((s, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": rec.id,
                "text": rec.text,
                "score": float(score),
                "kind": rec.kind,
                "meta": dict(rec.meta),
                "backend": self._backend_name,
            }
            for score, rec in scored[:k]
            if score > 0
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self._backend_name,
            "collection": self.collection,
            "n": len(self.records),
            "records": [r.to_dict() for r in self.records[-50:]],
        }


def make_vector_memory(*, backend: str = "auto", **kwargs: Any) -> VectorStoreMemory:
    return VectorStoreMemory(backend=backend, **kwargs)
