"""Local NLP, embedding, and vector-search adapters."""

from __future__ import annotations

import inspect
import re
from typing import Any, Optional, Sequence

from autocausal.integrations.adapters.base import LazyAdapter, bounded_int


class SpacyAdapter(LazyAdapter):
    id = "spacy.entities"
    integration_id = "spacy"
    module_name = "spacy"
    package_name = "spacy"
    capabilities = ("nlp.entities",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "nlp.entities":
            raise KeyError(capability)
        return self.entities(**kwargs)

    def entities(
        self,
        *,
        texts: Sequence[str],
        model: str = "en_core_web_sm",
        max_documents: int = 1_000,
        max_total_characters: int = 2_000_000,
        batch_size: int = 32,
        **_: Any,
    ) -> dict[str, Any]:
        documents = [str(item) for item in texts]
        if len(documents) > max_documents:
            raise ValueError(f"texts exceeds max_documents={max_documents}")
        if sum(len(item) for item in documents) > max_total_characters:
            raise ValueError("text payload exceeds max_total_characters")
        model_name = str(model)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", model_name):
            raise ValueError("spaCy model must be an installed package name")
        spacy = self._module()
        if not spacy.util.is_package(model_name):
            raise FileNotFoundError(
                f"spaCy model {model_name!r} is not installed; "
                "AutoCausal never downloads models automatically"
            )
        pipeline = spacy.load(
            model_name,
            disable=["parser", "textcat", "lemmatizer"],
        )
        results: list[list[dict[str, Any]]] = []
        for document in pipeline.pipe(
            documents,
            batch_size=bounded_int(
                batch_size,
                default=32,
                minimum=1,
                maximum=256,
                name="batch_size",
            ),
        ):
            results.append(
                [
                    {
                        "text": entity.text,
                        "label": entity.label_,
                        "start": int(entity.start_char),
                        "end": int(entity.end_char),
                    }
                    for entity in document.ents
                ]
            )
        return {
            "entities": results,
            "model": model_name,
            "n_documents": len(documents),
            "data_egress": False,
        }


class SentenceTransformersAdapter(LazyAdapter):
    id = "sentence-transformers.embeddings"
    integration_id = "sentence-transformers"
    module_name = "sentence_transformers"
    package_name = "sentence-transformers"
    capabilities = ("nlp.embeddings",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "nlp.embeddings":
            raise KeyError(capability)
        return self.embeddings(**kwargs)

    def embeddings(
        self,
        *,
        texts: Sequence[str],
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        allow_network: bool = False,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        max_documents: int = 5_000,
        max_total_characters: int = 5_000_000,
        **_: Any,
    ) -> dict[str, Any]:
        documents = [str(item) for item in texts]
        if len(documents) > max_documents:
            raise ValueError(f"texts exceeds max_documents={max_documents}")
        if sum(len(item) for item in documents) > max_total_characters:
            raise ValueError("text payload exceeds max_total_characters")
        model_name = str(model)
        if ".." in model_name or not re.fullmatch(r"[A-Za-z0-9_./-]+", model_name):
            raise ValueError("invalid sentence-transformer model identifier")
        selected_device = str(device).lower()
        if selected_device != "cpu":
            raise ValueError(
                "this adapter is CPU-safe by default; GPU routing requires a custom plugin"
            )
        module = self._module()
        constructor = module.SentenceTransformer
        signature = inspect.signature(constructor)
        constructor_kwargs: dict[str, Any] = {"device": "cpu"}
        if "local_files_only" in signature.parameters:
            constructor_kwargs["local_files_only"] = not bool(allow_network)
        elif not allow_network:
            raise RuntimeError(
                "installed sentence-transformers cannot guarantee local-only loading"
            )
        if "trust_remote_code" in signature.parameters:
            constructor_kwargs["trust_remote_code"] = False
        encoder = constructor(model_name, **constructor_kwargs)
        matrix = encoder.encode(
            documents,
            batch_size=bounded_int(
                batch_size,
                default=32,
                minimum=1,
                maximum=128,
                name="batch_size",
            ),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=bool(normalize),
        )
        return {
            "embeddings": matrix,
            "shape": tuple(matrix.shape),
            "model": model_name,
            "device": "cpu",
            "network_allowed": bool(allow_network),
            "data_egress": False,
        }


class FaissAdapter(LazyAdapter):
    id = "faiss.vector-search"
    integration_id = "faiss"
    module_name = "faiss"
    package_name = "faiss-cpu"
    capabilities = ("nlp.vector_search",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "nlp.vector_search":
            raise KeyError(capability)
        return self.search(**kwargs)

    def search(
        self,
        *,
        vectors: Any,
        queries: Any,
        k: int = 5,
        metric: str = "cosine",
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np

        faiss = self._module()
        base = np.asarray(vectors, dtype="float32").copy()
        query = np.asarray(queries, dtype="float32").copy()
        if base.ndim != 2 or query.ndim != 2 or base.shape[1] != query.shape[1]:
            raise ValueError("vectors and queries must be compatible 2D matrices")
        if len(base) > 2_000_000 or len(query) > 10_000:
            raise ValueError("FAISS request exceeds bounded in-memory limits")
        selected = str(metric).lower()
        if selected == "cosine":
            faiss.normalize_L2(base)
            faiss.normalize_L2(query)
            index = faiss.IndexFlatIP(base.shape[1])
        elif selected in ("l2", "euclidean"):
            index = faiss.IndexFlatL2(base.shape[1])
        else:
            raise ValueError("metric must be cosine or l2")
        index.add(base)
        count = min(
            bounded_int(k, default=5, minimum=1, maximum=100, name="k"),
            len(base),
        )
        scores, indices = index.search(query, count)
        return {
            "indices": indices,
            "scores": scores,
            "metric": selected,
            "backend": "faiss-cpu",
            "persistent": False,
        }


class ChromaAdapter(LazyAdapter):
    id = "chromadb.local-vector-search"
    integration_id = "chromadb"
    module_name = "chromadb"
    package_name = "chromadb"
    capabilities = ("nlp.vector_search",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "nlp.vector_search":
            raise KeyError(capability)
        return self.search(**kwargs)

    def search(
        self,
        *,
        vectors: Any,
        queries: Any,
        ids: Optional[Sequence[str]] = None,
        k: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np

        chromadb = self._module()
        base = np.asarray(vectors, dtype=float)
        query = np.asarray(queries, dtype=float)
        if base.ndim != 2 or query.ndim != 2 or base.shape[1] != query.shape[1]:
            raise ValueError("vectors and queries must be compatible 2D matrices")
        if len(base) > 100_000 or len(query) > 1_000:
            raise ValueError("Chroma request exceeds local ephemeral limits")
        item_ids = (
            [str(item) for item in ids]
            if ids is not None
            else [f"row-{index}" for index in range(len(base))]
        )
        if len(item_ids) != len(base):
            raise ValueError("ids length must equal vectors length")
        client = chromadb.EphemeralClient()
        collection = client.create_collection(
            name="autocausal-ephemeral",
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(ids=item_ids, embeddings=base.tolist())
        count = min(
            bounded_int(k, default=5, minimum=1, maximum=100, name="k"),
            len(base),
        )
        result = collection.query(
            query_embeddings=query.tolist(),
            n_results=count,
            include=["distances"],
        )
        return {
            "ids": result.get("ids"),
            "distances": result.get("distances"),
            "backend": "chromadb-ephemeral",
            "persistent": False,
            "remote_client_allowed": False,
        }


def nlp_adapters() -> tuple[LazyAdapter, ...]:
    return (
        SpacyAdapter(),
        SentenceTransformersAdapter(),
        FaissAdapter(),
        ChromaAdapter(),
    )


__all__ = [
    "ChromaAdapter",
    "FaissAdapter",
    "SentenceTransformersAdapter",
    "SpacyAdapter",
    "nlp_adapters",
]
