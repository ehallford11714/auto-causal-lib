"""Retrieval provider interfaces and offline-testable metadata adapters."""

from __future__ import annotations

import ipaddress
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence
from urllib.parse import urlparse

from autocausal.research.models import ResearchPolicy, SourceRecord


class ProviderError(RuntimeError):
    """Retrieval failed without inventing a source."""


@dataclass
class ProviderQuery:
    query: str
    limit: int
    publication_year_min: Optional[int] = None
    publication_year_max: Optional[int] = None
    languages: list[str] = field(default_factory=lambda: ["en"])
    question_id: str = ""
    round_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "limit": self.limit,
            "publication_year_min": self.publication_year_min,
            "publication_year_max": self.publication_year_max,
            "languages": list(self.languages),
            "question_id": self.question_id,
            "round_index": self.round_index,
        }


class ResearchProvider(Protocol):
    name: str
    network: bool

    def search(
        self,
        request: ProviderQuery,
        *,
        policy: ResearchPolicy,
    ) -> list[SourceRecord]: ...


def _cache_key(provider: str, request: ProviderQuery) -> str:
    blob = json.dumps(
        {
            "provider": provider,
            "query": " ".join(request.query.lower().split()),
            "limit": request.limit,
            "publication_year_min": request.publication_year_min,
            "publication_year_max": request.publication_year_max,
            "languages": sorted(request.languages),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(blob.encode("utf-8")).hexdigest()


class ResearchCache:
    """Small deterministic metadata cache; never stores credentials."""

    def __init__(self, directory: Optional[str | Path] = None) -> None:
        self.directory = Path(directory) if directory else None
        self.memory: dict[str, list[dict[str, Any]]] = {}
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    def get(
        self, provider: str, request: ProviderQuery
    ) -> Optional[list[SourceRecord]]:
        key = _cache_key(provider, request)
        payload = self.memory.get(key)
        if payload is None and self.directory is not None:
            path = self.directory / f"{key}.json"
            try:
                if path.is_file():
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    payload = list(raw.get("sources") or [])
                    self.memory[key] = payload
            except Exception:
                payload = None
        if payload is None:
            return None
        return [SourceRecord.from_dict(item) for item in payload]

    def put(
        self,
        provider: str,
        request: ProviderQuery,
        sources: Sequence[SourceRecord],
    ) -> None:
        key = _cache_key(provider, request)
        payload = [source.to_dict() for source in sources]
        self.memory[key] = payload
        if self.directory is not None:
            path = self.directory / f"{key}.json"
            body = {
                "provider": provider,
                "request": request.to_dict(),
                "sources": payload,
                "contains_secrets": False,
            }
            path.write_text(
                json.dumps(body, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )


def _valid_public_url(url: str, allowed_hosts: Sequence[str]) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            return False
        host = parsed.hostname.lower().rstrip(".")
        if not any(host == item or host.endswith("." + item) for item in allowed_hosts):
            return False
        if host in ("localhost", "localhost.localdomain"):
            return False
        try:
            ip = ipaddress.ip_address(host)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            ):
                return False
        except ValueError:
            pass
        return True
    except Exception:
        return False


class _HTTPProvider:
    name = "http"
    network = True
    base_url = ""

    def _get(
        self,
        url: str,
        *,
        params: Mapping[str, Any],
        policy: ResearchPolicy,
    ) -> bytes:
        allowed = tuple(policy.provider_domains.get(self.name) or ())
        if not allowed or not _valid_public_url(url, allowed):
            raise ProviderError(f"{self.name}: URL/domain rejected by provider policy")
        if not policy.permits_provider(self.name, network=True):
            raise ProviderError(
                f"{self.name}: network/provider not allowed or consent missing"
            )
        try:
            import httpx
        except Exception as exc:
            raise ProviderError("httpx is required for network providers") from exc

        last_error = ""
        for attempt in range(policy.retry_attempts + 1):
            try:
                response = httpx.get(
                    url,
                    params=dict(params),
                    headers={
                        "User-Agent": policy.user_agent,
                        "Accept": "application/json, application/atom+xml, application/xml",
                    },
                    timeout=policy.request_timeout_seconds,
                    follow_redirects=False,
                )
                if response.status_code in (301, 302, 307, 308):
                    raise ProviderError(
                        f"{self.name}: redirects are refused for domain safety"
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    raise ProviderError(
                        f"{self.name}: retryable HTTP {response.status_code}"
                    )
                response.raise_for_status()
                content = bytes(response.content)
                size_limit = min(
                    policy.response_size_limit_bytes,
                    policy.maximum_budget.max_bytes,
                )
                if len(content) > size_limit:
                    raise ProviderError(
                        f"{self.name}: response {len(content)} bytes exceeds "
                        f"limit {size_limit}"
                    )
                self.last_response_bytes = len(content)
                return content
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= policy.retry_attempts:
                    break
                time.sleep(min(2.0, 0.25 * (2**attempt)))
        raise ProviderError(f"{self.name} retrieval failed: {last_error}")

    @staticmethod
    def _year_allowed(date: Optional[str], request: ProviderQuery) -> bool:
        if not date:
            return True
        match = re.search(r"\b(19|20)\d{2}\b", str(date))
        if not match:
            return True
        year = int(match.group(0))
        if (
            request.publication_year_min is not None
            and year < request.publication_year_min
        ):
            return False
        if (
            request.publication_year_max is not None
            and year > request.publication_year_max
        ):
            return False
        return True


class ArxivProvider(_HTTPProvider):
    """Official arXiv Atom API metadata adapter."""

    name = "arxiv"
    base_url = "https://export.arxiv.org/api/query"

    def search(
        self, request: ProviderQuery, *, policy: ResearchPolicy
    ) -> list[SourceRecord]:
        content = self._get(
            self.base_url,
            params={
                "search_query": f"all:{request.query}",
                "start": 0,
                "max_results": request.limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            policy=policy,
        )
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            raise ProviderError(f"arxiv XML parse failed: {exc}") from exc
        atom = "{http://www.w3.org/2005/Atom}"
        arxiv_ns = "{http://arxiv.org/schemas/atom}"
        out: list[SourceRecord] = []
        for entry in root.findall(f"{atom}entry"):
            id_url = (entry.findtext(f"{atom}id") or "").strip()
            arxiv_id = id_url.rstrip("/").split("/")[-1]
            title = " ".join((entry.findtext(f"{atom}title") or "").split())
            abstract = " ".join((entry.findtext(f"{atom}summary") or "").split())
            date = (entry.findtext(f"{atom}published") or "").strip() or None
            if not title or not arxiv_id or not self._year_allowed(date, request):
                continue
            authors = [
                " ".join((author.findtext(f"{atom}name") or "").split())
                for author in entry.findall(f"{atom}author")
            ]
            doi = (entry.findtext(f"{arxiv_ns}doi") or "").strip() or None
            out.append(
                SourceRecord(
                    provider=self.name,
                    stable_id=arxiv_id,
                    arxiv_id=arxiv_id,
                    doi=doi,
                    url=id_url or f"https://arxiv.org/abs/{arxiv_id}",
                    title=title,
                    authors=authors,
                    date=date,
                    abstract=abstract or None,
                    availability="abstract" if abstract else "metadata",
                    license=None,
                    metadata={"endpoint": "arxiv_atom_api"},
                )
            )
        return out[: request.limit]


class CrossrefProvider(_HTTPProvider):
    """Official Crossref REST metadata adapter."""

    name = "crossref"
    base_url = "https://api.crossref.org/works"

    def search(
        self, request: ProviderQuery, *, policy: ResearchPolicy
    ) -> list[SourceRecord]:
        params: dict[str, Any] = {
            "query.bibliographic": request.query,
            "rows": request.limit,
            "select": (
                "DOI,title,author,published-print,published-online,created,"
                "abstract,URL,license,container-title,reference,language"
            ),
        }
        filters: list[str] = []
        if request.publication_year_min is not None:
            filters.append(f"from-pub-date:{request.publication_year_min}-01-01")
        if request.publication_year_max is not None:
            filters.append(f"until-pub-date:{request.publication_year_max}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        content = self._get(self.base_url, params=params, policy=policy)
        try:
            items = json.loads(content).get("message", {}).get("items", [])
        except Exception as exc:
            raise ProviderError(f"crossref JSON parse failed: {exc}") from exc
        out: list[SourceRecord] = []
        for item in items:
            doi = str(item.get("DOI") or "").strip().lower()
            titles = item.get("title") or []
            title = " ".join(str(titles[0] if titles else "").split())
            if not doi or not title:
                continue
            language = str(item.get("language") or "").lower() or None
            if language and request.languages and language not in request.languages:
                continue
            date_parts: list[Any] = []
            for key in ("published-print", "published-online", "created"):
                candidate = item.get(key) or {}
                parts = candidate.get("date-parts") or []
                if parts:
                    date_parts = list(parts[0])
                    break
            date = "-".join(str(value) for value in date_parts) or None
            if not self._year_allowed(date, request):
                continue
            authors = []
            for author in item.get("author") or []:
                name = " ".join(
                    part
                    for part in (
                        str(author.get("given") or "").strip(),
                        str(author.get("family") or "").strip(),
                    )
                    if part
                )
                if name:
                    authors.append(name)
            abstract = re.sub(r"<[^>]+>", " ", str(item.get("abstract") or ""))
            abstract = " ".join(abstract.split()) or None
            licenses = item.get("license") or []
            license_url = str(licenses[0].get("URL") or "") if licenses else None
            references = [
                str(ref.get("DOI") or "").lower()
                for ref in item.get("reference") or []
                if ref.get("DOI")
            ]
            venue_values = item.get("container-title") or []
            out.append(
                SourceRecord(
                    provider=self.name,
                    stable_id=doi,
                    doi=doi,
                    url=str(item.get("URL") or f"https://doi.org/{doi}"),
                    title=title,
                    authors=authors,
                    date=date,
                    abstract=abstract,
                    availability="abstract" if abstract else "metadata",
                    license=license_url,
                    language=language,
                    venue=str(venue_values[0]) if venue_values else None,
                    references=references,
                    metadata={"endpoint": "crossref_rest"},
                )
            )
        return out[: request.limit]


def _openalex_abstract(inverted: Any) -> Optional[str]:
    if not isinstance(inverted, Mapping):
        return None
    positioned: list[tuple[int, str]] = []
    for token, positions in inverted.items():
        for position in positions or []:
            try:
                positioned.append((int(position), str(token)))
            except (TypeError, ValueError):
                continue
    if not positioned:
        return None
    return " ".join(token for _, token in sorted(positioned))


class OpenAlexProvider(_HTTPProvider):
    """Official OpenAlex works API metadata adapter."""

    name = "openalex"
    base_url = "https://api.openalex.org/works"

    def search(
        self, request: ProviderQuery, *, policy: ResearchPolicy
    ) -> list[SourceRecord]:
        filters: list[str] = []
        if request.publication_year_min is not None:
            filters.append(
                f"from_publication_date:{request.publication_year_min}-01-01"
            )
        if request.publication_year_max is not None:
            filters.append(f"to_publication_date:{request.publication_year_max}-12-31")
        params: dict[str, Any] = {
            "search": request.query,
            "per-page": request.limit,
        }
        if filters:
            params["filter"] = ",".join(filters)
        content = self._get(self.base_url, params=params, policy=policy)
        try:
            items = json.loads(content).get("results", [])
        except Exception as exc:
            raise ProviderError(f"openalex JSON parse failed: {exc}") from exc
        out: list[SourceRecord] = []
        for item in items:
            openalex_id = str(item.get("id") or "").rstrip("/").split("/")[-1]
            title = " ".join(str(item.get("display_name") or "").split())
            if not openalex_id or not title:
                continue
            language = str(item.get("language") or "").lower() or None
            if language and request.languages and language not in request.languages:
                continue
            date = str(item.get("publication_date") or "") or None
            if not self._year_allowed(date, request):
                continue
            ids = item.get("ids") or {}
            doi = str(ids.get("doi") or "").replace("https://doi.org/", "") or None
            authors = [
                str((entry.get("author") or {}).get("display_name") or "").strip()
                for entry in item.get("authorships") or []
            ]
            authors = [item for item in authors if item]
            abstract = _openalex_abstract(item.get("abstract_inverted_index"))
            primary = item.get("primary_location") or {}
            source = primary.get("source") or {}
            references = [
                str(ref).rstrip("/").split("/")[-1]
                for ref in item.get("referenced_works") or []
            ]
            out.append(
                SourceRecord(
                    provider=self.name,
                    stable_id=openalex_id,
                    doi=doi,
                    url=str(primary.get("landing_page_url") or item.get("id") or "")
                    or None,
                    title=title,
                    authors=authors,
                    date=date,
                    abstract=abstract,
                    availability="abstract" if abstract else "metadata",
                    license=str(primary.get("license") or "") or None,
                    language=language,
                    venue=str(source.get("display_name") or "") or None,
                    references=references,
                    metadata={
                        "endpoint": "openalex_works",
                        "study_type": str(item.get("type") or ""),
                    },
                )
            )
        return out[: request.limit]


class SemanticScholarProvider(_HTTPProvider):
    """Soft, low-volume Semantic Scholar Graph API adapter (no key required)."""

    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(
        self, request: ProviderQuery, *, policy: ResearchPolicy
    ) -> list[SourceRecord]:
        # Keep unauthenticated use deliberately small and rely on backoff.
        limit = min(20, request.limit)
        content = self._get(
            self.base_url,
            params={
                "query": request.query,
                "limit": limit,
                "fields": (
                    "paperId,title,authors,year,abstract,url,externalIds,"
                    "openAccessPdf,venue,publicationDate"
                ),
            },
            policy=policy,
        )
        try:
            items = json.loads(content).get("data", [])
        except Exception as exc:
            raise ProviderError(f"semantic_scholar JSON parse failed: {exc}") from exc
        out: list[SourceRecord] = []
        for item in items:
            paper_id = str(item.get("paperId") or "")
            title = " ".join(str(item.get("title") or "").split())
            if not paper_id or not title:
                continue
            date = str(item.get("publicationDate") or item.get("year") or "") or None
            if not self._year_allowed(date, request):
                continue
            external = item.get("externalIds") or {}
            abstract = " ".join(str(item.get("abstract") or "").split()) or None
            pdf = item.get("openAccessPdf") or {}
            out.append(
                SourceRecord(
                    provider=self.name,
                    stable_id=paper_id,
                    doi=str(external.get("DOI") or "") or None,
                    arxiv_id=str(external.get("ArXiv") or "") or None,
                    url=str(item.get("url") or "") or None,
                    title=title,
                    authors=[
                        str(author.get("name") or "")
                        for author in item.get("authors") or []
                        if author.get("name")
                    ],
                    date=date,
                    abstract=abstract,
                    availability="abstract" if abstract else "metadata",
                    license="open-access" if pdf.get("url") else None,
                    venue=str(item.get("venue") or "") or None,
                    metadata={
                        "endpoint": "semantic_scholar_graph",
                        "open_access_pdf_url": str(pdf.get("url") or "") or None,
                    },
                )
            )
        return out


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.split(r"[^a-z0-9]+", str(text).lower()) if len(token) > 1
    }


class LocalDocumentProvider:
    """Search supplied records, JSONL/doc files, or a soft vector-store adapter."""

    name = "local"
    network = False

    def __init__(
        self,
        records: Optional[Iterable[SourceRecord | Mapping[str, Any]]] = None,
        *,
        paths: Optional[Sequence[str | Path]] = None,
        vector_store: Any = None,
    ) -> None:
        self.records: list[SourceRecord] = []
        self.vector_store = vector_store
        for item in records or []:
            self.records.append(
                item if isinstance(item, SourceRecord) else SourceRecord.from_dict(item)
            )
        for path_value in paths or []:
            self._load_path(Path(path_value))

    def _load_path(self, path: Path) -> None:
        if not path.is_file():
            return
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    self.records.append(SourceRecord.from_dict(item))
                except Exception:
                    continue
        elif suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                items = (
                    payload
                    if isinstance(payload, list)
                    else payload.get("sources") or [payload]
                )
                for item in items:
                    self.records.append(SourceRecord.from_dict(item))
            except Exception:
                return
        elif suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")[:2_000_000]
            title = next(
                (
                    line.lstrip("# ").strip()
                    for line in text.splitlines()
                    if line.strip()
                ),
                path.stem,
            )
            self.records.append(
                SourceRecord(
                    provider=self.name,
                    stable_id=f"file:{sha256(str(path.resolve()).encode()).hexdigest()[:16]}",
                    title=title[:300],
                    snippet=text[:10_000],
                    availability="user_supplied",
                    metadata={"local_path_name": path.name},
                )
            )

    def _vector_results(self, request: ProviderQuery) -> list[SourceRecord]:
        if self.vector_store is None:
            return []
        method = getattr(self.vector_store, "query", None) or getattr(
            self.vector_store, "search", None
        )
        if not callable(method):
            return []
        try:
            raw = method(request.query, k=request.limit)
        except TypeError:
            raw = method(request.query, request.limit)
        out: list[SourceRecord] = []
        for index, item in enumerate(raw or []):
            if isinstance(item, SourceRecord):
                out.append(item)
                continue
            row = dict(item) if isinstance(item, Mapping) else {"text": str(item)}
            if row.get("title") and (
                row.get("stable_id") or row.get("doi") or row.get("url")
            ):
                row.setdefault("provider", self.name)
                out.append(SourceRecord.from_dict(row))
                continue
            text = str(row.get("text") or row.get("document") or "").strip()
            if text:
                out.append(
                    SourceRecord(
                        provider=self.name,
                        stable_id=f"vector:{sha256(text.encode()).hexdigest()[:16]}",
                        title=str(
                            row.get("title") or f"Local vector result {index + 1}"
                        ),
                        snippet=text[:10_000],
                        availability="user_supplied",
                        metadata={"vector_score": row.get("score")},
                    )
                )
        return out

    def search(
        self, request: ProviderQuery, *, policy: ResearchPolicy
    ) -> list[SourceRecord]:
        if not policy.permits_provider(self.name, network=False):
            raise ProviderError("local provider is not allowed by policy")
        query = str(request.query or "")
        query_tokens = _tokens(query)
        query_lower = query.lower()
        doi_query = None
        arxiv_query = None
        doi_match = re.search(r"\b10\.\d{4,9}/\S+", query_lower)
        if doi_match:
            doi_query = doi_match.group(0).rstrip(".")
        arxiv_match = re.search(
            r"(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7})",
            query_lower,
        )
        if arxiv_match:
            arxiv_query = arxiv_match.group(1)
        candidates = list(self.records) + self._vector_results(request)
        ranked: list[tuple[float, str, SourceRecord]] = []
        for record in candidates:
            year_ok = _HTTPProvider._year_allowed(record.date, request)
            if not year_ok:
                continue
            if record.language and request.languages:
                if record.language.lower() not in request.languages:
                    continue
            text_tokens = _tokens(
                f"{record.title} {record.abstract or ''} {record.snippet or ''}"
            )
            overlap = len(query_tokens & text_tokens)
            score = overlap / max(1, len(query_tokens))
            # Prefer exact DOI/arXiv hits from related-work identifier expansion.
            blob = (
                f"{record.title} {record.abstract or ''} {record.snippet or ''}"
            ).lower()
            record_doi = str(record.doi or "").lower().rstrip(".")
            record_arxiv = str(record.arxiv_id or "").lower()
            if doi_query and (record_doi == doi_query or doi_query in blob):
                score = max(score, 1.0)
            if arxiv_query and (
                record_arxiv == arxiv_query or arxiv_query in blob
            ):
                score = max(score, 1.0)
            if score > 0 or not query_tokens:
                ranked.append((score, record.source_id, record))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        return [record for _, _, record in ranked[: request.limit]]


class GenericWebSearchProvider:
    """Explicit callback adapter; it never scrapes or invents citations itself."""

    name = "generic_web"
    network = True

    def __init__(
        self,
        search_callback: Callable[
            [str, int], Sequence[SourceRecord | Mapping[str, Any]]
        ],
    ) -> None:
        self.search_callback = search_callback

    def search(
        self, request: ProviderQuery, *, policy: ResearchPolicy
    ) -> list[SourceRecord]:
        if not (
            policy.allow_generic_web
            and policy.permits_provider(self.name, network=True)
        ):
            raise ProviderError(
                "generic web search requires explicit provider, network, and consent policy"
            )
        raw = self.search_callback(request.query, request.limit)
        out: list[SourceRecord] = []
        for item in raw:
            record = (
                item if isinstance(item, SourceRecord) else SourceRecord.from_dict(item)
            )
            if record.url:
                parsed = urlparse(record.url)
                if parsed.scheme not in ("https",) or not parsed.hostname:
                    continue
                host = parsed.hostname.lower()
                if host in ("localhost", "127.0.0.1", "::1"):
                    continue
            out.append(record)
        return out[: request.limit]


def default_provider(
    name: str,
    *,
    local_records: Optional[Iterable[SourceRecord | Mapping[str, Any]]] = None,
) -> ResearchProvider:
    normalized = str(name).strip().lower()
    if normalized == "local":
        return LocalDocumentProvider(local_records)
    if normalized == "arxiv":
        return ArxivProvider()
    if normalized == "crossref":
        return CrossrefProvider()
    if normalized == "openalex":
        return OpenAlexProvider()
    if normalized in ("semantic_scholar", "semanticscholar"):
        return SemanticScholarProvider()
    raise KeyError(f"unknown research provider {name!r}")


__all__ = [
    "ArxivProvider",
    "CrossrefProvider",
    "GenericWebSearchProvider",
    "LocalDocumentProvider",
    "OpenAlexProvider",
    "ProviderError",
    "ProviderQuery",
    "ResearchCache",
    "ResearchProvider",
    "SemanticScholarProvider",
    "default_provider",
]
