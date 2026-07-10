"""Soft-optional NLTK backend detection and corpus download helpers.

NLTK is never a hard dependency. All helpers soft-fail when the package or
corpora are missing so core autocausal installs keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# Corpora / models commonly needed by autocausal.nlp
DEFAULT_RESOURCES = (
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "vader_lexicon",
    "wordnet",
    "omw-1.4",
    "stopwords",
)


@dataclass
class NltkStatus:
    """Availability snapshot for NLTK + selected resources."""

    installed: bool
    version: Optional[str] = None
    resources: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "version": self.version,
            "resources": dict(self.resources),
            "notes": list(self.notes),
        }


def soft_import_nltk() -> Any:
    """Return the ``nltk`` module or ``None`` if not installed."""
    try:
        import nltk  # type: ignore

        return nltk
    except ImportError:
        return None


def nltk_available() -> bool:
    return soft_import_nltk() is not None


def _resource_find_paths(name: str) -> list[str]:
    """Candidate ``nltk.data.find`` paths for a resource name."""
    mapping = {
        "punkt": ["tokenizers/punkt"],
        "punkt_tab": ["tokenizers/punkt_tab"],
        "averaged_perceptron_tagger": ["taggers/averaged_perceptron_tagger"],
        "averaged_perceptron_tagger_eng": ["taggers/averaged_perceptron_tagger_eng"],
        "vader_lexicon": ["sentiment/vader_lexicon.zip", "sentiment/vader_lexicon"],
        "wordnet": ["corpora/wordnet", "corpora/wordnet.zip"],
        "omw-1.4": ["corpora/omw-1.4", "corpora/omw-1.4.zip"],
        "stopwords": ["corpora/stopwords"],
    }
    return mapping.get(name, [name])


def resource_available(name: str, nltk: Any = None) -> bool:
    """Return True if an NLTK resource can be found locally (no download)."""
    nltk = nltk if nltk is not None else soft_import_nltk()
    if nltk is None:
        return False
    for path in _resource_find_paths(name):
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            continue
        except Exception:
            continue
    return False


def ensure_nltk_data(
    resources: Optional[tuple[str, ...] | list[str]] = None,
    *,
    quiet: bool = True,
) -> dict[str, Any]:
    """Attempt to download NLTK resources; soft-fail per resource (no raise).

    Returns a report dict::

        {"ok": bool, "installed": bool, "downloaded": [...], "failed": [...], "notes": [...]}
    """
    names = tuple(resources) if resources is not None else DEFAULT_RESOURCES
    nltk = soft_import_nltk()
    notes: list[str] = []
    downloaded: list[str] = []
    failed: list[str] = []
    already: list[str] = []

    if nltk is None:
        return {
            "ok": False,
            "installed": False,
            "downloaded": [],
            "failed": list(names),
            "already": [],
            "notes": ["nltk not installed; pip install 'autocausal[nlp]'"],
        }

    for name in names:
        if resource_available(name, nltk):
            already.append(name)
            continue
        try:
            ok = bool(nltk.download(name, quiet=quiet))
            if ok or resource_available(name, nltk):
                downloaded.append(name)
            else:
                failed.append(name)
                notes.append(f"download returned false for {name}")
        except Exception as e:
            failed.append(name)
            notes.append(f"{name} download soft-fail: {type(e).__name__}: {e}")

    return {
        "ok": len(failed) == 0,
        "installed": True,
        "downloaded": downloaded,
        "failed": failed,
        "already": already,
        "notes": notes,
    }


def nltk_status(resources: Optional[tuple[str, ...] | list[str]] = None) -> NltkStatus:
    """Inspect NLTK install + local resource availability (no network)."""
    names = tuple(resources) if resources is not None else DEFAULT_RESOURCES
    nltk = soft_import_nltk()
    if nltk is None:
        return NltkStatus(
            installed=False,
            notes=["nltk not installed; regex/lexicon fallbacks will be used"],
        )
    version = getattr(nltk, "__version__", None)
    res = {name: resource_available(name, nltk) for name in names}
    notes: list[str] = []
    missing = [k for k, v in res.items() if not v]
    if missing:
        notes.append(
            "missing corpora (optional): "
            + ", ".join(missing)
            + " — call ensure_nltk_data() or use fallbacks"
        )
    return NltkStatus(installed=True, version=str(version) if version else None, resources=res, notes=notes)
