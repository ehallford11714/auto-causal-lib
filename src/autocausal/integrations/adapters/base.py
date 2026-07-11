"""Standard-library-only adapter helpers."""

from __future__ import annotations

import importlib
from importlib import metadata
from typing import Any

from autocausal.integrations.types import ProbeResult


PREDICTIVE_CAVEAT = (
    "Predictive performance or feature attribution does not establish a causal effect."
)
CAUSAL_CAVEAT = (
    "Estimator availability does not establish identification; review design assumptions."
)


def bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
    name: str,
) -> int:
    resolved = default if value is None else int(value)
    if resolved < minimum or resolved > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return resolved


def as_2d_controls(controls: Any, n: int) -> Any:
    import numpy as np

    if controls is None:
        return np.empty((n, 0), dtype=float)
    value = np.asarray(controls, dtype=float)
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    if value.ndim != 2 or value.shape[0] != n:
        raise ValueError("controls must have shape (n_rows, n_controls)")
    return value


def residualize(values: Any, controls: Any) -> tuple[Any, Any]:
    import numpy as np

    vector = np.asarray(values, dtype=float).reshape(-1)
    matrix = as_2d_controls(controls, len(vector))
    finite = np.isfinite(vector)
    if matrix.shape[1]:
        finite &= np.isfinite(matrix).all(axis=1)
    if finite.sum() < max(3, matrix.shape[1] + 2):
        raise ValueError("insufficient complete observations")
    y = vector[finite]
    if matrix.shape[1] == 0:
        return y - y.mean(), finite
    design = np.column_stack([np.ones(len(y)), matrix[finite]])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficients, finite


class LazyAdapter:
    """Base class whose probe is the only eager-import operation."""

    id = ""
    integration_id = ""
    module_name = ""
    package_name = ""
    capabilities: tuple[str, ...] = ()

    def _module(self) -> Any:
        return importlib.import_module(self.module_name)

    def probe(self) -> ProbeResult:
        module = self._module()
        version = getattr(module, "__version__", None)
        if not version and self.package_name:
            try:
                version = metadata.version(self.package_name)
            except Exception:
                version = None
        return ProbeResult(
            True,
            f"imported {self.module_name}",
            str(version) if version else None,
        )

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        raise NotImplementedError


__all__ = [
    "CAUSAL_CAVEAT",
    "PREDICTIVE_CAVEAT",
    "LazyAdapter",
    "as_2d_controls",
    "bounded_int",
    "residualize",
]
