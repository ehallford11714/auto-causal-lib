"""Soft bridge: IntentIsolates layer motifs → AutoCausal / IV.

Primary implementation lives in ``intentisolates.causal``. This module
re-exports when ``intentisolates`` is installed.
"""

from __future__ import annotations

from typing import Any


def _load():
    try:
        from intentisolates.causal import (  # type: ignore
            LayerCausalResult,
            LayerCausalSuite,
            build_feature_frame,
            estimate_indication,
            estimate_layer_iv,
        )
    except ImportError as e:
        raise ImportError(
            "isolates-causal requires the intentisolates package. "
            "Install with: pip install intentisolates"
        ) from e
    return {
        "LayerCausalSuite": LayerCausalSuite,
        "LayerCausalResult": LayerCausalResult,
        "build_feature_frame": build_feature_frame,
        "estimate_indication": estimate_indication,
        "estimate_layer_iv": estimate_layer_iv,
    }


def __getattr__(name: str) -> Any:
    if name in (
        "LayerCausalSuite",
        "LayerCausalResult",
        "build_feature_frame",
        "estimate_indication",
        "estimate_layer_iv",
    ):
        return _load()[name]
    raise AttributeError(f"module 'autocausal.isolates_bridge' has no attribute {name!r}")


def run_isolates_causal(
    text: str,
    *,
    outcome_hint: str | None = None,
    mock_iv: bool = False,
    n_bootstrap: int = 48,
    seed: int = 17,
    backend: str = "rule",
) -> Any:
    """Convenience: text → LayerCausalResult (requires intentisolates)."""
    Suite = _load()["LayerCausalSuite"]
    suite = Suite.from_text(text, backend=backend)
    return suite.run(
        outcome_hint=outcome_hint,
        mock_iv=mock_iv,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


__all__ = [
    "LayerCausalSuite",
    "LayerCausalResult",
    "build_feature_frame",
    "estimate_indication",
    "estimate_layer_iv",
    "run_isolates_causal",
]
