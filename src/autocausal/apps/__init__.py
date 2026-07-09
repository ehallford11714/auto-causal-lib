"""Optional UI apps (Streamlit). Soft-optional — not imported by core autocausal."""

from __future__ import annotations

__all__ = ["physics_demo_path", "synthetic_oscillator", "synthetic_kpi_panel"]


def physics_demo_path() -> str:
    """Absolute path to the physics Streamlit demo script."""
    from pathlib import Path

    return str(Path(__file__).resolve().parent / "physics_streamlit.py")


def __getattr__(name: str):
    if name in ("synthetic_oscillator", "synthetic_kpi_panel", "load_demo_frame"):
        from autocausal.apps import samples as _samples

        return getattr(_samples, name)
    raise AttributeError(f"module 'autocausal.apps' has no attribute {name!r}")
