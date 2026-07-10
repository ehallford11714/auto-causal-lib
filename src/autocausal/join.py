"""Generic multi-frame alignment / join helpers (beyond public-suite joins)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence, Union

import pandas as pd

__all__ = ["AlignReport", "align", "suggest_keys"]


@dataclass
class AlignReport:
    """Audit trail for ``align``."""

    how: str
    keys: list[str]
    n_frames: int
    shape_before: list[tuple[int, int]] = field(default_factory=list)
    shape_after: tuple[int, int] = (0, 0)
    notes: list[str] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def suggest_keys(
    frames: Sequence[pd.DataFrame],
    *,
    prefer: Optional[Sequence[str]] = None,
) -> list[str]:
    """Suggest join keys as intersection of column names (prefer named keys)."""
    if not frames:
        return []
    common = set(str(c) for c in frames[0].columns)
    for f in frames[1:]:
        common &= set(str(c) for c in f.columns)
    preferred = [k for k in (prefer or []) if k in common]
    if preferred:
        return preferred
    # prefer id-like names
    ranked = sorted(
        common,
        key=lambda c: (
            0 if c.lower().endswith("_id") or c.lower() in ("id", "unit", "entity", "key") else 1,
            c,
        ),
    )
    return ranked[:3]


def align(
    frames: Sequence[pd.DataFrame],
    keys: Optional[Union[str, Sequence[str]]] = None,
    *,
    how: str = "outer",
    suffixes: Optional[Sequence[tuple[str, str]]] = None,
    validate: Optional[str] = None,
) -> tuple[pd.DataFrame, AlignReport]:
    """Align / successively join multiple frames on shared keys.

    Parameters
    ----------
    frames :
        Two or more DataFrames.
    keys :
        Join key(s). If None, uses :func:`suggest_keys`.
    how :
        pandas merge how: left | right | outer | inner.
    suffixes :
        Optional per-step suffix pairs; defaults to ``_0/_1``, ``_1/_2``, …
    validate :
        Optional pandas merge validate (one_to_one, …).
    """
    frames = list(frames)
    if len(frames) < 1:
        raise ValueError("align() requires at least one frame")
    if len(frames) == 1:
        report = AlignReport(
            how=how,
            keys=[],
            n_frames=1,
            shape_before=[(len(frames[0]), len(frames[0].columns))],
            shape_after=(len(frames[0]), len(frames[0].columns)),
            notes=["Single frame — no join performed."],
        )
        return frames[0].copy(), report

    if keys is None:
        key_list = suggest_keys(frames)
        if not key_list:
            raise ValueError("No common columns to use as join keys; pass keys= explicitly.")
    elif isinstance(keys, str):
        key_list = [keys]
    else:
        key_list = list(keys)

    shape_before = [(len(f), len(f.columns)) for f in frames]
    notes = [
        "Generic multi-frame align — verify key uniqueness and leakage before discovery.",
    ]
    log: list[dict[str, Any]] = []
    out = frames[0].copy()
    for i, right in enumerate(frames[1:], start=1):
        missing = [k for k in key_list if k not in out.columns or k not in right.columns]
        if missing:
            raise ValueError(f"Join keys missing at step {i}: {missing}")
        if suffixes and i - 1 < len(suffixes):
            suf = suffixes[i - 1]
        else:
            suf = (f"_{i - 1}", f"_{i}")
        before = out.shape
        merge_kwargs: dict[str, Any] = {"on": key_list, "how": how, "suffixes": suf}
        if validate:
            merge_kwargs["validate"] = validate
        out = out.merge(right, **merge_kwargs)
        entry = {
            "step": i,
            "how": how,
            "keys": list(key_list),
            "left_shape": before,
            "right_shape": (len(right), len(right.columns)),
            "out_shape": out.shape,
            "suffixes": list(suf),
        }
        log.append(entry)
        notes.append(f"step {i}: {before} ⋈ {right.shape} → {out.shape}")

    report = AlignReport(
        how=how,
        keys=list(key_list),
        n_frames=len(frames),
        shape_before=shape_before,
        shape_after=(len(out), len(out.columns)),
        notes=notes,
        log=log,
    )
    return out, report
