"""Column role inference (numeric / categorical / datetime / id-like)."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class ColumnRole(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    ID = "id"
    TEXT = "text"
    UNKNOWN = "unknown"


_ID_NAME_HINTS = ("id", "uuid", "guid", "pk", "key", "index")


def infer_column_roles(df: pd.DataFrame, *, max_cat_cardinality: int = 32) -> dict[str, ColumnRole]:
    roles: dict[str, ColumnRole] = {}
    n = len(df)
    for col in df.columns:
        s = df[col]
        name = str(col).lower()
        if any(h == name or name.endswith(f"_{h}") or name.startswith(f"{h}_") for h in _ID_NAME_HINTS):
            if s.nunique(dropna=True) >= max(0.9 * max(n, 1), 1):
                roles[col] = ColumnRole.ID
                continue

        if pd.api.types.is_bool_dtype(s):
            roles[col] = ColumnRole.BOOLEAN
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            roles[col] = ColumnRole.DATETIME
            continue

        if pd.api.types.is_numeric_dtype(s):
            nunq = int(s.nunique(dropna=True))
            if nunq <= 2:
                roles[col] = ColumnRole.BOOLEAN
            elif nunq <= max_cat_cardinality and nunq < max(n * 0.05, 3):
                roles[col] = ColumnRole.CATEGORICAL
            else:
                roles[col] = ColumnRole.NUMERIC
            continue

        # object / string
        parsed = pd.to_datetime(s, errors="coerce", utc=False)
        if parsed.notna().mean() > 0.8:
            roles[col] = ColumnRole.DATETIME
            continue

        nunq = int(s.nunique(dropna=True))
        avg_len = s.dropna().astype(str).str.len().mean() if nunq else 0
        if nunq <= max_cat_cardinality:
            roles[col] = ColumnRole.CATEGORICAL
        elif avg_len and avg_len > 40:
            roles[col] = ColumnRole.TEXT
        else:
            roles[col] = ColumnRole.CATEGORICAL if nunq < max(n * 0.5, 10) else ColumnRole.TEXT

    return roles


def numeric_matrix(df: pd.DataFrame, roles: dict[str, ColumnRole]) -> tuple[pd.DataFrame, list[str]]:
    """Return a float matrix for discovery (one-hot light: categoricals as codes)."""
    cols: list[str] = []
    pieces: list[pd.Series] = []
    for col, role in roles.items():
        if role in (ColumnRole.ID, ColumnRole.TEXT, ColumnRole.DATETIME):
            continue
        s = df[col]
        if role == ColumnRole.NUMERIC:
            pieces.append(pd.to_numeric(s, errors="coerce").astype(float))
            cols.append(col)
        elif role in (ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN):
            codes = pd.Categorical(s).codes.astype(float)
            codes[codes < 0] = np.nan
            pieces.append(pd.Series(codes, index=df.index, name=col))
            cols.append(col)
    if not pieces:
        return pd.DataFrame(index=df.index), []
    mat = pd.concat(pieces, axis=1)
    mat.columns = cols
    return mat, cols
