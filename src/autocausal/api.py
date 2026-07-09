"""Public AutoCausal API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

from autocausal.discovery import discover_relationships
from autocausal.impute import ImputationReport, impute_dataframe
from autocausal.ingest import load_csv, load_parquet, load_sqlalchemy
from autocausal.results import DiscoveryResult
from autocausal.roles import ColumnRole, infer_column_roles


ImputeMethod = Literal["auto", "median_mode", "knn"]

__all__ = ["AutoCausal", "DiscoveryResult"]


class AutoCausal:
    """Load tabular data, impute missing fields, discover exploratory causal edges."""

    def __init__(self, df: pd.DataFrame, *, source: str = "memory") -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self._raw = df.copy()
        self._df = df.copy()
        self.source = source
        self.imputation: Optional[ImputationReport] = None
        self.result: Optional[DiscoveryResult] = None
        self.roles: dict[str, ColumnRole] = infer_column_roles(self._df)

    @classmethod
    def from_csv(cls, path: str | Path, **read_csv_kwargs: Any) -> "AutoCausal":
        df = load_csv(path, **read_csv_kwargs)
        return cls(df, source=f"csv:{path}")

    @classmethod
    def from_parquet(cls, path: str | Path, **kwargs: Any) -> "AutoCausal":
        df = load_parquet(path, **kwargs)
        return cls(df, source=f"parquet:{path}")

    @classmethod
    def from_sqlalchemy(
        cls,
        url: str,
        *,
        table: Optional[str] = None,
        query: Optional[str] = None,
        schema: Optional[str] = None,
        limit: Optional[int] = None,
        **engine_kwargs: Any,
    ) -> "AutoCausal":
        df = load_sqlalchemy(
            url,
            table=table,
            query=query,
            schema=schema,
            limit=limit,
            **engine_kwargs,
        )
        label = table or (query[:40] + "…" if query and len(query) > 40 else query) or "sql"
        return cls(df, source=f"sql:{label}")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, source: str = "dataframe") -> "AutoCausal":
        return cls(df, source=source)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def impute(self, method: ImputeMethod = "auto", *, knn_k: int = 5) -> "AutoCausal":
        self._df, self.imputation = impute_dataframe(self._df, method=method, knn_k=knn_k)
        self.roles = infer_column_roles(self._df)
        return self

    def discover(
        self,
        *,
        alpha: float = 0.05,
        max_cond_size: int = 2,
        min_abs_corr: float = 0.15,
        use_iv: bool = True,
    ) -> DiscoveryResult:
        if self.imputation is None and self._df.isna().any().any():
            self.impute(method="auto")
        self.roles = infer_column_roles(self._df)
        result = discover_relationships(
            self._df,
            roles=self.roles,
            alpha=alpha,
            max_cond_size=max_cond_size,
            min_abs_corr=min_abs_corr,
            use_iv=use_iv,
        )
        result.imputation = self.imputation
        self.result = result
        return result

    def report(self, *, as_markdown: bool = True) -> str:
        if self.result is None:
            self.discover()
        assert self.result is not None
        if as_markdown:
            return self.result.to_markdown()
        return self.result.to_json()

    def run(
        self,
        *,
        impute_method: ImputeMethod = "auto",
        **discover_kwargs: Any,
    ) -> DiscoveryResult:
        self.impute(method=impute_method)
        return self.discover(**discover_kwargs)
