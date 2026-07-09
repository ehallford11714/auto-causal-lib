"""Public AutoCausal API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Union

import pandas as pd

from autocausal.discovery import discover_relationships
from autocausal.impute import ImputationReport, impute_dataframe
from autocausal.ingest import load_csv, load_parquet, load_sqlalchemy
from autocausal.results import AutoResult, DiscoveryResult
from autocausal.roles import ColumnRole, infer_column_roles


ImputeMethod = Literal["auto", "median_mode", "knn"]

__all__ = ["AutoCausal", "DiscoveryResult", "AutoResult"]


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
        self.mining: Any = None
        self.guide_result: Any = None
        self.creation_result: Any = None
        self.inference_result: Any = None
        self.grounding: Any = None
        self.join_log: list[dict[str, Any]] = []
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

    @classmethod
    def connect(cls, url: Optional[str] = None, /, **kwargs: Any) -> Any:
        """Unified DB connect — see autocausal.db.connect."""
        from autocausal.db import connect

        return connect(url, **kwargs)

    @classmethod
    def ping(cls, url: str, *, timeout: float = 5.0) -> Any:
        from autocausal.db import ping

        return ping(url, timeout=timeout)

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def join_public(
        self,
        public_id: Union[str, list[str]],
        *,
        on: Optional[Union[str, list[str]]] = None,
        how: str = "left",
        allow_network: bool = False,
    ) -> "AutoCausal":
        """Join one or more public-suite tables into the current frame."""
        from autocausal.public_suite import join_public_frames

        joined, log = join_public_frames(
            self._df,
            public_id,
            on=on,
            how=how,
            allow_network=allow_network,
        )
        self._df = joined
        self.join_log.extend(log)
        self.roles = infer_column_roles(self._df)
        self.source = f"{self.source}+join:{public_id}"
        return self

    def mine(self, *, min_score: float = 0.15) -> "AutoCausal":
        from autocausal.mining import mine

        self.mining = mine(self._df, min_score=min_score)
        return self

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
        focus_columns: Optional[list[str]] = None,
    ) -> DiscoveryResult:
        if self.imputation is None and self._df.isna().any().any():
            self.impute(method="auto")
        work = self._df
        if focus_columns:
            keep = [c for c in focus_columns if c in work.columns]
            # keep enough columns; fall back if too few
            if len(keep) >= 2:
                work = work[keep]
        self.roles = infer_column_roles(work if focus_columns else self._df)
        result = discover_relationships(
            work if focus_columns else self._df,
            roles=self.roles,
            alpha=alpha,
            max_cond_size=max_cond_size,
            min_abs_corr=min_abs_corr,
            use_iv=use_iv,
        )
        result.imputation = self.imputation
        if self.mining is not None:
            result.mining = self.mining.to_dict() if hasattr(self.mining, "to_dict") else self.mining
        self.result = result
        return result

    def guide(
        self,
        *,
        text: Optional[str] = None,
        use_slm: bool = False,
        model_name: Optional[str] = None,
    ) -> Any:
        from autocausal.slm import guide_pipeline

        context: dict[str, Any] = {
            "text": text or "",
            "columns": (
                self.mining.columns
                if self.mining is not None
                else [{"name": c} for c in self._df.columns]
            ),
            "associations": self.mining.associations if self.mining is not None else [],
            "edges": self.result.edges if self.result is not None else [],
            "candidates": self.result.candidates if self.result is not None else {},
        }
        self.guide_result = guide_pipeline(context, use_slm=use_slm, model_name=model_name)
        if self.result is not None:
            self.result.guide = self.guide_result.to_dict()
        return self.guide_result

    def create(
        self,
        *,
        text: Optional[str] = None,
        use_slm: bool = False,
        model_name: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> Any:
        """SLM/rule *creation*: propose questions, instruments, morphemes."""
        from autocausal.slm import create_from_context

        context: dict[str, Any] = {
            "text": text or "",
            "columns": (
                self.mining.columns
                if self.mining is not None
                else [{"name": c} for c in self._df.columns]
            ),
            "candidates": self.result.candidates if self.result is not None else {},
            "edges": self.result.edges if self.result is not None else [],
        }
        if extra_context:
            context.update(extra_context)
        self.creation_result = create_from_context(
            context, use_slm=use_slm, model_name=model_name
        )
        return self.creation_result

    def interpret(
        self,
        *,
        text: Optional[str] = None,
        use_slm: bool = False,
        model_name: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> Any:
        """SLM/rule *inference* narrative + caveats over discovery/IV results."""
        from autocausal.slm import infer_from_results

        iv = None
        if self.result is not None:
            for e in self.result.edges or []:
                if e.get("type") == "iv_2sls" or e.get("first_stage_f") is not None:
                    iv = {
                        "coef": e.get("score"),
                        "first_stage_f": e.get("first_stage_f"),
                        "pvalue": e.get("pvalue"),
                        "instrument": e.get("instrument"),
                    }
                    break
        context: dict[str, Any] = {
            "text": text or "",
            "edges": self.result.edges if self.result is not None else [],
            "iv": iv,
            "candidates": self.result.candidates if self.result is not None else {},
        }
        if extra_context:
            context.update(extra_context)
        self.inference_result = infer_from_results(
            context, use_slm=use_slm, model_name=model_name
        )
        return self.inference_result

    def validate_tools(
        self,
        *,
        y: Optional[str] = None,
        d: Optional[str] = None,
        z: Optional[str] = None,
        claims_text: str = "",
    ) -> Any:
        """Run suite_tools.validate_pipeline on current frame/result."""
        from autocausal.suite_tools import validate_pipeline

        report: dict[str, Any] = {}
        if self.mining is not None and hasattr(self.mining, "to_dict"):
            report.update(self.mining.to_dict())
        if self.result is not None:
            report["edges"] = self.result.edges
            report["candidates"] = self.result.candidates
        return validate_pipeline(
            report,
            df=self._df,
            claims_text=claims_text,
            y=y,
            d=d,
            z=z,
        )

    def ground(self, *, use_web: bool = False, timeout: float = 3.0) -> Any:
        from autocausal.grounding import ground_edges

        edges = self.result.edges if self.result is not None else []
        self.grounding = ground_edges(edges, use_web=use_web, timeout=timeout)
        if self.result is not None:
            self.result.grounding = self.grounding.to_dict()
        return self.grounding

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

    @classmethod
    def auto(
        cls,
        path_or_url: str,
        *,
        table: Optional[str] = None,
        query: Optional[str] = None,
        text: Optional[str] = None,
        use_slm: bool = False,
        join: Optional[Union[str, list[str]]] = None,
        join_on: Optional[Union[str, list[str]]] = None,
        use_web_ground: bool = False,
        impute_method: ImputeMethod = "auto",
        second_pass: bool = True,
        **discover_kwargs: Any,
    ) -> AutoResult:
        """Orchestrated flow: load → ping? → join? → mine → impute → discover → guide → ground."""
        from autocausal.db import ping
        from autocausal.ingest import dialect_from_url

        notes: list[str] = []
        ping_info = None
        path = str(path_or_url)
        lower = path.lower()

        if lower.endswith(".csv"):
            ac = cls.from_csv(path)
        elif lower.endswith(".parquet"):
            ac = cls.from_parquet(path)
        elif "://" in path or path.startswith("sqlite:"):
            ping_info = ping(path, timeout=5.0)
            notes.append(f"ping ok={ping_info.ok} latency_ms={ping_info.latency_ms}")
            if not table and not query:
                # try bundled demo table name if sqlite demo
                raise ValueError("Database URL requires table= or query=")
            ac = cls.from_sqlalchemy(path, table=table, query=query)
            notes.append(f"dialect={dialect_from_url(path)}")
        else:
            # treat as CSV path fallback
            ac = cls.from_csv(path)

        if join:
            ac.join_public(join, on=join_on, allow_network=False)
            notes.append(f"joined public: {join}")

        ac.mine()
        mining = ac.mining
        ac.impute(method=impute_method)
        result = ac.discover(**discover_kwargs)
        guide = ac.guide(text=text, use_slm=use_slm)

        if second_pass and guide.focus_columns:
            focus = [c for c in guide.focus_columns if c in ac.df.columns]
            if len(focus) >= 2:
                notes.append(f"second-pass focus: {focus[:12]}")
                result = ac.discover(focus_columns=focus, **discover_kwargs)
                guide = ac.guide(text=text, use_slm=use_slm)

        grounding = ac.ground(use_web=use_web_ground)
        auto_result = AutoResult(
            discovery=result,
            mining=mining.to_dict() if mining is not None and hasattr(mining, "to_dict") else mining,
            guide=guide.to_dict() if guide is not None else None,
            grounding=grounding.to_dict() if grounding is not None else None,
            join_log=list(ac.join_log),
            ping=ping_info.to_dict() if ping_info is not None else None,
            source=ac.source,
            notes=notes,
        )
        return auto_result
