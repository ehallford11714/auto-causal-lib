"""Data quality, dataframe conversion, and opt-in instrumentation adapters."""

from __future__ import annotations

import re
from typing import Any, Mapping
from urllib.parse import urlparse

from autocausal.integrations.adapters.base import LazyAdapter


class PanderaAdapter(LazyAdapter):
    id = "pandera.validation"
    integration_id = "pandera"
    module_name = "pandera"
    package_name = "pandera"
    capabilities = ("data.validate",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "data.validate":
            raise KeyError(capability)
        return self.validate(**kwargs)

    @staticmethod
    def validate(
        *,
        frame: Any,
        schema: Any,
        lazy: bool = True,
        return_frame: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        if schema is None or not hasattr(schema, "validate"):
            raise TypeError("schema must be a caller-supplied Pandera schema")
        try:
            validated = schema.validate(frame, lazy=bool(lazy))
        except Exception as exc:
            failure_count = None
            failure_cases = getattr(exc, "failure_cases", None)
            if failure_cases is not None:
                try:
                    failure_count = int(len(failure_cases))
                except Exception:
                    failure_count = None
            return {
                "valid": False,
                "error_type": type(exc).__name__,
                "failure_count": failure_count,
                "message": str(exc)[:2_000],
                "raw_failure_values_included": False,
            }
        output: dict[str, Any] = {
            "valid": True,
            "n_rows": int(len(validated)),
            "n_columns": int(len(validated.columns)),
            "columns": [str(item) for item in validated.columns],
        }
        if return_frame:
            output["frame"] = validated
        return output


class PolarsAdapter(LazyAdapter):
    id = "polars.convert"
    integration_id = "polars"
    module_name = "polars"
    package_name = "polars"
    capabilities = ("data.convert",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "data.convert":
            raise KeyError(capability)
        return self.convert(**kwargs)

    def convert(self, *, data: Any, target: str = "polars", **_: Any) -> Any:
        polars = self._module()
        selected = str(target).lower()
        if selected == "polars":
            if isinstance(data, polars.DataFrame):
                return data
            if type(data).__module__.startswith("pyarrow"):
                return polars.from_arrow(data)
            if type(data).__module__.startswith("pandas"):
                return polars.from_pandas(data)
            return polars.DataFrame(data)
        if selected == "pandas":
            if not isinstance(data, polars.DataFrame):
                data = polars.DataFrame(data)
            return data.to_pandas()
        if selected == "arrow":
            if not isinstance(data, polars.DataFrame):
                data = polars.DataFrame(data)
            return data.to_arrow()
        raise ValueError("target must be polars, pandas, or arrow")


class PyArrowAdapter(LazyAdapter):
    id = "pyarrow.convert"
    integration_id = "pyarrow"
    module_name = "pyarrow"
    package_name = "pyarrow"
    capabilities = ("data.convert",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "data.convert":
            raise KeyError(capability)
        return self.convert(**kwargs)

    def convert(self, *, data: Any, target: str = "arrow", **_: Any) -> Any:
        pyarrow = self._module()
        selected = str(target).lower()
        if selected == "arrow":
            if isinstance(data, pyarrow.Table):
                return data
            if hasattr(data, "to_arrow"):
                return data.to_arrow()
            if hasattr(data, "columns"):
                return pyarrow.Table.from_pandas(data, preserve_index=False)
            return pyarrow.Table.from_pylist(list(data))
        if selected == "pandas":
            table = data if isinstance(data, pyarrow.Table) else self.convert(
                data=data,
                target="arrow",
            )
            return table.to_pandas()
        raise ValueError("target must be arrow or pandas")


class MLflowAdapter(LazyAdapter):
    id = "mlflow.redacted-manifest"
    integration_id = "mlflow"
    module_name = "mlflow"
    package_name = "mlflow"
    capabilities = ("ops.mlflow.log_manifest",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "ops.mlflow.log_manifest":
            raise KeyError(capability)
        return self.log_manifest(**kwargs)

    def log_manifest(
        self,
        *,
        metrics: Mapping[str, Any],
        params: Mapping[str, Any],
        tracking_uri: str,
        enabled: bool = False,
        allow_network: bool = False,
        run_name: str | None = None,
        tags: Mapping[str, str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not enabled:
            raise PermissionError(
                "MLflow instrumentation is disabled by default; pass enabled=True"
            )
        if not tracking_uri:
            raise ValueError("an explicit tracking_uri is required")
        parsed = urlparse(str(tracking_uri))
        windows_path = bool(re.match(r"^[A-Za-z]:[\\/]", str(tracking_uri)))
        is_remote = not windows_path and (
            parsed.scheme not in ("", "file", "sqlite") or bool(parsed.hostname)
        )
        if is_remote and not allow_network:
            raise PermissionError(
                "remote MLflow tracking requires allow_network=True; no data was sent"
            )
        if len(metrics) > 100 or len(params) > 100:
            raise ValueError("metrics and params are each capped at 100 entries")
        safe_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"metric {key!r} must be numeric")
            safe_metrics[str(key)[:250]] = float(value)
        safe_params: dict[str, Any] = {}
        for key, value in params.items():
            if not isinstance(value, (str, int, float, bool)) and value is not None:
                raise TypeError(
                    f"param {key!r} must be a scalar; raw frames are forbidden"
                )
            safe_params[str(key)[:250]] = str(value)[:500]
        mlflow = self._module()
        mlflow.set_tracking_uri(str(tracking_uri))
        with mlflow.start_run(run_name=run_name, tags=dict(tags or {})) as run:
            if safe_params:
                mlflow.log_params(safe_params)
            if safe_metrics:
                mlflow.log_metrics(safe_metrics)
            run_id = str(run.info.run_id)
        return {
            "run_id": run_id,
            "tracking_uri": str(tracking_uri),
            "remote": is_remote,
            "logged_metric_count": len(safe_metrics),
            "logged_param_count": len(safe_params),
            "raw_data_logged": False,
            "telemetry_default": False,
        }


def data_adapters() -> tuple[LazyAdapter, ...]:
    return (
        PanderaAdapter(),
        PolarsAdapter(),
        PyArrowAdapter(),
        MLflowAdapter(),
    )


__all__ = [
    "MLflowAdapter",
    "PanderaAdapter",
    "PolarsAdapter",
    "PyArrowAdapter",
    "data_adapters",
]
