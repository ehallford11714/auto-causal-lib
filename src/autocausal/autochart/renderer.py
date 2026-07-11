"""Headless, dependency-soft chart rendering with production-safe aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Optional

import numpy as np
import pandas as pd

from autocausal.autochart.report import AutoChartReport, RenderedChart
from autocausal.autochart.specs import ChartFilter, ChartSpec


COLORBLIND_SAFE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
)


def _plotly_available() -> bool:
    return find_spec("plotly") is not None


def _matplotlib_available() -> bool:
    return find_spec("matplotlib") is not None


def available_backends() -> dict[str, bool]:
    return {
        "plotly": _plotly_available(),
        "matplotlib": _matplotlib_available(),
        "data": True,
    }


@dataclass
class _PreparedChart:
    frame: pd.DataFrame
    kind: str
    x: Optional[str]
    y: Optional[str]
    color: Optional[str]
    payload: dict[str, Any]
    warnings: list[str]
    aggregated: bool
    contains_raw_values: bool
    redact_annotations: bool


def _apply_filter(frame: pd.DataFrame, item: ChartFilter) -> pd.DataFrame:
    series = frame[item.column]
    op = item.operator
    if op == "eq":
        mask = series == item.value
    elif op == "ne":
        mask = series != item.value
    elif op == "lt":
        mask = series < item.value
    elif op == "le":
        mask = series <= item.value
    elif op == "gt":
        mask = series > item.value
    elif op == "ge":
        mask = series >= item.value
    elif op == "in":
        values = (
            item.value
            if isinstance(item.value, Sequence)
            and not isinstance(item.value, (str, bytes))
            else [item.value]
        )
        mask = series.isin(values)
    elif op == "not_in":
        values = (
            item.value
            if isinstance(item.value, Sequence)
            and not isinstance(item.value, (str, bytes))
            else [item.value]
        )
        mask = ~series.isin(values)
    elif op == "isna":
        mask = series.isna()
    elif op == "notna":
        mask = series.notna()
    else:  # guarded by ChartFilter
        raise ValueError(f"unsupported filter operator {op!r}")
    return frame.loc[mask.fillna(False)]


def _records(frame: pd.DataFrame, *, limit: int = 5_000) -> list[dict[str, Any]]:
    safe = frame.iloc[:limit].copy()
    safe = safe.where(pd.notna(safe), None)
    return safe.to_dict(orient="records")


def _histogram_summary(
    series: pd.Series,
    *,
    bins: int = 20,
    category_redaction: bool = False,
    minimum_count: int = 1,
) -> pd.DataFrame:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(3, int(series.notna().sum() * 0.8)):
        values = numeric.dropna().to_numpy(dtype=float)
        if len(values) < minimum_count:
            return pd.DataFrame({"bin": [], "count": []})
        counts, boundaries = np.histogram(values, bins=min(bins, max(2, len(values))))
        labels = [
            f"[{boundaries[index]:.6g}, {boundaries[index + 1]:.6g})"
            for index in range(len(counts))
        ]
        output = pd.DataFrame({"bin": labels, "count": counts.astype(int)})
        return output.loc[output["count"] >= minimum_count].reset_index(drop=True)
    counts = series.fillna("[MISSING]").astype(str).value_counts(dropna=False)
    counts = counts.loc[counts >= minimum_count]
    labels = [
        f"category_{index + 1}" if category_redaction else str(value)
        for index, value in enumerate(counts.index)
    ]
    return pd.DataFrame({"bin": labels, "count": counts.to_numpy(dtype=int)})


def _quantile_relationship(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    bins: int = 12,
    minimum_group_size: int = 3,
) -> pd.DataFrame:
    data = frame[[x, y]].copy()
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna()
    if len(data) < minimum_group_size:
        return pd.DataFrame({"x_bin": [], "y_mean": [], "n": []})
    unique = int(data[x].nunique())
    if unique <= bins:
        grouped = data.groupby(x, observed=True)[y].agg(["mean", "size"]).reset_index()
        grouped.columns = ["x_bin", "y_mean", "n"]
        if int(grouped["size" if "size" in grouped else "n"].min()) >= minimum_group_size:
            return grouped
    quantile_count = min(bins, max(2, len(data) // minimum_group_size))
    try:
        data["_bin"] = pd.qcut(
            data[x], q=min(quantile_count, unique), duplicates="drop"
        )
    except ValueError:
        data["_bin"] = pd.cut(
            data[x], bins=min(quantile_count, unique), duplicates="drop"
        )
    grouped = (
        data.groupby("_bin", observed=True)
        .agg(x_bin=(x, "mean"), y_mean=(y, "mean"), n=(y, "size"))
        .reset_index(drop=True)
    )
    return grouped.loc[grouped["n"] >= minimum_group_size].reset_index(drop=True)


def _correlation_long(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().sum() >= 2]
    if numeric.shape[1] < 2:
        return pd.DataFrame({"column_x": [], "column_y": [], "value": []})
    matrix = numeric.corr()
    long = (
        matrix.rename_axis("column_x")
        .reset_index()
        .melt(id_vars="column_x", var_name="column_y", value_name="value")
    )
    return long


def _balance_frame(
    frame: pd.DataFrame,
    treatment: str,
    covariates: Sequence[str],
) -> pd.DataFrame:
    groups = list(frame[treatment].dropna().unique())
    if len(groups) != 2:
        return pd.DataFrame({"covariate": [], "standardized_difference": []})
    output: list[dict[str, Any]] = []
    for column in covariates:
        values = pd.to_numeric(frame[column], errors="coerce")
        left = values[frame[treatment] == groups[0]].dropna()
        right = values[frame[treatment] == groups[1]].dropna()
        if len(left) < 2 or len(right) < 2:
            continue
        pooled = np.sqrt((float(left.var()) + float(right.var())) / 2.0)
        difference = 0.0 if pooled == 0 else (float(right.mean()) - float(left.mean())) / pooled
        output.append(
            {
                "covariate": str(column),
                "standardized_difference": float(difference),
                "absolute_standardized_difference": abs(float(difference)),
            }
        )
    return pd.DataFrame(output)


def _edge_frame(context: Mapping[str, Any], spec: ChartSpec) -> pd.DataFrame:
    edges = context.get("edges") or spec.metadata.get("edges") or []
    rows: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        source, target = edge.get("source"), edge.get("target")
        if source is None or target is None:
            continue
        rows.append(
            {
                "source": str(source),
                "target": str(target),
                "stability": edge.get("stability"),
                "score": edge.get("score"),
                "evidence_grade": edge.get("evidence_grade", "unverified"),
            }
        )
    return pd.DataFrame(rows)


def _gate_frame(context: Mapping[str, Any], spec: ChartSpec) -> pd.DataFrame:
    gates = context.get("gates") or spec.metadata.get("gates") or []
    if hasattr(gates, "results"):
        gates = getattr(gates, "results")
    rows: list[dict[str, Any]] = []
    for gate in gates or []:
        if hasattr(gate, "to_dict"):
            gate = gate.to_dict()
        if not isinstance(gate, Mapping):
            continue
        status = str(
            gate.get("status")
            or ("pass" if gate.get("ok") is True else "fail")
        ).lower()
        rows.append(
            {
                "gate": str(gate.get("id") or gate.get("name") or "gate"),
                "stage": str(gate.get("stage") or "unspecified"),
                "status": status,
                "status_code": {
                    "pass": 1.0,
                    "warn": 0.5,
                    "skip": 0.25,
                    "fail": -1.0,
                    "escalate": -1.0,
                }.get(status, 0.0),
            }
        )
    return pd.DataFrame(rows)


def _curve_frame(chart_type: str, context: Mapping[str, Any]) -> pd.DataFrame:
    truth = np.asarray(context.get("y_true") if context.get("y_true") is not None else [])
    score = np.asarray(
        context.get("y_score") if context.get("y_score") is not None else []
    )
    if len(truth) == 0 or len(truth) != len(score):
        return pd.DataFrame({"x": [], "y": []})
    try:
        if chart_type == "calibration":
            from sklearn.calibration import calibration_curve

            observed, predicted = calibration_curve(
                truth, score, n_bins=10, strategy="quantile"
            )
            return pd.DataFrame(
                {"predicted_probability": predicted, "observed_fraction": observed}
            )
        if chart_type == "roc":
            from sklearn.metrics import roc_curve

            false_positive, true_positive, _ = roc_curve(truth, score)
            return pd.DataFrame(
                {"false_positive_rate": false_positive, "true_positive_rate": true_positive}
            )
        from sklearn.metrics import precision_recall_curve

        precision, recall, _ = precision_recall_curve(truth, score)
        return pd.DataFrame({"recall": recall, "precision": precision})
    except Exception:
        return pd.DataFrame({"x": [], "y": []})


def _prepare(
    frame: pd.DataFrame,
    spec: ChartSpec,
    *,
    production: bool,
    allow_raw_values: bool,
    context: Optional[Mapping[str, Any]],
) -> _PreparedChart:
    spec.validate(frame, production=production)
    context = dict(context or {})
    filtered = frame
    for item in spec.filters:
        filtered = _apply_filter(filtered, item)
    n_filtered = len(filtered)
    warnings: list[str] = []
    safe_production = production and not allow_raw_values
    aggregated = False
    contains_raw = False
    chart_frame = pd.DataFrame()
    kind = spec.type
    x, y, color = (
        spec.x,
        (spec.y_columns[0] if spec.y_columns else None),
        spec.color or spec.facet,
    )

    if kind == "missingness":
        columns = spec.metadata.get("required_columns") or list(filtered.columns)
        columns = [column for column in columns if column in filtered.columns]
        chart_frame = pd.DataFrame(
            {
                "column": [str(column) for column in columns],
                "missing_fraction": [
                    float(filtered[column].isna().mean()) for column in columns
                ],
            }
        )
        kind, x, y, color, aggregated = (
            "bar",
            "column",
            "missing_fraction",
            None,
            True,
        )
    elif kind in ("correlation", "association"):
        columns = spec.metadata.get("required_columns") or spec.referenced_columns()
        if not columns:
            columns = [
                str(column)
                for column in filtered.select_dtypes(include=np.number).columns
            ]
        chart_frame = _correlation_long(filtered, columns)
        kind, x, y, color, aggregated = (
            "heatmap",
            "column_x",
            "column_y",
            "value",
            True,
        )
    elif kind == "covariate_balance":
        treatment = spec.x
        required = spec.metadata.get("required_columns") or []
        covariates = [
            column for column in required if column != treatment and column in filtered
        ]
        if treatment is None:
            chart_frame = pd.DataFrame()
        else:
            chart_frame = _balance_frame(filtered, treatment, covariates)
        kind, x, y, color, aggregated = (
            "bar",
            "covariate",
            "standardized_difference",
            None,
            True,
        )
    elif kind in ("dag", "network", "edge_stability"):
        chart_frame = _edge_frame(context, spec)
        if kind == "edge_stability":
            if not chart_frame.empty:
                chart_frame["edge"] = (
                    chart_frame["source"] + " → " + chart_frame["target"]
                )
                chart_frame["stability"] = pd.to_numeric(
                    chart_frame["stability"], errors="coerce"
                )
                chart_frame = chart_frame.dropna(subset=["stability"]).sort_values(
                    "stability", ascending=False
                )
            kind, x, y, color = "bar", "edge", "stability", None
        else:
            kind, x, y, color = "network", None, None, None
        aggregated = True
    elif kind in ("gate_dashboard", "evidence_matrix"):
        chart_frame = _gate_frame(context, spec)
        kind, x, y, color, aggregated = (
            "gate_matrix",
            "gate",
            "status_code",
            "status",
            True,
        )
    elif kind in ("calibration", "roc", "pr"):
        chart_frame = _curve_frame(kind, context)
        mappings = {
            "calibration": ("predicted_probability", "observed_fraction"),
            "roc": ("false_positive_rate", "true_positive_rate"),
            "pr": ("recall", "precision"),
        }
        x, y = mappings[kind]
        kind, color, aggregated = "line", None, True
    elif kind == "feature_importance":
        importance = context.get("feature_importance") or spec.metadata.get(
            "feature_importance"
        ) or []
        chart_frame = pd.DataFrame(importance)
        if not chart_frame.empty:
            feature_column = (
                "feature" if "feature" in chart_frame else chart_frame.columns[0]
            )
            importance_column = (
                "importance_mean"
                if "importance_mean" in chart_frame
                else "importance"
                if "importance" in chart_frame
                else chart_frame.columns[-1]
            )
            chart_frame = chart_frame.sort_values(
                importance_column, ascending=False
            ).iloc[:30]
            x, y = str(feature_column), str(importance_column)
        else:
            x, y = "feature", "importance"
        kind, color, aggregated = "bar", None, True
    elif kind == "residual_diagnostics":
        residuals = context.get("residuals")
        if residuals is None and y and y in filtered:
            residuals = filtered[y]
        chart_frame = _histogram_summary(
            pd.Series([] if residuals is None else residuals),
            bins=20,
            minimum_count=3 if safe_production else 1,
        )
        kind, x, y, color, aggregated = "bar", "bin", "count", None, True
    elif kind == "distribution":
        if x is None:
            chart_frame = pd.DataFrame()
        elif safe_production:
            chart_frame = _histogram_summary(
                filtered[x], category_redaction=True, minimum_count=3
            )
            kind, x, y, color, aggregated = "bar", "bin", "count", None, True
        else:
            chart_frame = filtered[[x]].dropna()
            kind = "histogram"
            contains_raw = True
    elif kind in ("iv_first_stage", "scatter", "treatment_outcome"):
        if x is None or y is None:
            chart_frame = pd.DataFrame()
        elif safe_production:
            chart_frame = _quantile_relationship(filtered, x, y)
            kind, x, y, color, aggregated = (
                "scatter",
                "x_bin",
                "y_mean",
                None,
                True,
            )
            warnings.append(
                "Production scatter was converted to binned means to avoid row-level values."
            )
        else:
            columns = [x, y] + ([color] if color else [])
            chart_frame = filtered[columns].dropna(subset=[x, y])
            kind = "scatter" if kind == "iv_first_stage" else kind
            contains_raw = True
    elif kind == "overlap":
        score_column = y
        if score_column is None:
            required = spec.metadata.get("required_columns") or []
            score_column = next(
                (
                    column
                    for column in required
                    if column != x
                    and column in filtered
                    and pd.api.types.is_numeric_dtype(filtered[column])
                ),
                None,
            )
        if x is None or score_column is None:
            chart_frame = pd.DataFrame()
        elif safe_production:
            pieces: list[pd.DataFrame] = []
            groups = list(filtered[x].dropna().unique())
            for index, group in enumerate(groups[:10]):
                summary = _histogram_summary(
                    filtered.loc[filtered[x] == group, score_column],
                    bins=12,
                    minimum_count=3,
                )
                summary["group"] = f"group_{index + 1}"
                pieces.append(summary)
            chart_frame = (
                pd.concat(pieces, ignore_index=True)
                if pieces
                else pd.DataFrame({"bin": [], "count": [], "group": []})
            )
            kind, x, y, color, aggregated = "bar", "bin", "count", "group", True
        else:
            chart_frame = filtered[[x, score_column]].dropna()
            kind, x, y, color = "histogram_grouped", score_column, None, x
            contains_raw = True
    elif kind == "panel_trend":
        if x is None or y is None:
            chart_frame = pd.DataFrame()
        else:
            columns = [x, y] + ([color] if color else [])
            subset = filtered[columns].dropna(subset=[x, y])
            if safe_production:
                subset = subset.sort_values(x, kind="stable").reset_index(drop=True)
                period_count = min(20, len(subset) // 3)
                if period_count >= 1:
                    subset["_period"] = pd.qcut(
                        np.arange(len(subset)),
                        q=period_count,
                        labels=False,
                        duplicates="drop",
                    )
                    groupers = ["_period"] + ([color] if color else [])
                    chart_frame = (
                        subset.groupby(groupers, observed=True)[y]
                        .agg(["mean", "size"])
                        .reset_index()
                        .rename(columns={"mean": y, "size": "n"})
                    )
                    chart_frame = chart_frame.loc[chart_frame["n"] >= 3]
                    chart_frame["period"] = chart_frame["_period"].map(
                        lambda value: f"period_{int(value) + 1}"
                    )
                    chart_frame = chart_frame.drop(columns=["_period"])
                else:
                    chart_frame = pd.DataFrame(
                        columns=["period", y, "n"] + ([color] if color else [])
                    )
                x = "period"
                if color and color in chart_frame:
                    aliases = {
                        value: f"group_{index + 1}"
                        for index, value in enumerate(
                            chart_frame[color].drop_duplicates().tolist()
                        )
                    }
                    chart_frame[color] = chart_frame[color].map(aliases)
                aggregated = True
            elif spec.aggregation:
                groupers = [x] + ([color] if color else [])
                chart_frame = (
                    subset.groupby(groupers, observed=True)[y]
                    .mean()
                    .reset_index()
                )
                aggregated = True
            else:
                chart_frame = subset
                contains_raw = True
            kind = "line"
    elif kind == "subgroup_effects":
        if x is None or y is None:
            chart_frame = pd.DataFrame()
        else:
            groupers = [x] + ([color] if color else [])
            chart_frame = (
                filtered.groupby(groupers, observed=True)[y]
                .agg(["mean", "size"])
                .reset_index()
                .rename(columns={"mean": "outcome_mean", "size": "n"})
            )
            if safe_production:
                chart_frame = chart_frame.loc[chart_frame["n"] >= 3].copy()
                aliases = {
                    value: f"subgroup_{index + 1}"
                    for index, value in enumerate(
                        chart_frame[x].drop_duplicates().tolist()
                    )
                }
                chart_frame[x] = chart_frame[x].map(aliases)
                if color:
                    color_aliases = {
                        value: f"treatment_{index + 1}"
                        for index, value in enumerate(
                            chart_frame[color].drop_duplicates().tolist()
                        )
                    }
                    chart_frame[color] = chart_frame[color].map(color_aliases)
            kind, y, aggregated = "bar", "outcome_mean", True
    else:
        columns = spec.referenced_columns()
        chart_frame = filtered[columns].copy() if columns else pd.DataFrame(index=filtered.index)
        if safe_production and x and y:
            chart_frame = _quantile_relationship(filtered, x, y)
            x, y, color, aggregated = "x_bin", "y_mean", None, True
        else:
            contains_raw = bool(columns)

    if len(chart_frame) > spec.max_rows:
        if spec.deterministic_sample:
            original_render_rows = len(chart_frame)
            chart_frame = chart_frame.sample(
                n=spec.max_rows, random_state=spec.random_state
            ).sort_index()
            warnings.append(
                f"Deterministically sampled {spec.max_rows} of "
                f"{original_render_rows} render rows."
            )
        else:
            # Validation normally prevents this path.
            chart_frame = chart_frame.iloc[: spec.max_rows]

    if safe_production:
        contains_raw = False
    payload = {
        "schema": "AutoCausalChartData.v1",
        "n_input_rows": int(len(frame)),
        "n_filtered_rows": int(n_filtered),
        "n_render_rows": int(len(chart_frame)),
        "columns": [str(column) for column in chart_frame.columns],
        "records": _records(chart_frame, limit=spec.max_rows),
        "aggregated": aggregated,
        "contains_raw_values": contains_raw,
    }
    if chart_frame.empty:
        warnings.append("No compatible rows or analysis metadata were available.")
    return _PreparedChart(
        frame=chart_frame,
        kind=kind,
        x=x,
        y=y,
        color=color,
        payload=payload,
        warnings=warnings,
        aggregated=aggregated,
        contains_raw_values=contains_raw,
        redact_annotations=safe_production,
    )


def _network_layout(edges: pd.DataFrame) -> dict[str, tuple[float, float]]:
    nodes = sorted(
        set(edges.get("source", pd.Series(dtype=str)).astype(str))
        | set(edges.get("target", pd.Series(dtype=str)).astype(str))
    )
    if not nodes:
        return {}
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    return {
        node: (float(np.cos(angle)), float(np.sin(angle)))
        for node, angle in zip(nodes, angles)
    }


def _render_plotly(prepared: _PreparedChart, spec: ChartSpec) -> Any:
    import plotly.graph_objects as go

    frame = prepared.frame
    figure = go.Figure()
    if prepared.kind == "network":
        positions = _network_layout(frame)
        for row in frame.to_dict(orient="records"):
            source, target = str(row["source"]), str(row["target"])
            if source not in positions or target not in positions:
                continue
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            figure.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line={"color": "#777777", "width": 1.5},
                    hoverinfo="text",
                    text=[
                        f"{source} → {target}; evidence={row.get('evidence_grade')}",
                        "",
                    ],
                    showlegend=False,
                )
            )
        if positions:
            figure.add_trace(
                go.Scatter(
                    x=[positions[node][0] for node in positions],
                    y=[positions[node][1] for node in positions],
                    text=list(positions),
                    mode="markers+text",
                    textposition="top center",
                    marker={"size": 14, "color": COLORBLIND_SAFE[0]},
                    showlegend=False,
                )
            )
    elif prepared.kind == "heatmap":
        if not frame.empty:
            matrix = frame.pivot(index=prepared.y, columns=prepared.x, values=prepared.color)
            figure.add_trace(
                go.Heatmap(
                    z=matrix.to_numpy(),
                    x=[str(value) for value in matrix.columns],
                    y=[str(value) for value in matrix.index],
                    colorscale="RdBu",
                    zmid=0,
                    colorbar={"title": "association"},
                )
            )
    elif prepared.kind == "gate_matrix":
        if not frame.empty:
            colors = [
                {
                    "pass": COLORBLIND_SAFE[2],
                    "warn": COLORBLIND_SAFE[1],
                    "skip": COLORBLIND_SAFE[4],
                    "fail": COLORBLIND_SAFE[5],
                    "escalate": COLORBLIND_SAFE[5],
                }.get(str(value), "#777777")
                for value in frame["status"]
            ]
            figure.add_trace(
                go.Bar(
                    x=frame["gate"],
                    y=frame["status_code"],
                    marker_color=colors,
                    text=frame["status"] if spec.accessibility.show_labels else None,
                )
            )
    elif prepared.kind in ("bar", "line", "scatter"):
        groups = (
            list(frame[prepared.color].drop_duplicates())
            if prepared.color and prepared.color in frame
            else [None]
        )
        trace_type = {"bar": go.Bar, "line": go.Scatter, "scatter": go.Scatter}[
            prepared.kind
        ]
        for index, group in enumerate(groups):
            subset = frame if group is None else frame.loc[frame[prepared.color] == group]
            options: dict[str, Any] = {
                "x": subset[prepared.x] if prepared.x in subset else [],
                "y": subset[prepared.y] if prepared.y in subset else [],
                "name": str(group) if group is not None else None,
                "marker": {"color": COLORBLIND_SAFE[index % len(COLORBLIND_SAFE)]},
            }
            if prepared.kind in ("line", "scatter"):
                options["mode"] = (
                    "lines+markers" if prepared.kind == "line" else "markers"
                )
            if spec.accessibility.show_labels and prepared.kind == "bar":
                options["text"] = options["y"]
            figure.add_trace(trace_type(**options))
    elif prepared.kind in ("histogram", "histogram_grouped"):
        groups = (
            list(frame[prepared.color].drop_duplicates())
            if prepared.color and prepared.color in frame
            else [None]
        )
        for index, group in enumerate(groups):
            subset = frame if group is None else frame.loc[frame[prepared.color] == group]
            figure.add_trace(
                go.Histogram(
                    x=subset[prepared.x] if prepared.x in subset else [],
                    name=str(group) if group is not None else None,
                    marker_color=COLORBLIND_SAFE[index % len(COLORBLIND_SAFE)],
                    opacity=0.65,
                )
            )
        figure.update_layout(barmode="overlay")
    elif prepared.kind in ("treatment_outcome", "box"):
        figure.add_trace(
            go.Box(
                x=frame[prepared.x] if prepared.x in frame else [],
                y=frame[prepared.y] if prepared.y in frame else [],
                marker_color=COLORBLIND_SAFE[0],
                boxmean=True,
            )
        )
    figure.update_layout(
        title=spec.title,
        xaxis_title=prepared.x,
        yaxis_title=prepared.y,
        template="plotly_white",
        colorway=list(COLORBLIND_SAFE),
        meta={
            "alt_text": spec.accessibility.alt_text,
            "causal_interpretation": "not_established",
        },
    )
    for annotation in spec.annotations:
        figure.add_annotation(
            text=(
                "[REDACTED ANNOTATION]"
                if prepared.redact_annotations
                else annotation.text
            ),
            x=None if prepared.redact_annotations else annotation.x,
            y=None if prepared.redact_annotations else annotation.y,
            showarrow=annotation.kind != "note",
        )
    return figure


def _render_matplotlib(prepared: _PreparedChart, spec: ChartSpec) -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    frame = prepared.frame
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if prepared.kind == "network":
        positions = _network_layout(frame)
        for row in frame.to_dict(orient="records"):
            source, target = str(row["source"]), str(row["target"])
            if source in positions and target in positions:
                axis.annotate(
                    "",
                    xy=positions[target],
                    xytext=positions[source],
                    arrowprops={"arrowstyle": "->", "color": "#777777"},
                )
        for node, (node_x, node_y) in positions.items():
            axis.scatter([node_x], [node_y], color=COLORBLIND_SAFE[0], s=80)
            axis.text(node_x, node_y + 0.07, node, ha="center")
        axis.axis("off")
    elif prepared.kind == "heatmap":
        if not frame.empty:
            matrix = frame.pivot(index=prepared.y, columns=prepared.x, values=prepared.color)
            image = axis.imshow(
                matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto"
            )
            axis.set_xticks(range(len(matrix.columns)), [str(v) for v in matrix.columns], rotation=45, ha="right")
            axis.set_yticks(range(len(matrix.index)), [str(v) for v in matrix.index])
            figure.colorbar(image, ax=axis, label="association")
    elif prepared.kind == "gate_matrix":
        colors = [
            {
                "pass": COLORBLIND_SAFE[2],
                "warn": COLORBLIND_SAFE[1],
                "skip": COLORBLIND_SAFE[4],
                "fail": COLORBLIND_SAFE[5],
                "escalate": COLORBLIND_SAFE[5],
            }.get(str(value), "#777777")
            for value in frame.get("status", [])
        ]
        axis.bar(frame.get("gate", []), frame.get("status_code", []), color=colors)
        axis.tick_params(axis="x", rotation=45)
    elif prepared.kind == "bar":
        if prepared.color and prepared.color in frame:
            pivot = frame.pivot_table(
                index=prepared.x,
                columns=prepared.color,
                values=prepared.y,
                aggfunc="first",
            )
            pivot.plot(kind="bar", ax=axis, color=list(COLORBLIND_SAFE))
        else:
            axis.bar(
                frame[prepared.x] if prepared.x in frame else [],
                frame[prepared.y] if prepared.y in frame else [],
                color=COLORBLIND_SAFE[0],
            )
        axis.tick_params(axis="x", rotation=45)
    elif prepared.kind == "line":
        if prepared.color and prepared.color in frame:
            for index, (group, subset) in enumerate(
                frame.groupby(prepared.color, observed=True)
            ):
                axis.plot(
                    subset[prepared.x],
                    subset[prepared.y],
                    marker="o",
                    label=str(group),
                    color=COLORBLIND_SAFE[index % len(COLORBLIND_SAFE)],
                )
            axis.legend()
        else:
            axis.plot(
                frame[prepared.x] if prepared.x in frame else [],
                frame[prepared.y] if prepared.y in frame else [],
                marker="o",
                color=COLORBLIND_SAFE[0],
            )
    elif prepared.kind == "scatter":
        axis.scatter(
            frame[prepared.x] if prepared.x in frame else [],
            frame[prepared.y] if prepared.y in frame else [],
            color=COLORBLIND_SAFE[0],
            alpha=0.75,
        )
        if prepared.x in frame and prepared.y in frame and len(frame) >= 2:
            x_values = pd.to_numeric(frame[prepared.x], errors="coerce")
            y_values = pd.to_numeric(frame[prepared.y], errors="coerce")
            valid = x_values.notna() & y_values.notna()
            if valid.sum() >= 2:
                slope, intercept = np.polyfit(x_values[valid], y_values[valid], 1)
                ordered = np.sort(x_values[valid])
                axis.plot(
                    ordered,
                    slope * ordered + intercept,
                    color=COLORBLIND_SAFE[5],
                    linewidth=1.5,
                )
    elif prepared.kind in ("histogram", "histogram_grouped"):
        if prepared.color and prepared.color in frame:
            for index, (group, subset) in enumerate(
                frame.groupby(prepared.color, observed=True)
            ):
                axis.hist(
                    subset[prepared.x],
                    bins=20,
                    alpha=0.5,
                    label=str(group),
                    color=COLORBLIND_SAFE[index % len(COLORBLIND_SAFE)],
                )
            axis.legend()
        else:
            axis.hist(
                frame[prepared.x] if prepared.x in frame else [],
                bins=20,
                color=COLORBLIND_SAFE[0],
                alpha=0.8,
            )
    elif prepared.kind in ("treatment_outcome", "box"):
        if prepared.x in frame and prepared.y in frame:
            groups = [
                subset[prepared.y].dropna().to_numpy()
                for _, subset in frame.groupby(prepared.x, observed=True)
            ]
            labels = [
                str(group) for group in frame[prepared.x].dropna().drop_duplicates()
            ]
            axis.boxplot(groups, labels=labels)
    axis.set_title(spec.title)
    axis.set_xlabel(prepared.x or "")
    axis.set_ylabel(prepared.y or "")
    return figure


class AutoChart:
    """Render a :class:`ChartSpec` using Plotly, Matplotlib, or data-only output."""

    def __init__(
        self,
        spec: Optional[ChartSpec] = None,
        *,
        backend: str = "auto",
        production: bool = False,
        allow_raw_values: bool = False,
    ) -> None:
        backend = str(backend).lower()
        if backend not in ("auto", "plotly", "matplotlib", "data"):
            raise ValueError("backend must be auto, plotly, matplotlib, or data")
        self.spec = spec
        self.backend = backend
        self.production = bool(production)
        self.allow_raw_values = bool(allow_raw_values)

    def _backend_order(self) -> list[str]:
        if self.backend == "auto":
            return ["plotly", "matplotlib", "data"]
        if self.backend == "plotly":
            return ["plotly", "matplotlib", "data"]
        if self.backend == "matplotlib":
            return ["matplotlib", "data"]
        return ["data"]

    def render(
        self,
        frame: pd.DataFrame,
        spec: Optional[ChartSpec] = None,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> RenderedChart:
        active_spec = spec or self.spec
        if active_spec is None:
            raise ValueError("AutoChart.render requires a ChartSpec")
        if not isinstance(active_spec, ChartSpec):
            active_spec = ChartSpec.from_dict(active_spec)  # type: ignore[arg-type]
        if self.production and self.allow_raw_values:
            active_spec.provenance.setdefault(
                "privacy_override",
                "Explicit allow_raw_values=True at renderer construction.",
            )
        prepared = _prepare(
            frame,
            active_spec,
            production=self.production,
            allow_raw_values=self.allow_raw_values,
            context=context,
        )
        warnings = list(prepared.warnings)
        artifact: Any = None
        selected_backend = "data"
        for backend in self._backend_order():
            try:
                if backend == "plotly":
                    if not _plotly_available():
                        raise ImportError("plotly is not installed")
                    artifact = _render_plotly(prepared, active_spec)
                    selected_backend = "plotly"
                    break
                if backend == "matplotlib":
                    if not _matplotlib_available():
                        raise ImportError("matplotlib is not installed")
                    artifact = _render_matplotlib(prepared, active_spec)
                    selected_backend = "matplotlib"
                    break
                selected_backend = "data"
                artifact = None
                break
            except Exception as exc:
                warnings.append(
                    f"{backend} renderer unavailable; falling back safely "
                    f"({type(exc).__name__}: {exc})."
                )
        provenance = {
            **dict(active_spec.provenance),
            "backend": selected_backend,
            "production": self.production,
            "aggregated": prepared.aggregated,
            "deterministic_sample": active_spec.deterministic_sample,
            "random_state": active_spec.random_state,
            "contains_raw_values": prepared.contains_raw_values,
            "causal_interpretation": "not_established",
        }
        return RenderedChart(
            spec=active_spec,
            backend=selected_backend,
            artifact=artifact,
            data_payload=prepared.payload,
            provenance=provenance,
            warnings=warnings,
            production=self.production,
        )

    def render_many(
        self,
        frame: pd.DataFrame,
        specs: Sequence[ChartSpec],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AutoChartReport:
        charts = [self.render(frame, spec, context=context) for spec in specs]
        return AutoChartReport(
            charts=charts,
            notes=[
                "All backends are headless; no GUI window is opened.",
                "Charts are descriptive and never establish causal identification.",
            ],
        )


__all__ = [
    "AutoChart",
    "COLORBLIND_SAFE",
    "available_backends",
]
