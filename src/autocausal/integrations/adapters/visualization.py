"""AutoChart-connected local visualization adapters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from autocausal.integrations.adapters.base import LazyAdapter, bounded_int


class AutoChartBackendAdapter(LazyAdapter):
    def __init__(
        self,
        integration_id: str,
        module_name: str,
        package_name: str,
        backend: str,
        *,
        supports_dag: bool = False,
    ) -> None:
        self.integration_id = integration_id
        self.module_name = module_name
        self.package_name = package_name
        self.backend = backend
        self.capabilities = (
            ("viz.chart", "viz.dag") if supports_dag else ("viz.chart",)
        )
        self.id = f"{integration_id}.autochart"

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability == "viz.chart":
            return self.render_chart(**kwargs)
        if capability == "viz.dag" and "viz.dag" in self.capabilities:
            return self.render_dag(**kwargs)
        raise KeyError(capability)

    def render_chart(
        self,
        *,
        frame: Any,
        spec: Any,
        production: bool = False,
        allow_raw_values: bool = False,
        context: Optional[Mapping[str, Any]] = None,
        **_: Any,
    ) -> Any:
        from autocausal.autochart import AutoChart

        renderer = AutoChart(
            spec=spec,
            backend=self.backend,
            production=bool(production),
            allow_raw_values=bool(allow_raw_values),
        )
        return renderer.render(frame, context=context)

    @staticmethod
    def render_dag(
        *,
        edges: Sequence[Any],
        nodes: Optional[Sequence[str]] = None,
        title: str = "Causal graph (exploratory)",
        **_: Any,
    ) -> Any:
        import plotly.graph_objects as go

        normalized: list[tuple[str, str]] = []
        node_set = set(str(item) for item in (nodes or ()))
        for edge in edges:
            if isinstance(edge, Mapping):
                source, target = str(edge.get("source")), str(edge.get("target"))
            else:
                source, target = str(edge[0]), str(edge[1])
            node_set.update((source, target))
            normalized.append((source, target))
        ordered = sorted(node_set)
        if len(ordered) > 2_000:
            raise ValueError("DAG render exceeds 2,000-node cap")
        count = max(len(ordered), 1)
        positions = {
            node: (
                math.cos(2.0 * math.pi * index / count),
                math.sin(2.0 * math.pi * index / count),
            )
            for index, node in enumerate(ordered)
        }
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for source, target in normalized:
            edge_x.extend([positions[source][0], positions[target][0], None])
            edge_y.extend([positions[source][1], positions[target][1], None])
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 1, "color": "#6b7280"},
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[positions[node][0] for node in ordered],
                y=[positions[node][1] for node in ordered],
                text=ordered,
                mode="markers+text",
                textposition="top center",
                marker={"size": 12, "color": "#2563eb"},
                hovertemplate="%{text}<extra></extra>",
            )
        )
        figure.update_layout(
            title=title,
            showlegend=False,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "Exploratory graph; edges do not establish causality.",
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": -0.08,
                }
            ],
        )
        return figure


class NetworkXAdapter(LazyAdapter):
    id = "networkx.dag"
    integration_id = "networkx"
    module_name = "networkx"
    package_name = "networkx"
    capabilities = ("viz.dag",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "viz.dag":
            raise KeyError(capability)
        return self.dag(**kwargs)

    def dag(
        self,
        *,
        edges: Sequence[Any],
        nodes: Optional[Sequence[str]] = None,
        layout: str = "spring",
        random_state: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        nx = self._module()
        graph = nx.DiGraph()
        graph.add_nodes_from(str(item) for item in (nodes or ()))
        for edge in edges:
            if isinstance(edge, Mapping):
                source, target = str(edge.get("source")), str(edge.get("target"))
                attrs = {
                    str(key): value
                    for key, value in edge.items()
                    if key not in ("source", "target")
                }
            else:
                source, target = str(edge[0]), str(edge[1])
                attrs = {}
            graph.add_edge(source, target, **attrs)
        if graph.number_of_nodes() > 10_000:
            raise ValueError("networkx DAG exceeds 10,000-node cap")
        selected = str(layout).lower()
        if selected == "spring":
            positions = nx.spring_layout(graph, seed=int(random_state))
        elif selected == "circular":
            positions = nx.circular_layout(graph)
        elif selected == "shell":
            positions = nx.shell_layout(graph)
        else:
            raise ValueError("layout must be spring, circular, or shell")
        return {
            "graph": graph,
            "positions": {
                str(node): [float(value[0]), float(value[1])]
                for node, value in positions.items()
            },
            "is_dag": bool(nx.is_directed_acyclic_graph(graph)),
            "n_nodes": int(graph.number_of_nodes()),
            "n_edges": int(graph.number_of_edges()),
            "caveat": "Graph structure is exploratory; it is not causal proof.",
        }


class KaleidoAdapter(LazyAdapter):
    id = "kaleido.export"
    integration_id = "kaleido"
    module_name = "kaleido"
    package_name = "kaleido"
    capabilities = ("viz.export",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "viz.export":
            raise KeyError(capability)
        return self.export(**kwargs)

    @staticmethod
    def export(
        *,
        figure: Any,
        path: str | Path,
        width: int = 1_200,
        height: int = 800,
        scale: float = 1.0,
        **_: Any,
    ) -> dict[str, Any]:
        output = Path(path)
        if output.suffix.lower() not in (".png", ".svg", ".pdf", ".webp"):
            raise ValueError("Kaleido export path must end in png, svg, pdf, or webp")
        if not output.parent.exists():
            raise FileNotFoundError(
                "export parent directory must already exist; it is not auto-created"
            )
        resolved_width = bounded_int(
            width,
            default=1_200,
            minimum=100,
            maximum=5_000,
            name="width",
        )
        resolved_height = bounded_int(
            height,
            default=800,
            minimum=100,
            maximum=5_000,
            name="height",
        )
        resolved_scale = float(scale)
        if not 0.25 <= resolved_scale <= 4.0:
            raise ValueError("scale must be between 0.25 and 4")
        figure.write_image(
            str(output),
            width=resolved_width,
            height=resolved_height,
            scale=resolved_scale,
        )
        return {
            "path": str(output),
            "format": output.suffix.lower().lstrip("."),
            "width": resolved_width,
            "height": resolved_height,
            "scale": resolved_scale,
        }


def visualization_adapters() -> tuple[LazyAdapter, ...]:
    return (
        AutoChartBackendAdapter(
            "plotly",
            "plotly",
            "plotly",
            "plotly",
            supports_dag=True,
        ),
        AutoChartBackendAdapter(
            "matplotlib",
            "matplotlib",
            "matplotlib",
            "matplotlib",
        ),
        NetworkXAdapter(),
        KaleidoAdapter(),
    )


__all__ = [
    "AutoChartBackendAdapter",
    "KaleidoAdapter",
    "NetworkXAdapter",
    "visualization_adapters",
]
