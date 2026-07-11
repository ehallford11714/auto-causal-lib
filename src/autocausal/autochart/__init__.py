"""Backend-neutral chart specifications and headless renderers."""

from autocausal.autochart.renderer import AutoChart, COLORBLIND_SAFE, available_backends
from autocausal.autochart.report import AutoChartReport, CHART_CAVEAT, RenderedChart
from autocausal.autochart.specs import (
    AccessibilitySpec,
    ChartAnnotation,
    ChartFilter,
    ChartSpec,
    ChartSpecError,
    FILTER_OPERATORS,
    SUPPORTED_CHART_TYPES,
)

__all__ = [
    "AccessibilitySpec",
    "AutoChart",
    "AutoChartReport",
    "CHART_CAVEAT",
    "COLORBLIND_SAFE",
    "ChartAnnotation",
    "ChartFilter",
    "ChartSpec",
    "ChartSpecError",
    "FILTER_OPERATORS",
    "RenderedChart",
    "SUPPORTED_CHART_TYPES",
    "available_backends",
]
