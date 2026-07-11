"""autocausal.grail — embellished Kineteq GRAIL adaptation for AutoCausal.

**GRAIL** (Kineteq): Generative Reflective Agentic Imputation Loop —
``grail_impute`` → ``grail_compose`` → ``grail_run`` / fold / memory surfaces
exposed on the Kineteq MCP bus and kernel-os ``grail`` strategy.

This package wraps live Kineteq when importable/configured, and always
provides a **rich offline stub** with the same primitive names. The stub is
an AutoCausal embellishment (clearer API, reports, insight/skilling/MCP hooks)
— **not** a claim of parity with full live GRAIL.

See ``docs/GRAIL.md``.
"""

from __future__ import annotations

from autocausal.grail.adapter import (
    GrailEngine,
    grail_backend_status,
    kineteq_grail_available,
)
from autocausal.grail.guide import GrailGuide, KineteqGrailGuide
from autocausal.grail.hooks import (
    grail_tool_schemas,
    insight_grail_step,
    register_grail_skilling_tools,
    try_register_mcp_tools,
)
from autocausal.grail.mcp_tools import TOOL_NAMES, dispatch_grail_tool
from autocausal.grail.stub import GrailStub
from autocausal.grail.types import (
    EPISTEMIC,
    Assumption,
    CycleTrace,
    ExpertChain,
    ExpertStep,
    FoldDiagnosis,
    GrailReport,
    GraphMemoryNode,
    ImputationAudit,
)

__all__ = [
    "EPISTEMIC",
    "Assumption",
    "ImputationAudit",
    "ExpertStep",
    "ExpertChain",
    "FoldDiagnosis",
    "CycleTrace",
    "GraphMemoryNode",
    "GrailReport",
    "GrailStub",
    "GrailEngine",
    "GrailGuide",
    "KineteqGrailGuide",
    "grail_backend_status",
    "kineteq_grail_available",
    "insight_grail_step",
    "register_grail_skilling_tools",
    "try_register_mcp_tools",
    "grail_tool_schemas",
    "dispatch_grail_tool",
    "TOOL_NAMES",
    "run_grail",
]


def run_grail(
    goal: str,
    *,
    context: dict | None = None,
    max_cycles: int = 2,
    chain_length: int = 3,
    domain: str = "causal",
    prefer_live: bool = True,
) -> GrailReport:
    """Convenience: ``GrailEngine(...).run(...)``."""
    return GrailEngine(domain=domain, prefer_live=prefer_live).run(
        goal,
        context=context,
        max_cycles=max_cycles,
        chain_length=chain_length,
        domain=domain,
    )
