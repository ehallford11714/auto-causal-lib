"""StateFlow / LangGraph soft cyclic runtime for agentic causal nodes.

Design inspiration (not a reimplementation):
- StateFlow (arXiv:2403.11322) — FSM-style agent state transitions
- LangGraph — cyclic graph orchestration (soft optional dependency)

Offline FSM stub always works. If ``langgraph`` is installed, an optional
compiled graph mirrors the same node sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from autocausal.agentic.state import LoopState

__all__ = [
    "NODE_ORDER",
    "GraphRuntime",
    "RuntimeResult",
    "langgraph_available",
]

NodeFn = Callable[[LoopState], LoopState]

# hypothesize → skill/tool → validate → compact → persist → route
NODE_ORDER: tuple[str, ...] = (
    "hypothesize",
    "skill",
    "validate",
    "compact",
    "persist",
    "route",
)


def langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class RuntimeResult:
    state: LoopState
    backend: str = "fsm"  # fsm | langgraph
    nodes_run: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "nodes_run": list(self.nodes_run),
            "notes": list(self.notes),
            "route": self.state.route,
            "round": self.state.round,
        }


class GraphRuntime:
    """Cyclic node runner with soft LangGraph backend.

    Register callables for each node name. ``run_cycle`` executes one full
    pass through ``NODE_ORDER`` (or a custom order). ``run_until`` repeats
    cycles while ``state.route == "continue"`` and ``round < max_rounds``.
    """

    def __init__(
        self,
        nodes: Optional[dict[str, NodeFn]] = None,
        *,
        prefer_langgraph: bool = True,
    ) -> None:
        self.nodes: dict[str, NodeFn] = dict(nodes or {})
        self.prefer_langgraph = bool(prefer_langgraph)
        self._lg_app: Any = None
        self.notes: list[str] = []

    def register(self, name: str, fn: NodeFn) -> None:
        self.nodes[name] = fn

    def _run_fsm_cycle(
        self, state: LoopState, order: Sequence[str]
    ) -> tuple[LoopState, list[str]]:
        ran: list[str] = []
        for name in order:
            fn = self.nodes.get(name)
            if fn is None:
                state.notes.append(f"Missing node `{name}` — skipped.")
                continue
            state = fn(state)
            state.record_node(name)
            ran.append(name)
            if name == "route" and state.route == "stop":
                break
        return state, ran

    def _try_build_langgraph(self, order: Sequence[str]) -> bool:
        if not self.prefer_langgraph or not langgraph_available():
            return False
        try:
            from langgraph.graph import END, StateGraph  # type: ignore

            # Use a plain dict channel for soft compatibility
            g: Any = StateGraph(dict)

            def _wrap(name: str) -> Callable[[dict], dict]:
                fn = self.nodes[name]

                def _inner(d: dict) -> dict:
                    st = d.get("_state")
                    if not isinstance(st, LoopState):
                        return d
                    st = fn(st)
                    st.record_node(name)
                    d["_state"] = st
                    d["route"] = st.route
                    d["nodes_run"] = list(d.get("nodes_run") or []) + [name]
                    return d

                return _inner

            for name in order:
                if name not in self.nodes:
                    continue
                g.add_node(name, _wrap(name))

            # Linear edges with conditional stop after route
            present = [n for n in order if n in self.nodes]
            if not present:
                return False
            g.set_entry_point(present[0])
            for a, b in zip(present, present[1:]):
                g.add_edge(a, b)

            last = present[-1]

            def _after_last(d: dict) -> str:
                if d.get("route") == "stop":
                    return END
                return END  # one cycle per invoke; outer loop handles repeats

            g.add_conditional_edges(last, _after_last)
            self._lg_app = g.compile()
            self.notes.append("LangGraph soft runtime compiled.")
            return True
        except Exception as e:
            self.notes.append(f"LangGraph soft-fail: {type(e).__name__}: {e}")
            self._lg_app = None
            return False

    def run_cycle(
        self,
        state: LoopState,
        *,
        order: Optional[Sequence[str]] = None,
    ) -> RuntimeResult:
        order = tuple(order or NODE_ORDER)
        notes = list(self.notes)

        if self.prefer_langgraph and self._lg_app is None:
            self._try_build_langgraph(order)
            notes = list(self.notes)

        if self._lg_app is not None:
            try:
                payload = {"_state": state, "route": state.route, "nodes_run": []}
                out = self._lg_app.invoke(payload)
                st = out.get("_state", state)
                ran = list(out.get("nodes_run") or [])
                return RuntimeResult(
                    state=st, backend="langgraph", nodes_run=ran, notes=notes
                )
            except Exception as e:
                notes.append(f"LangGraph invoke soft-fail → FSM: {type(e).__name__}")

        state, ran = self._run_fsm_cycle(state, order)
        return RuntimeResult(state=state, backend="fsm", nodes_run=ran, notes=notes)

    def run_until(
        self,
        state: LoopState,
        *,
        max_rounds: Optional[int] = None,
        order: Optional[Sequence[str]] = None,
        on_cycle_end: Optional[Callable[[LoopState], None]] = None,
    ) -> RuntimeResult:
        """Repeat cycles while route=continue and round < max_rounds."""
        cap = int(max_rounds if max_rounds is not None else state.max_rounds)
        all_nodes: list[str] = []
        notes: list[str] = []
        backend = "fsm"

        while state.round < cap and state.route == "continue":
            result = self.run_cycle(state, order=order)
            state = result.state
            backend = result.backend
            all_nodes.extend(result.nodes_run)
            notes.extend(result.notes)
            state.round_history.append(state.snapshot_round())
            if on_cycle_end is not None:
                try:
                    on_cycle_end(state)
                except Exception as e:
                    notes.append(f"on_cycle_end soft-fail: {type(e).__name__}")
            if state.route == "stop":
                break
            state.round += 1
            # Reset per-round scratch that should not accumulate unboundedly
            state.tool_traces = []
            state.hypotheses = []
            state.validation = {}

        if state.round >= cap and state.route == "continue":
            state.route = "stop"
            state.stop_reason = state.stop_reason or f"Reached max_rounds={cap}."

        return RuntimeResult(
            state=state, backend=backend, nodes_run=all_nodes, notes=notes
        )
