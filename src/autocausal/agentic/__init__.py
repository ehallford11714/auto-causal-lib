"""autocausal.agentic — SLM-guided agentic causal loop with SOTA-inspired memory.

Library-first cyclic research loop::

    hypothesize → skill/tool → validate → compact → persist → route

Design inspiration (cited as inspiration, not paper reimplementations):
ACON, MEM1/A-MEM, StateFlow, HippoRAG/Mem0. Soft optional: langgraph,
chromadb, faiss, HuggingFace SLM.

Example::

    from autocausal.agentic import AgenticCausalLoop, run_agentic_loop
    from autocausal import load_dataset

    report = run_agentic_loop(load_dataset("iris"), max_rounds=2, use_slm=False)
    print(report.to_markdown())
"""

from __future__ import annotations

from autocausal.agentic.compact import CompactBundle, Compactor
from autocausal.agentic.graph_runtime import (
    NODE_ORDER,
    GraphRuntime,
    RuntimeResult,
    langgraph_available,
)
from autocausal.agentic.langgraph_chain import (
    SLM_CHAIN_NODES,
    SLMChainReport,
    SLMLangGraphChain,
    run_slm_langgraph_loop,
)
from autocausal.agentic.loop import AgenticCausalLoop, run_agentic_loop
from autocausal.agentic.memory import (
    AgentMemory,
    EpisodicMemory,
    MemoryItem,
    WorkingMemory,
)
from autocausal.agentic.persist import EpisodeStore, load_episodes, persist_episode
from autocausal.agentic.report import AgenticLoopReport
from autocausal.agentic.state import EPISTEMIC, Hypothesis, LoopState
from autocausal.agentic.vector_memory import (
    VectorRecord,
    VectorStoreMemory,
    make_vector_memory,
)

__all__ = [
    "AgenticCausalLoop",
    "run_agentic_loop",
    "SLMLangGraphChain",
    "SLMChainReport",
    "run_slm_langgraph_loop",
    "SLM_CHAIN_NODES",
    "AgenticLoopReport",
    "LoopState",
    "Hypothesis",
    "EPISTEMIC",
    "WorkingMemory",
    "EpisodicMemory",
    "AgentMemory",
    "MemoryItem",
    "Compactor",
    "CompactBundle",
    "GraphRuntime",
    "RuntimeResult",
    "NODE_ORDER",
    "langgraph_available",
    "VectorStoreMemory",
    "VectorRecord",
    "make_vector_memory",
    "EpisodeStore",
    "persist_episode",
    "load_episodes",
]
