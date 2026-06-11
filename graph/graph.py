"""LangGraph ``StateGraph`` definition and compilation.

    planner -> researcher -> critic -> (should_retry) -> researcher | writer -> END

The graph is compiled with a ``MemorySaver`` checkpointer so each run's state is
persisted under its ``thread_id`` (we use the ``run_id``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from loguru import logger

if TYPE_CHECKING:  # import path differs across langgraph versions — annotation only
    from langgraph.graph.state import CompiledStateGraph

from agents.critic import critic_node
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.writer import writer_node
from graph.edges import should_retry
from graph.state import AgentState


def build_graph() -> StateGraph:
    """Construct the (uncompiled) research ``StateGraph``."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"researcher": "researcher", "writer": "writer"},
    )
    graph.add_edge("writer", END)

    return graph


@lru_cache(maxsize=1)
def get_compiled_graph() -> "CompiledStateGraph":
    """Build and compile the graph once, with an in-memory checkpointer.

    Returns:
        A compiled, runnable ``StateGraph``.
    """
    compiled = build_graph().compile(checkpointer=MemorySaver())
    logger.info("Compiled research graph (planner -> researcher -> critic -> writer)")
    return compiled
