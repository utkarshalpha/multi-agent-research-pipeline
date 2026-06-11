"""Planner agent: decomposes the user's query into 3-5 focused sub-questions."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from loguru import logger

from config import ainvoke_with_retry, get_llm
from graph.state import AgentState
from memory.redis_store import redis_store
from schemas.models import SubQuestions

SYSTEM_PROMPT = (
    "You are a research planning expert. Decompose the user's query into 3-5 "
    "focused sub-questions that together would fully answer the main query. "
    "Each sub-question must be self-contained, specific, and independently "
    "researchable. Return ONLY valid JSON matching the SubQuestions schema."
)


@traceable(name="planner", run_type="chain")
async def planner_node(state: AgentState) -> dict:
    """Plan the research by decomposing the query into sub-questions.

    Args:
        state: The current graph state; must contain ``query``.

    Returns:
        A partial state update with ``sub_questions`` populated.
    """
    query = state["query"]
    run_id = state.get("run_id", "")
    logger.info("[planner] decomposing query: {!r}", query)

    structured_llm = get_llm().with_structured_output(SubQuestions)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Main query: {query}\n\nDecompose this into 3-5 sub-questions."),
    ]

    result: SubQuestions = await ainvoke_with_retry(structured_llm, messages)
    sub_questions = result.questions
    logger.info("[planner] produced {} sub-questions", len(sub_questions))

    await redis_store.save(run_id, "sub_questions", sub_questions)
    return {"sub_questions": sub_questions}
