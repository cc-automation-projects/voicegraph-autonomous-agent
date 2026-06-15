from __future__ import annotations

import logging

from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from src.orchestrator.checkpointer import RedisCheckpointer
from src.orchestrator.nodes import (
    execute_calls_node,
    reflect_node,
    schedule_calls_node,
    score_candidates_node,
    should_continue_batch,
    should_continue_campaign,
)
from src.orchestrator.state import AgentState

logger = logging.getLogger(__name__)


def build_campaign_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("score_candidates", score_candidates_node)
    workflow.add_node("schedule_calls", schedule_calls_node)
    workflow.add_node("execute_calls", execute_calls_node)
    workflow.add_node("reflect", reflect_node)

    workflow.set_entry_point("score_candidates")

    workflow.add_conditional_edges(
        "score_candidates",
        should_continue_campaign,
        {
            "continue": "schedule_calls",
            "end": END,
        }
    )

    workflow.add_edge("schedule_calls", "execute_calls")
    workflow.add_edge("execute_calls", "reflect")

    workflow.add_conditional_edges(
        "reflect",
        should_continue_batch,
        {
            "continue": "schedule_calls",
            "end": END,
        }
    )

    return workflow.compile(checkpointer=checkpointer)


def create_campaign_graph() -> StateGraph:
    checkpointer = RedisCheckpointer()
    return build_campaign_graph(checkpointer=checkpointer)
