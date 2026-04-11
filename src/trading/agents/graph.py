from typing import Any
from trading.core.models import MarketState
from langgraph.graph import END, START, StateGraph

from trading.agents.htf_agent import run_htf_agent
from trading.agents.ltf_agent import run_ltf_agent


def should_proceed_to_ltf(state: MarketState) -> str:
    if not state.points_of_interest:
        return "end"
    return "ltf_agent"


def build_graph() -> Any:
    graph = StateGraph(MarketState)

    graph.add_node("htf_agent", run_htf_agent)
    graph.add_node("ltf_agent", run_ltf_agent)

    graph.add_edge(START, "htf_agent")
    graph.add_conditional_edges(
        "htf_agent",
        should_proceed_to_ltf,
        {
            "ltf_agent": "ltf_agent",
            "end": END,
        },
    )
    graph.add_edge("ltf_agent", END)

    return graph.compile()
