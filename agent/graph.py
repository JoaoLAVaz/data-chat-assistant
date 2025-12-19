"""
LangGraph Workflow Definition

This module defines the complete statistical analysis agent workflow including:
- Node definitions and connections
- Routing logic for tool execution
- Graph compilation and agent creation
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from .state import AgentState
from .nodes.llm_node import llm_node
from .nodes.tools_exec_node import execute_tools_node
from .nodes.missing_data_node import missing_data_node
from .nodes.explainer_node import explainer_node


# Tools that should use the missing-data pipeline before running
NEEDS_MISSING_PIPELINE = {"t_test", "anova_test", "correlation_test", "chi_square_test"}

# Tools that produce a "final analysis tool JSON" appropriate for the detailed explainer.
# (Do NOT include recommend_tests here.)
EXPLAINER_ELIGIBLE_TOOLS = {
    "t_test",
    "anova_test",
    "correlation_test",
    "chi_square_test",
    "clustering_kmeans",
}


def route_after_llm(state: AgentState) -> str:
    """
    Decide where to go after the LLM runs:
      - If there are tool calls and at least one needs missing-data preprocessing → 'missing'
      - If there are tool calls but none need preprocessing → 'tools'
      - Otherwise → 'end'
    """
    last = state["messages"][-1] if state.get("messages") else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        try:
            tool_names = {c.get("name") for c in (last.tool_calls or []) if isinstance(c, dict)}
        except Exception:
            tool_names = set()
        return "missing" if (tool_names & NEEDS_MISSING_PIPELINE) else "tools"
    return "end"


def route_after_tools(state: AgentState) -> str:
    """
    Decide where to go after tools have executed:

      - If config['explainer']['use_detailed'] is True AND the last AI tool-call batch
        includes at least one "eligible" analysis tool → 'explainer'
      - Otherwise → 'llm'
    """
    cfg = (state.get("config") or {}).get("explainer") or {}
    use_detailed = bool(cfg.get("use_detailed", False))
    if not use_detailed:
        return "llm"

    # Find the most recent AIMessage that issued tool calls (this corresponds to the tool batch we just ran)
    last_ai = None
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            last_ai = m
            break

    if not last_ai:
        return "llm"

    try:
        tool_names = {c.get("name") for c in (last_ai.tool_calls or []) if isinstance(c, dict)}
    except Exception:
        tool_names = set()

    # Only run the detailed explainer for real analysis tools (not recommend_tests, etc.)
    return "explainer" if (tool_names & EXPLAINER_ELIGIBLE_TOOLS) else "llm"


def compile_statistical_agent():
    """
    Compile the complete statistical analysis LangGraph agent.

    Nodes:
      - 'llm'       : tool-enabled LLM that decides what to run
      - 'missing'   : missing-data preprocessing (only for certain tools)
      - 'tools'     : executes tool calls with access to state
      - 'explainer' : optional detailed explainer on the *final analysis tool JSON*
                      (fine-tuned Qwen model, controlled by config flag)

    Flow:
      START   → llm → (missing | tools | END)
      missing → tools
      tools   → (explainer | llm)
      explainer → END
    """
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("llm", llm_node)
    graph.add_node("missing", missing_data_node)
    graph.add_node("tools", execute_tools_node)
    graph.add_node("explainer", explainer_node)

    # Edges
    graph.add_edge(START, "llm")

    # After LLM: decide if we need missing-data preprocessing, straight tools, or just stop
    graph.add_conditional_edges(
        "llm",
        route_after_llm,
        {"missing": "missing", "tools": "tools", "end": END},
    )

    # After missing-data -> always run tools
    graph.add_edge("missing", "tools")

    # After tools -> either detailed explainer (eligible tools only) or back to base LLM
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"explainer": "explainer", "llm": "llm"},
    )

    # Detailed explainer produces the final user-facing explanation → end this run
    graph.add_edge("explainer", END)

    return graph.compile()


def create_agent():
    """Factory function to create a new statistical analysis agent instance."""
    return compile_statistical_agent()


# Main agent instance (can be imported directly)
agent = compile_statistical_agent()
