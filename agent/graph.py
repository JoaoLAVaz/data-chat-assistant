"""
LangGraph Workflow Definition

This module defines the complete statistical analysis agent workflow including:
- Node definitions and connections
- Routing logic for tool execution
- Graph compilation and agent creation
"""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from .state import AgentState
from .nodes.llm_node import llm_node
from .nodes.tools_exec_node import execute_tools_node
from .nodes.missing_data_node import missing_data_node


NEEDS_MISSING_PIPELINE = {"t_test", "anova_test", "correlation_test", "chi_square_test"}


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
            tool_names = {c["name"] for c in last.tool_calls}
        except Exception:
            tool_names = set()
        return "missing" if (tool_names & NEEDS_MISSING_PIPELINE) else "tools"
    return "end"


def compile_statistical_agent():
    """
    Compile the complete statistical analysis LangGraph agent.

    Nodes:
      - 'llm'     : tool-enabled LLM that decides what to run
      - 'missing' : missing-data preprocessing (only for certain tools)
      - 'tools'   : executes tool calls with access to state

    Flow:
      START → llm → (missing | tools | END)
      missing → tools → llm
      tools   → llm
    """

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("llm", llm_node)
    graph.add_node("missing", missing_data_node)   # NEW
    graph.add_node("tools", execute_tools_node)


    # Edges
    graph.add_edge(START, "llm")


    graph.add_conditional_edges(
        "llm",
        route_after_llm,
        {"missing": "missing", "tools": "tools", "end": END},
    )


    # After missing-data -> always run tools
    graph.add_edge("missing", "tools")

    # After tools execution, return to LLM for interpretation / next step
    graph.add_edge("tools", "llm")

    return graph.compile()



def create_agent():
    """Factory function to create a new statistical analysis agent instance."""
    return compile_statistical_agent()



# Main agent instance (can be imported directly)
agent = compile_statistical_agent()