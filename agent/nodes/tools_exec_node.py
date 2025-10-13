"""
Tools Execution Node

This module handles the execution of statistical analysis tools called by the LLM.
It bridges between the tool schemas (what the LLM sees) and the actual statistical
implementations in the analysis/ directory.
"""

import json
from typing import Dict, Any, Callable
from langchain_core.messages import ToolMessage

from ..state import AgentState
from analysis.ttest.exec import t_test_impl
from analysis.anova.exec import anova_impl
from analysis.correlation.exec import correlation_impl
from analysis.chisquared.exec import chi_square_impl
from analysis.recommend.exec import recommend_tests_impl


# Registry mapping tool names to their implementations
STATE_AWARE_IMPLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "t_test": t_test_impl,
    "anova_test": anova_impl,
    "correlation_test": correlation_impl,
    "chi_square_test": chi_square_impl,
    "recommend_tests": recommend_tests_impl
}


def execute_tools_node(state: AgentState) -> AgentState:
    """
    Execute statistical analysis tools called by the LLM.

    This node:
    1. Extracts tool calls from the last AI message
    2. Looks up the appropriate implementation for each tool
    3. Executes the tool with access to the (possibly preprocessed) state data
    4. Returns tool results as ToolMessages for the LLM to interpret

    Args:
        state: Current agent state containing messages, dataset, and metadata

    Returns:
        Updated state with ToolMessages containing execution results
    """
    msgs = state["messages"]
    last = msgs[-1]
    tool_msgs = []

    # Prefer the cleaned subset produced by the missing-data node (if present)
    analysis_ctx = state.get("analysis_context") or {}
    df_use = analysis_ctx.get("working_df", state["df"])
    missing_report = analysis_ctx.get("missing_report")

    # Process each tool call from the last AI message
    for call in getattr(last, "tool_calls", []) or []:
        name = call["name"]
        args = call.get("args", {}) or {}

        # Check if we have an implementation for this tool
        if name not in STATE_AWARE_IMPLS:
            result: Dict[str, Any] = {
                "error": f"Tool '{name}' is not yet implemented",
                "available_tools": list(STATE_AWARE_IMPLS.keys()),
            }
        else:
            try:
                # Execute the tool implementation with access to state
                result = STATE_AWARE_IMPLS[name](
                    df=df_use,
                    metadata=state["metadata"],
                    **args,
                )

                # Attach missing-data transparency if available
                if isinstance(result, dict) and missing_report is not None:
                    result.setdefault("missing_data", missing_report)

            except Exception as e:
                result = {
                    "error": f"Tool '{name}' execution failed: {str(e)}",
                    "tool_name": name,
                    "arguments": args,
                }

        # Create ToolMessage with JSON-encoded results
        tool_msgs.append(
            ToolMessage(
                content=json.dumps(result, indent=2),  # Pretty JSON for better LLM parsing
                tool_call_id=call["id"],
            )
        )

    # Return updated state with tool results
    # The add_messages reducer will append these to the conversation
    return {"messages": tool_msgs}
