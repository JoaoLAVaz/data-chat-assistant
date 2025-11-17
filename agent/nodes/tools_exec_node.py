"""
Tools Execution Node

This module handles the execution of statistical analysis tools called by the LLM.
It bridges between the tool schemas (what the LLM sees) and the actual statistical
implementations in the analysis/ directory.
"""

import json
import inspect
from typing import Dict, Any, Callable, Set, Optional
from langchain_core.messages import ToolMessage

from ..state import AgentState
from analysis.ttest.exec import t_test_impl
from analysis.anova.exec import anova_impl
from analysis.correlation.exec import correlation_impl
from analysis.chisquared.exec import chi_square_impl
from analysis.recommend.exec import recommend_tests_impl
from analysis.clustering.exec import clustering_impl


# Registry mapping tool names to their implementations
STATE_AWARE_IMPLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "t_test": t_test_impl,
    "anova_test": anova_impl,
    "correlation_test": correlation_impl,
    "chi_square_test": chi_square_impl,
    "recommend_tests": recommend_tests_impl,
    "clustering_kmeans": clustering_impl,
}


def _required_columns_for_call(tool_name: str, args: Dict[str, Any]) -> Set[str]:
    """
    Infer which columns are needed for this tool call based on its schema args.
    This is used to decide whether we can reuse a scoped working_df or should fall back to the full df.
    """
    cols: Set[str] = set()
    n = tool_name

    # t-test / anova
    if n == "t_test" or n == "anova_test":
        gc = args.get("group_col")
        vc = args.get("value_col")
        if isinstance(gc, str):
            cols.add(gc)
        if isinstance(vc, str):
            cols.add(vc)

    # correlation
    elif n == "correlation_test":
        v1 = args.get("var1")
        v2 = args.get("var2")
        if isinstance(v1, str):
            cols.add(v1)
        if isinstance(v2, str):
            cols.add(v2)

    # chi-square
    elif n == "chi_square_test":
        v1 = args.get("var1")
        v2 = args.get("var2")
        if isinstance(v1, str):
            cols.add(v1)
        if isinstance(v2, str):
            cols.add(v2)

    # clustering
    elif n == "clustering_kmeans":
        feats = args.get("features")
        if isinstance(feats, (list, tuple)):
            for f in feats:
                if isinstance(f, str):
                    cols.add(f)

    # recommend_tests doesn't require columns
    return cols


def _filter_kwargs_for_func(func: Callable[..., Any], base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs that the function can accept (by name).
    Prevents errors when we pass optional fields like 'missing_report' to functions
    that don't declare them.
    """
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())
    # If function has **kwargs, accept everything
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return dict(base_kwargs)
    return {k: v for k, v in base_kwargs.items() if k in accepted}


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

    # Context possibly set by the missing-data node
    analysis_ctx = state.get("analysis_context") or {}
    scoped_df = analysis_ctx.get("working_df")
    scoped_missing_report = analysis_ctx.get("missing_report")

    # Process each tool call from the last AI message
    for call in getattr(last, "tool_calls", []) or []:
        name = call["name"]
        args = call.get("args", {}) or {}

        # Decide which dataframe to use for THIS call
        required_cols = _required_columns_for_call(name, args)
        using_scoped = False
        df_for_call = state["df"]  # default: full df
        missing_report_for_call: Optional[Dict[str, Any]] = None

        if scoped_df is not None and required_cols:
            # Use scoped df only if it contains ALL required columns
            if required_cols.issubset(set(scoped_df.columns)):
                df_for_call = scoped_df
                using_scoped = True
                missing_report_for_call = scoped_missing_report

        # If there are no specific required cols (e.g., recommend_tests), prefer full df
        # (df_for_call already set to state["df"])

        if name not in STATE_AWARE_IMPLS:
            result: Dict[str, Any] = {
                "error": f"Tool '{name}' is not yet implemented",
                "available_tools": list(STATE_AWARE_IMPLS.keys()),
            }
        else:
            impl = STATE_AWARE_IMPLS[name]
            try:
                # Base kwargs common to all tools
                call_kwargs: Dict[str, Any] = {
                    "df": df_for_call,
                    "metadata": state["metadata"],
                    **args,
                }
                # Attach missing_report only if we used scoped df
                if using_scoped and (scoped_missing_report is not None):
                    call_kwargs["missing_report"] = scoped_missing_report

                # Filter kwargs to match impl signature
                safe_kwargs = _filter_kwargs_for_func(impl, call_kwargs)

                # Execute
                result = impl(**safe_kwargs)

                # If the impl didn't include missing_data_report but we used scoped df, attach it
                if (
                    isinstance(result, dict)
                    and using_scoped
                    and (scoped_missing_report is not None)
                    and "missing_data_report" not in result
                ):
                    result["missing_data_report"] = scoped_missing_report

                # (Optional) annotate which df was used — helpful for debugging
                if isinstance(result, dict):
                    result.setdefault("_execution_context", {})["df_scope"] = "scoped" if using_scoped else "full"

            except Exception as e:
                result = {
                    "error": f"Tool '{name}' execution failed: {str(e)}",
                    "tool_name": name,
                    "arguments": args,
                }

        # Create ToolMessage with JSON-encoded results
        tool_msgs.append(
            ToolMessage(
                content=json.dumps(result, indent=2),
                tool_call_id=call["id"],
            )
        )

    # Debug log
    print(f"tools_exec return: {tool_msgs}")

    # Return updated state with tool results
    return {"messages": tool_msgs}
