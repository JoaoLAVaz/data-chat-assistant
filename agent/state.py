"""
Agent State Definition

This module defines the state structure for the statistical analysis LangGraph agent.
The state carries all necessary information through the workflow including conversation
messages, dataset context, and analysis results.
"""

from typing import TypedDict, Annotated, Sequence, Any, Dict, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State structure for the statistical analysis agent.
    
    This state is passed between all nodes in the LangGraph workflow and maintains
    all context needed for statistical analysis and conversation continuity.
    """
    
    # Conversation messages - aggregated by add_messages reducer
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Dataset and metadata
    df: Any  # pandas.DataFrame - the actual dataset being analyzed
    metadata: Dict[str, str]  # Column name -> type mapping (categorical/numerical/other)
    
    # Analysis workflow context (optional - for complex workflows)
    analysis_context: Optional[Dict[str, Any]]  # Store intermediate results, workflow state
    
    # Configuration (optional - for workflow parameters)
    config: Optional[Dict[str, Any]]  # Analysis parameters, significance levels, etc.


# Type aliases for cleaner imports
StatisticalAgentState = AgentState



#TypedDict is total by default, so static type checkers will treat all keys as required. At runtime it won’t break, 
# but editors/mypy/pyright may complain if you initialize the state without df, metadata, analysis_context, or config.