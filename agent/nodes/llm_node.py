"""
LLM Node Implementation

This module contains the LLM node that processes user requests and decides when to call tools.
The LLM has access to tool schemas and can make intelligent decisions about which statistical
analyses to perform based on the dataset context.
"""

from langchain_core.messages import SystemMessage
from ..state import AgentState
from ..llm import get_llm_with_tools


def llm_node(state: AgentState) -> AgentState:
    """
    LLM node that processes user messages and determines appropriate actions.
    
    This node:
    1. Takes the current conversation state
    2. Applies a system prompt for statistical analysis context
    3. Invokes the tool-enabled LLM
    4. Returns the LLM's response (which may include tool calls)
    
    Args:
        state: Current agent state containing messages, dataset, and metadata
        
    Returns:
        Updated state with the LLM's response message
    """
    
    # Create system prompt for statistical analysis context
    system_prompt = SystemMessage(
        content=
        """You are a statistical analysis assistant. Use the available tools to analyze the dataset and explain results in plain English.
        Key guidelines:
        - Use the dataset information provided in the conversation to make smart column choices
        - When users ask for statistical comparisons, identify appropriate grouping and outcome variables
        - Run statistical tests when requested and interpret the results clearly
        - If you need clarification about the user's intent, ask specific questions
        """
    )
    
    # Get the tool-enabled LLM
    llm_model = get_llm_with_tools()
    
    # Invoke the LLM with system prompt + conversation history
    response = llm_model.invoke([system_prompt] + state["messages"])

    #print(f"llm_node: {response}")
    
    # Return updated state with the LLM response
    # The add_messages reducer will automatically append this to the conversation
    return {"messages": [response]}

