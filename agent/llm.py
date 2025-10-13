"""
LLM Configuration and Factory Functions

This module handles the initialization and configuration of language models
for the statistical analysis agent. It provides factory functions to create
tool-enabled LLMs with consistent configuration.
"""

import os
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_llm_with_tools(
    tools: Optional[List[BaseTool]] = None,
    model: str = "gpt-4o", 
    temperature: float = 0.1,
    max_tokens: Optional[int] = None
) -> Runnable:
    """
    Create a tool-enabled language model instance.
    
    Args:
        tools: List of tools to bind to the LLM (default: None, imports from tools.py)
        model: Model name to use (default: gpt-4o)
        temperature: Sampling temperature (default: 0.1 for consistent analysis)
        max_tokens: Maximum tokens in response (default: None for model default)
        
    Returns:
        Tool-enabled language model ready for statistical analysis
    """
    # Create LLM directly
    llm = ChatOpenAI(
        model=model,
        temperature=temperature, 
        max_tokens=max_tokens
    )
    
    # Import tools if not provided
    if tools is None:
        from .tools import get_all_tools
        tools = get_all_tools()
    
    return llm.bind_tools(tools)