"""
Metadata Extraction and Dataset Analysis Utilities

This module provides functions for analyzing dataset structure, extracting column
metadata, and preparing dataset context for statistical analysis workflows.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage
from pandas.api.types import (
    is_bool_dtype, is_categorical_dtype, is_object_dtype,
    is_integer_dtype, is_float_dtype, is_datetime64_any_dtype
)

def extract_metadata(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detect column types for statistical analysis.
    
    Categorizes columns as:
    - "categorical": object, category, bool types
    - "numerical": int, float types  
    - "other": datetime, complex, or other types
    
    Args:
        df: pandas DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to their detected types
    """
    metadata = {}
        
    for col in df.columns:
        s = df[col]
        if is_bool_dtype(s) or is_categorical_dtype(s) or is_object_dtype(s):
            metadata[col] = "categorical"
        elif is_integer_dtype(s) or is_float_dtype(s):
            metadata[col] = "numerical"
        elif is_datetime64_any_dtype(s):
            metadata[col] = "datetime"
        else:
            metadata[col] = "other"
    
    return metadata


def get_sample_data(df: pd.DataFrame, n_samples: int = 3) -> Dict[str, List]:
    """
    Extract sample values from each column for LLM context.
    
    Args:
        df: pandas DataFrame to sample from
        n_samples: Number of sample values to extract per column (default: 3)
        
    Returns:
        Dictionary mapping column names to lists of sample values
    """
    sample_data = {}
    
    for col in df.columns:
        # Get first n_samples non-null values
        non_null_values = df[col].dropna().head(n_samples)
        sample_data[col] = non_null_values.tolist()
    
    return sample_data


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive dataset information for analysis context.
    
    Args:
        df: pandas DataFrame to analyze
        
    Returns:
        Dictionary with dataset shape, missing data info, and column statistics
    """
    info = {
        "shape": df.shape,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "column_names": list(df.columns),
        "missing_data": {
            "total_missing": df.isnull().sum().sum(),
            "missing_by_column": df.isnull().sum().to_dict(),
            "complete_cases": len(df.dropna())
        }
    }
    
    return info


def create_dataset_summary_message(
    metadata: Dict[str, str],
    df: pd.DataFrame,
    n_rows: int = 5
) -> HumanMessage:
    """
    Create a comprehensive dataset summary message for the LLM.
    
    This creates the initial context message that informs the LLM about:
    - Dataset structure and column types
    - Sample data for understanding content
    - Basic statistics for analysis planning
    
    Args:
        metadata: Column type mapping from extract_metadata()
        df: The dataset being analyzed
        n_rows: Number of sample rows to include (default: 5)
        
    Returns:
        HumanMessage containing formatted dataset summary
    """
    # Build schema information
    schema_lines = [f"- {col}: {col_type}" for col, col_type in metadata.items()]
    schema_text = "\n".join(schema_lines)

    # Collect categorical and numerical columns
    categorical_cols = [col for col, col_type in metadata.items() if col_type == "categorical"]
    numerical_cols = [col for col, col_type in metadata.items() if col_type == "numerical"]
    other_cols = [col for col, col_type in metadata.items() if col_type not in ["categorical", "numerical"]]

    # Get sample data preview
    #preview = df.head(n_rows).to_string(index=False, max_cols=10, max_colwidth=20) max_colwidth n é uma opcao de display e nao uma funct arg
    with pd.option_context("display.max_colwidth", 20, "display.max_columns", 10):
        preview = df.head(n_rows).to_string(index=False, max_cols=10)


    # Get basic dataset info
    dataset_info = get_dataset_info(df)
    
    # Build comprehensive summary
    summary_text = f"""Dataset Summary

    Schema (all variables):
    {schema_text}

    Sample data (first {n_rows} rows):
    {preview}

    Dataset Information:
    - Total rows: {dataset_info['n_rows']:,}
    - Total columns: {dataset_info['n_columns']}
    - Missing values: {dataset_info['missing_data']['total_missing']:,}
    - Complete cases: {dataset_info['missing_data']['complete_cases']:,}

    Column Categories:
    - Categorical columns ({len(categorical_cols)}): {categorical_cols}
    - Numerical columns ({len(numerical_cols)}): {numerical_cols}
    """

    if other_cols:
        summary_text += f"\n- Other columns ({len(other_cols)}): {other_cols}"


    return HumanMessage(content=summary_text)


