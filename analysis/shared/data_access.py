# analysis/shared/data_access.py
"""
Shared data-access utilities for statistical tools.

These functions abstract away the details of retrieving and validating
the working dataset from agent state or analysis context.

Included:
- get_working_df(state, cols)
- validate_groups(df, group_col, expect=2)
- coerce_numeric(df, cols)

They’re intentionally lightweight: these helpers don’t compute stats,
only ensure the data going into your tests is well-formed.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from ...graph.state import AgentState


# -----------------------------------------------------
# Retrieve working DataFrame
# -----------------------------------------------------

def get_working_df(state: AgentState, cols: List[str]) -> pd.DataFrame:
    """
    Retrieve the cleaned DataFrame that should be used for analysis.

    Priority:
      1. state["analysis_context"]["working_df"] (already cleaned by missing-data node)
      2. fallback: state["df"]

    Then subset to the specified columns if available.
    """
    ctx = state.get("analysis_context") or {}
    df = ctx.get("working_df") if isinstance(ctx.get("working_df"), pd.DataFrame) else state.get("df")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("No valid DataFrame found in state or analysis_context.")

    # Subset safely
    available = [c for c in cols if c in df.columns]
    if not available:
        raise ValueError(f"None of the required columns {cols} found in DataFrame.")
    return df[available].copy()


# -----------------------------------------------------
# Validate grouping structure
# -----------------------------------------------------

def validate_groups(df: pd.DataFrame, group_col: str, expect: int = 2) -> Dict[str, Any]:
    """
    Validate that group_col exists and has the expected number of unique groups.

    Returns:
      {
        "ok": bool,
        "levels": List[str],
        "sizes": Dict[str, int],
        "error": str|None
      }
    """
    out = {"ok": False, "levels": [], "sizes": {}, "error": None}
    if group_col not in df.columns:
        out["error"] = f"Grouping column '{group_col}' not found."
        return out

    levels = pd.Index(df[group_col].dropna().unique()).tolist()
    out["levels"] = levels
    sizes = df[group_col].value_counts(dropna=True).to_dict()
    out["sizes"] = {str(k): int(v) for k, v in sizes.items()}

    if len(levels) < expect:
        out["error"] = f"Expected at least {expect} groups; found {len(levels)}."
        return out

    if expect and len(levels) != expect:
        out["error"] = f"Expected {expect} groups; found {len(levels)} ({levels})."
        return out

    out["ok"] = True
    return out


# -----------------------------------------------------
# Coerce numeric columns safely
# -----------------------------------------------------

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Attempt to coerce the given columns to numeric dtype.
    Returns cleaned df + metadata describing which columns were coerced.

    Returns:
      (df_numeric, info)
      info = {"coerced": List[str], "non_numeric": List[str], "notes": List[str]}
    """
    info = {"coerced": [], "non_numeric": [], "notes": []}
    df_num = df.copy()

    for c in cols:
        if c not in df.columns:
            info["notes"].append(f"Column '{c}' not found.")
            continue
        before_dtype = str(df[c].dtype)
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().sum() == 0:
            info["non_numeric"].append(c)
            info["notes"].append(f"Column '{c}' has no numeric values after coercion.")
        else:
            info["coerced"].append(c)
        df_num[c] = coerced
        info["notes"].append(f"Coerced '{c}' from {before_dtype} to float.")

    return df_num, info
