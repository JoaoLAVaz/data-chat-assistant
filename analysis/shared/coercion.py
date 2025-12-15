# analysis/shared/coercion.py
"""
Helpers for gentle, transparent type coercions in analysis pipelines.

Right now this focuses on:
- Interpreting low-cardinality integer-coded numeric columns as categorical
  when the user explicitly uses them in a categorical role (group_col, chi-square vars, etc).
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def coerce_numeric_to_categorical_if_safe(
    df: pd.DataFrame,
    metadata: Dict[str, str],
    *,
    col: str,
    role: str,
    max_unique: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Decide whether it's safe/reasonable to treat a numeric column as categorical
    for a *categorical* role in an analysis (group_col, chi-square var, etc).

    Heuristic:
      - metadata[col] is not already "categorical"
      - column exists in df
      - non-null values >= 2
      - number of distinct non-null values <= max_unique
      - all distinct values are (approximately) integers

    If conditions are met, returns a small report dict describing the coercion.
    If not, returns None and the caller should treat the column as numeric and
    likely raise/return an error for that role.

    This helper does NOT mutate df or metadata. It is purely advisory.
    """
    kind = metadata.get(col, "other")

    # If it already says categorical, nothing to do.
    if kind == "categorical":
        return None

    if col not in df.columns:
        return None

    s = pd.to_numeric(df[col], errors="coerce")
    non_na = s.dropna()

    # Need at least 2 non-missing values
    if non_na.size < 2:
        return None

    # Distinct value count must be small
    uniq = np.unique(non_na.values)
    n_unique = uniq.size
    if n_unique == 0 or n_unique > max_unique:
        return None

    # All values should be "integer-like" (0, 1, 2, 3, ...)
    rounded = np.rint(uniq)
    if not np.allclose(uniq, rounded):
        return None

    # Safe to treat as categorical for this analysis
    # Build a small transparency report for the result JSON.
    preview_vals = [float(v) for v in uniq[:10]]
    return {
        "original_kind": kind,
        "role": role,
        "n_unique": int(n_unique),
        "values_preview": preview_vals,
        "note": (
            "Column stored as numbers but used in a categorical role; "
            "treated as integer-coded categories for this analysis only."
        ),
    }
