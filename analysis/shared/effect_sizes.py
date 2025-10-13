# analysis/shared/effect_sizes.py
"""
Reusable effect size utilities.

Included:
- hedges_g(x, y): Cohen's d and Hedges' g for two independent groups
- rank_biserial_from_u(u_stat, n1, n2): rank-biserial correlation from Mann–Whitney U
- cramers_v_from_table(table): Cramér's V from a contingency table

Notes
-----
- Arrays are cleaned to float and NaNs are dropped before computation.
- When pooled variance is zero (constant samples), returns value=None with a note.
- Rank-biserial sign depends on which sample is passed as "x" when U is computed;
  see docstring for interpretation.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------
# Helpers
# -----------------------------

def _to_clean_array(x) -> np.ndarray:
    """Convert input to float ndarray and drop NaNs."""
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = np.asarray(x).ravel()
    arr = pd.to_numeric(np.asarray(x).ravel(), errors="coerce").astype(float)
    return arr[~np.isnan(arr)]


def _pooled_sd(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Unbiased pooled standard deviation (sqrt of pooled sample variance, ddof=1).
    Returns None if either group has n<2.
    """
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        return None
    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)
    num = (n1 - 1) * v1 + (n2 - 1) * v2
    den = (n1 + n2 - 2)
    if den <= 0:
        return None
    psd = np.sqrt(num / den)
    return float(psd)


# -----------------------------
# Hedges' g (and Cohen's d)
# -----------------------------

def hedges_g(x, y) -> Dict[str, Any]:
    """
    Compute Cohen's d and Hedges' g for two independent groups.

    Returns
    -------
    {
      "cohen_d": float|None,
      "hedges_g": float|None,
      "n1": int, "n2": int,
      "pooled_sd": float|None,
      "note": str|None
    }

    Notes
    -----
    - d = (mean_x - mean_y) / s_pooled
    - g = J * d, where J = 1 - 3/(4*(n1+n2) - 9)  (Hedges & Olkin small-sample correction)
    - If pooled_sd==0 (constant data) or insufficient n, returns None with a note.
    """
    x = _to_clean_array(x)
    y = _to_clean_array(y)
    n1, n2 = x.size, y.size

    out: Dict[str, Any] = {
        "cohen_d": None,
        "hedges_g": None,
        "n1": int(n1),
        "n2": int(n2),
        "pooled_sd": None,
        "note": None,
    }

    if n1 == 0 or n2 == 0:
        out["note"] = "Empty group(s); cannot compute effect size."
        return out
    psd = _pooled_sd(x, y)
    out["pooled_sd"] = psd

    if psd is None or psd == 0.0 or np.isnan(psd):
        out["note"] = "Pooled SD undefined or zero (constant/insufficient variability)."
        return out

    d = (np.mean(x) - np.mean(y)) / psd
    out["cohen_d"] = float(d)

    N = n1 + n2
    if N > 2:
        J = 1.0 - 3.0 / (4.0 * N - 9.0)
    else:
        J = 1.0  # degenerate; won't really happen with valid t-tests
    g = J * d
    out["hedges_g"] = float(g)
    return out


# -----------------------------
# Rank-biserial correlation (from U)
# -----------------------------

def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> Dict[str, Any]:
    """
    Compute rank-biserial correlation r_rb from Mann–Whitney U.

    Parameters
    ----------
    u_stat : float
        U statistic returned by scipy.stats.mannwhitneyu(x, y, ...).
        (In modern SciPy, this is the U for the first sample passed as `x`.)
    n1, n2 : int
        Sample sizes of the two groups (len(x), len(y)).

    Returns
    -------
    {
      "r": float|None,          # rank-biserial correlation
      "n1": int, "n2": int,
      "note": str|None
    }

    Interpretation
    --------------
    r = 1 - (2U)/(n1*n2)
    - If r > 0, ranks in the FIRST sample (the one used to compute U) tend to be higher.
    - If r < 0, ranks in the SECOND sample tend to be higher.
    - Magnitude ~ effect size (|r| ~ 0.1 small, 0.3 medium, 0.5 large as a rough guide).

    Notes
    -----
    - When older SciPy versions return min(Ux, Uy), the sign becomes ambiguous.
      In that case, treat |r| as the magnitude and infer direction from medians.
    """
    out: Dict[str, Any] = {"r": None, "n1": int(n1), "n2": int(n2), "note": None}
    if n1 <= 0 or n2 <= 0:
        out["note"] = "Invalid sample sizes for rank-biserial."
        return out
    try:
        r = 1.0 - (2.0 * float(u_stat)) / (float(n1) * float(n2))
        out["r"] = float(r)
        return out
    except Exception as e:
        out["note"] = f"Failed to compute rank-biserial: {e.__class__.__name__}"
        return out


# -----------------------------
# Cramér's V (contingency tables)
# -----------------------------

def cramers_v_from_table(table: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute Cramér's V from an observed contingency table.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table of observed counts (rows: categories of var1, cols: categories of var2).

    Returns
    -------
    {
      "v": float|None,
      "chi2": float|None,
      "dof": int|None,
      "n": int,
      "note": str|None
    }

    Notes
    -----
    - Uses chi2_contingency(table, correction=False) to get chi2 without Yates' correction.
    - V = sqrt(chi2 / (n * (min(r-1, c-1))))
    - If min(r-1, c-1) == 0 (i.e., one-dimensional), V is undefined.
    """
    out: Dict[str, Any] = {"v": None, "chi2": None, "dof": None, "n": int(np.sum(table.values)), "note": None}
    try:
        if table.size == 0:
            out["note"] = "Empty table."
            return out

        r, c = table.shape
        if r < 2 or c < 2:
            out["note"] = "Cramér's V undefined for 1D tables."
            return out

        chi2, p, dof, expected = stats.chi2_contingency(table.values, correction=False)
        denom = out["n"] * float(min(r - 1, c - 1))
        if denom <= 0:
            out["note"] = "Invalid denominator for Cramér's V."
            return out

        v = np.sqrt(chi2 / denom)
        out.update({"v": float(v), "chi2": float(chi2), "dof": int(dof)})
        return out
    except Exception as e:
        out["note"] = f"Cramér's V failed: {e.__class__.__name__}"
        return out
