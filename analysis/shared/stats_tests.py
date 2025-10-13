# analysis/shared/stats_tests.py
"""
Shared statistical test utilities:
- Normality checks (Shapiro–Wilk) by group and for pairs
- Variance homogeneity (Levene / Brown–Forsythe)
- Contingency helpers (expected counts) and χ² vs Fisher chooser
- Correlation chooser (Pearson vs Spearman) based on normality

All functions are defensive: they handle small n, constant arrays, NaNs, and
return structured dicts with values + booleans + notes suitable for narration.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------
# Helpers
# -----------------------------

def _is_constant(arr: np.ndarray) -> bool:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return False
    return np.nanmax(arr) == np.nanmin(arr)


def _nan_clean(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)




# -----------------------------
# Normality (Shapiro–Wilk)
# -----------------------------


def shapiro_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    min_n: int = 3,
    max_n: int = 5000,
) -> Dict[str, Any]:
    """
    Run Shapiro–Wilk per group (on value_col within each level of group_col).

    Returns:
        {
          "per_group": {level: {"n": int, "stat": float|None, "p": float|None, "ok": bool|None, "note": str|None}},
          "any_non_normal": bool,            # True if any group tested and failed (p < .05)
          "tested_groups": int,              # how many groups were actually tested
          "notes": List[str],
        }
    Rules:
      - n < min_n: not tested (ok=None), note recorded.
      - n > max_n: not tested (ok=None), note recorded (Shapiro not valid for very large n in SciPy).
      - constant data: not tested, note recorded.
    """
    out: Dict[str, Any] = {"per_group": {}, "any_non_normal": False, "tested_groups": 0, "notes": []}
    try:
        levels = pd.Index(df[group_col].dropna().unique()).tolist()
    except Exception:
        out["notes"].append("shapiro_by_group: failed to enumerate groups.")
        return out

    for lvl in levels:
        vals = _nan_clean(df.loc[df[group_col] == lvl, value_col])
        vals = vals[~np.isnan(vals)]
        info: Dict[str, Any] = {"n": int(vals.size), "stat": None, "p": None, "ok": None, "note": None}

        if info["n"] < min_n:
            info["note"] = f"Shapiro not run for '{lvl}' (n<{min_n})."
        elif info["n"] > max_n:
            info["note"] = f"Shapiro not run for '{lvl}' (n>{max_n})."
        elif _is_constant(vals):
            info["note"] = f"Shapiro not run for '{lvl}' (constant values)."
        else:
            try:
                stat, p = stats.shapiro(vals)
                info["stat"] = float(stat)
                info["p"] = float(p)
                info["ok"] = bool(p >= 0.05)
                out["tested_groups"] += 1
            except Exception as e:
                info["note"] = f"Shapiro failed for '{lvl}': {e.__class__.__name__}"

        out["per_group"][lvl] = info

    # any_non_normal only considers groups that were actually tested
    for lvl, info in out["per_group"].items():
        if info["ok"] is False:
            out["any_non_normal"] = True

    return out



def shapiro_pair(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    min_n: int = 3,
    max_n: int = 5000,
) -> Dict[str, Any]:
    """
    Shapiro–Wilk for two numeric variables (continuous × continuous correlation prep).

    Returns:
      {
        "x": {"n": int, "stat": float|None, "p": float|None, "ok": bool|None, "note": str|None},
        "y": {...},
        "both_normal": bool|None,   # None if neither was testable; else bool
        "notes": List[str],
      }
    """
    out: Dict[str, Any] = {"x": {}, "y": {}, "both_normal": None, "notes": []}

    def _one(series: pd.Series, label: str) -> Dict[str, Any]:
        vals = _nan_clean(series)
        vals = vals[~np.isnan(vals)]
        info = {"n": int(vals.size), "stat": None, "p": None, "ok": None, "note": None}
        if info["n"] < min_n:
            info["note"] = f"Shapiro not run for {label} (n<{min_n})."
        elif info["n"] > max_n:
            info["note"] = f"Shapiro not run for {label} (n>{max_n})."
        elif _is_constant(vals):
            info["note"] = f"Shapiro not run for {label} (constant values)."
        else:
            try:
                stat, p = stats.shapiro(vals)
                info["stat"] = float(stat)
                info["p"] = float(p)
                info["ok"] = bool(p >= 0.05)
            except Exception as e:
                info["note"] = f"Shapiro failed for {label}: {e.__class__.__name__}"
        return info

    out["x"] = _one(df[x_col], "x")
    out["y"] = _one(df[y_col], "y")

    oks = [k["ok"] for k in (out["x"], out["y"]) if k["ok"] is not None]
    out["both_normal"] = (len(oks) == 2 and all(oks)) if oks else None
    return out




# -----------------------------
# Variance homogeneity (Levene / Brown–Forsythe)
# -----------------------------

def levene_test(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    center: str = "median",  # "mean" (Levene), "median" (Brown–Forsythe), or "trimmed"
) -> Dict[str, Any]:
    """
    Levene/Brown–Forsythe homogeneity test across groups.
    Returns:
      {"stat": float|None, "p": float|None, "equal_var": bool|None, "k": int, "note": str|None}
    """
    out: Dict[str, Any] = {"stat": None, "p": None, "equal_var": None, "k": 0, "note": None}
    try:
        groups = []
        for lvl, sub in df[[group_col, value_col]].dropna().groupby(group_col):
            vals = _nan_clean(sub[value_col])
            vals = vals[~np.isnan(vals)]
            if vals.size >= 2 and not _is_constant(vals):
                groups.append(vals)
        out["k"] = len(groups)
        if len(groups) < 2:
            out["note"] = "Levene not run (need ≥2 non-constant groups)."
            return out

        stat, p = stats.levene(*groups, center=center)
        out["stat"] = float(stat); out["p"] = float(p)
        out["equal_var"] = bool(p >= 0.05)
        return out
    except Exception as e:
        out["note"] = f"Levene failed: {e.__class__.__name__}"
        return out




# -----------------------------
# Contingency / χ² vs Fisher
# -----------------------------


def contingency_expected_counts(
    df: pd.DataFrame,
    var1: str,
    var2: str,
) -> Dict[str, Any]:
    """
    Build contingency table and compute expected counts (for χ² assumptions).

    Returns:
      {
        "table": pd.DataFrame,
        "expected": np.ndarray|None,
        "any_expected_lt_5": bool|None,
        "zero_rows_cols": bool,
        "r": int, "c": int,
        "note": str|None
      }
    """
    out: Dict[str, Any] = {
        "table": None, "expected": None, "any_expected_lt_5": None,
        "zero_rows_cols": False, "r": 0, "c": 0, "note": None
    }
    try:
        tab = pd.crosstab(df[var1], df[var2])
        out["table"] = tab
        out["r"], out["c"] = tab.shape

        if tab.size == 0 or out["r"] == 0 or out["c"] == 0:
            out["note"] = "Empty contingency table."
            return out

        # zero rows/cols?
        out["zero_rows_cols"] = bool(((tab.sum(axis=1) == 0).any()) or ((tab.sum(axis=0) == 0).any()))

        try:
            chi2, p, dof, expected = stats.chi2_contingency(tab.values, correction=True)
            out["expected"] = expected
            out["any_expected_lt_5"] = bool((expected < 5).any())
        except Exception as e:
            out["note"] = f"chi2_contingency failed: {e.__class__.__name__}"
        return out

    except Exception as e:
        out["note"] = f"Contingency build failed: {e.__class__.__name__}"
        return out



def chi2_or_fisher(
    df: pd.DataFrame,
    var1: str,
    var2: str
) -> Dict[str, Any]:
    """
    Decide between Fisher's Exact and Chi-Squared based on table size/expected counts.

    Returns:
      {
        "chosen": "fisher"|"chi2",
        "stat": float|None,
        "p": float|None,
        "dof": int|None,           # only for chi2
        "table": pd.DataFrame,
        "expected": np.ndarray|None, # for chi2
        "note": str|None
      }
    """
    out: Dict[str, Any] = {"chosen": None, "stat": None, "p": None, "dof": None, "table": None, "expected": None, "note": None}

    info = contingency_expected_counts(df, var1, var2)
    out["table"] = info["table"]; out["expected"] = info["expected"]

    if info["table"] is None:
        out["note"] = info["note"] or "No table."
        return out

    r, c = info["r"], info["c"]

    # Fisher only defined for 2x2
    fisher_possible = (r == 2 and c == 2)

    if fisher_possible:
        try:
            # scipy returns (oddsratio, p)
            _, p = stats.fisher_exact(info["table"].values)
            out.update({"chosen": "fisher", "stat": None, "p": float(p), "dof": None})
            return out
        except Exception as e:
            out["note"] = f"Fisher failed: {e.__class__.__name__} → falling back to chi2."

    # Otherwise use chi-square (with Yates correction by default in expected_counts)
    try:
        chi2, p, dof, expected = stats.chi2_contingency(info["table"].values, correction=True)
        out.update({"chosen": "chi2", "stat": float(chi2), "p": float(p), "dof": int(dof), "expected": expected})
        # annotate low expected counts
        if expected is not None and (expected < 5).any() and not fisher_possible:
            out["note"] = "Some expected counts < 5; Fisher not available for non-2x2 tables."
        return out
    except Exception as e:
        out["note"] = f"Chi-square failed: {e.__class__.__name__}"
        return out




# -----------------------------
# Correlation chooser (Pearson vs Spearman)
# -----------------------------

def correlation_auto(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    min_pairs: int = 3
) -> Dict[str, Any]:
    """
    Choose Pearson (both normal) vs Spearman (otherwise), compute correlation.

    Returns:
      {
        "chosen": "pearson"|"spearman"|None,
        "r": float|None,         # Pearson r or Spearman rho
        "p": float|None,
        "n": int,
        "normality": dict,       # result of shapiro_pair
        "note": str|None
      }
    """
    out: Dict[str, Any] = {"chosen": None, "r": None, "p": None, "n": 0, "normality": {}, "note": None}

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = (~x.isna()) & (~y.isna())
    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)

    out["n"] = int(min(xv.size, yv.size))
    if out["n"] < min_pairs:
        out["note"] = f"Not enough paired observations (n<{min_pairs})."
        return out

    # normality on the full vectors; shapiro requires arrays, not series
    norm = shapiro_pair(pd.DataFrame({x_col: xv, y_col: yv}), x_col, y_col)
    out["normality"] = norm

    both_normal = norm.get("both_normal")
    if both_normal is True:
        # Pearson
        try:
            r, p = stats.pearsonr(xv, yv)
            out.update({"chosen": "pearson", "r": float(r), "p": float(p)})
        except Exception as e:
            out["note"] = f"Pearson failed: {e.__class__.__name__}; falling back to Spearman."
            try:
                r, p = stats.spearmanr(xv, yv, nan_policy="omit")
                out.update({"chosen": "spearman", "r": float(r), "p": float(p)})
            except Exception as e2:
                out["note"] = f"Spearman also failed: {e2.__class__.__name__}"
    else:
        # Spearman
        try:
            r, p = stats.spearmanr(xv, yv, nan_policy="omit")
            out.update({"chosen": "spearman", "r": float(r), "p": float(p)})
        except Exception as e:
            out["note"] = f"Spearman failed: {e.__class__.__name__}"

    return out
