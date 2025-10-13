# analysis/ttest/exec.py
"""
T-Test Statistical Analysis Implementation (assumption-driven)

Assumes missing-data handling has already been applied upstream (missing_data_node).
This implementation:
  - Validates inputs and optional group filters
  - Runs Shapiro–Wilk by group; if any non-normal → Mann–Whitney U
  - Else runs Levene/Brown–Forsythe; equal var → Student’s t; else → Welch’s t
  - Computes effect sizes (Hedges’ g for t-tests; rank-biserial for MWU)
  - Creates a violin plot and returns its temp file path
  - Returns a structured, JSON-serializable payload
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from scipy import stats
import os
import tempfile
import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

from analysis.shared.stats_tests import shapiro_by_group, levene_test
from analysis.shared.effect_sizes import hedges_g, rank_biserial_from_u


# -----------------------------
# Plot helper
# -----------------------------
def _violin_plot_to_tempfile(g1, g2, group_names: List[str], ylabel: str, title: str) -> str:
    """
    Make a violin plot with jittered points and save to a temp .png.
    Returns the file path.
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    # Violin
    parts = ax.violinplot([g1, g2], positions=[1, 2], showmeans=False, showmedians=True, widths=0.8)
    # Neutral aesthetics
    for pc in parts["bodies"]:
        pc.set_alpha(0.4)

    # Jittered scatter of points
    rng = np.random.default_rng(0)
    x1 = 1 + rng.normal(0, 0.04, size=g1.size)
    x2 = 2 + rng.normal(0, 0.04, size=g2.size)
    ax.plot(x1, g1, "o", ms=3, alpha=0.5)
    ax.plot(x2, g2, "o", ms=3, alpha=0.5)

    # Means as dashed lines
    if g1.size > 0:
        ax.hlines(np.mean(g1), 0.8, 1.2, linestyles="--")
    if g2.size > 0:
        ax.hlines(np.mean(g2), 1.8, 2.2, linestyles="--")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(group_names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

    fd, path = tempfile.mkstemp(suffix=".png", prefix="ttest_", text=False)
    os.close(fd)  # matplotlib will write by file path
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------
# Robust MWU helper
# -----------------------------
def _robust_mwu(g1, g2, warnings: List[str]):
    """
    Try Mann–Whitney U with method='auto', then fallback to 'asymptotic'.
    Returns (u, p, method_used). Raises RuntimeError if both fail.
    """
    try:
        u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided", method="auto")
        return float(u), float(p), "auto"
    except Exception as e1:
        try:
            u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided", method="asymptotic")
            warnings.append(
                f"Mann–Whitney switched to 'asymptotic' after 'auto' failed: {e1.__class__.__name__}: {e1}"
            )
            return float(u), float(p), "asymptotic"
        except Exception as e2:
            raise RuntimeError(
                f"MWU auto failed ({e1.__class__.__name__}: {e1}); "
                f"asymptotic failed ({e2.__class__.__name__}: {e2})"
            )


# -----------------------------
# Main implementation
# -----------------------------
def t_test_impl(
    df: pd.DataFrame,
    metadata: Dict[str, str],
    *,
    group_col: str,
    value_col: str,
    equal_var: bool = False,  # user override: force Student's t (still report Levene as "ignored")
    group_a: Optional[str] = None,  # optional filter to exactly two group levels
    group_b: Optional[str] = None,
    nonparametric: bool = False,  # optional override to force Mann–Whitney
) -> Dict[str, Any]:
    """
    Two-sample comparison with assumption checks and branching.
    """

    # --- 0) Validate metadata/types ---
    if group_col not in metadata:
        return {"error": f"Column '{group_col}' not found in dataset"}
    if value_col not in metadata:
        return {"error": f"Column '{value_col}' not found in dataset"}
    if metadata[group_col] != "categorical":
        return {"error": f"'{group_col}' should be categorical, but is {metadata[group_col]}"}
    if metadata[value_col] != "numerical":
        return {"error": f"'{value_col}' should be numerical, but is {metadata[value_col]}"}

    # --- 1) Pull required columns (no NA handling here; upstream should have cleaned) ---
    data = df[[group_col, value_col]].copy()

    # Enforce no-missing invariant
    na_counts = data.isna().sum().to_dict()
    total_na = int(sum(na_counts.values()))
    if total_na > 0:
        return {
            "error": "Missing values detected in required columns after preprocessing step.",
            "details": {
                "columns_checked": [group_col, value_col],
                "missing_by_column": {k: int(v) for k, v in na_counts.items()},
                "hint": "Ensure missing-data preprocessing produced a 'working_df' and is being used.",
            },
        }

    # --- 2) Available levels and optional filtering ---
    levels = pd.Index(data[group_col].unique()).tolist()

    if (group_a is not None) or (group_b is not None):
        if not (group_a and group_b):
            return {"error": "If specifying filters, both 'group_a' and 'group_b' must be provided."}
        missing_levels = [g for g in (group_a, group_b) if g not in levels]
        if missing_levels:
            return {
                "error": "Requested groups not found in the data.",
                "requested": [group_a, group_b],
                "available_levels": levels,
            }
        data = data[data[group_col].isin([group_a, group_b])]

    # Recompute levels after filtering
    levels = pd.Index(data[group_col].unique()).tolist()

    if len(levels) < 2:
        return {"error": "Fewer than two groups available for comparison after filtering.", "available_levels": levels}
    if len(levels) > 2:
        return {
            "error": f"More than two groups present in '{group_col}'. Please specify exactly two groups.",
            "available_levels": levels,
            "hint": "Call t_test with group_a='LEVEL1', group_b='LEVEL2'.",
        }

    # --- 3) Extract groups (deterministic order) ---
    g1_name, g2_name = sorted(levels)
    g1 = data.loc[data[group_col] == g1_name, value_col].astype(float).to_numpy()
    g2 = data.loc[data[group_col] == g2_name, value_col].astype(float).to_numpy()

    if g1.size < 2 or g2.size < 2:
        return {"error": "Each group needs at least 2 observations for the test.",
                "sizes": {g1_name: int(g1.size), g2_name: int(g2.size)}}

    # Group summaries
    def _summary(arr: np.ndarray) -> Dict[str, float]:
        q1, q3 = np.percentile(arr, [25, 75]) if arr.size > 0 else (np.nan, np.nan)
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan"),
            "median": float(np.median(arr)),
            "iqr": float(q3 - q1) if np.isfinite(q1) and np.isfinite(q3) else float("nan"),
        }

    groups_block = {
        "group1": {"name": g1_name, **_summary(g1)},
        "group2": {"name": g2_name, **_summary(g2)},
    }

    warnings: List[str] = []

    # --- 4) If user forces nonparametric, go straight to MWU ---
    if nonparametric:
        try:
            u, p, method_used = _robust_mwu(g1, g2, warnings)
            rrb = rank_biserial_from_u(u, g1.size, g2.size)
            title = f"Mann–Whitney U: {g1_name} vs {g2_name}"
            plot_path = _violin_plot_to_tempfile(g1, g2, [g1_name, g2_name], ylabel=value_col, title=title)
            return {
                "chosen_test": "mann_whitney",
                "test_name": "Mann–Whitney U",
                "stats": {"U": u, "p_value": p, "method": method_used},
                "effect_size": {"name": "rank_biserial", "value": rrb["r"], "note": rrb.get("note")},
                "groups": groups_block,
                "assumptions": {
                    "normality": {"forced_nonparametric": True},
                    "variance": {"note": "Not applicable for Mann–Whitney."},
                },
                "warnings": warnings,
                "plot_path": plot_path,
            }
        except Exception as e:
            return {
                "error": "Mann–Whitney U failed",
                "details": str(e),
                "assumptions": {"normality": {"forced_nonparametric": True}},
            }

    # --- 5) Normality: Shapiro–Wilk per group ---
    normality = shapiro_by_group(
        pd.DataFrame({group_col: [g1_name] * g1.size + [g2_name] * g2.size, value_col: np.concatenate([g1, g2])}),
        group_col,
        value_col,
        min_n=3,
    )

    # Decide parametric vs nonparametric
    go_nonparametric = bool(normality.get("any_non_normal", False))
    tested_groups = int(normality.get("tested_groups", 0))

    # If neither group could be tested (e.g., n<3 or constant), prefer safer nonparametric
    if tested_groups == 0:
        go_nonparametric = True
        warnings.append("Normality not testable (n<3 or constant); defaulting to nonparametric test.")

    if go_nonparametric:
        try:
            u, p, method_used = _robust_mwu(g1, g2, warnings)
            rrb = rank_biserial_from_u(u, g1.size, g2.size)
            title = f"Mann–Whitney U: {g1_name} vs {g2_name}"
            plot_path = _violin_plot_to_tempfile(g1, g2, [g1_name, g2_name], ylabel=value_col, title=title)
            return {
                "chosen_test": "mann_whitney",
                "test_name": "Mann–Whitney U",
                "stats": {"U": u, "p_value": p, "method": method_used},
                "effect_size": {"name": "rank_biserial", "value": rrb["r"], "note": rrb.get("note")},
                "groups": groups_block,
                "assumptions": {
                    "normality": normality,
                    "variance": {"note": "Not applicable for Mann–Whitney."},
                },
                "warnings": warnings,
                "plot_path": plot_path,
            }
        except Exception as e:
            return {
                "error": "Mann–Whitney U failed",
                "details": str(e),
                "assumptions": {"normality": normality},
            }

    # --- 6) Variance homogeneity: Levene/Brown–Forsythe ---
    lev = levene_test(
        pd.DataFrame({group_col: [g1_name] * g1.size + [g2_name] * g2.size, value_col: np.concatenate([g1, g2])}),
        group_col,
        value_col,
        center="median",
    )

    # Decide Student vs Welch
    if equal_var:
        # User forced Student's t, but still report Levene as ignored
        if lev.get("p") is not None:
            warnings.append("equal_var=True override: Levene/Brown–Forsythe result was ignored for test choice.")
        use_equal_var = True
    else:
        use_equal_var = bool(lev.get("equal_var", False))

    # --- 7) Run t-test ---
    try:
        t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=use_equal_var)
    except Exception as e:
        return {"error": f"t-test failed: {e.__class__.__name__}", "assumptions": {"normality": normality, "variance": lev}}

    # Effect size: Hedges' g
    g_eff = hedges_g(g1, g2)

    # Plot
    label = "Student's t-test" if use_equal_var else "Welch's t-test"
    title = f"{label}: {g1_name} vs {g2_name}"
    plot_path = _violin_plot_to_tempfile(g1, g2, [g1_name, g2_name], ylabel=value_col, title=title)

    chosen = "student_t" if use_equal_var else "welch_t"
    result = {
        "chosen_test": chosen,
        "test_name": label,
        "stats": {"t": float(t_stat), "p_value": float(p_value)},
        "effect_size": {"name": "hedges_g", "value": g_eff.get("hedges_g"), "note": g_eff.get("note")},
        "groups": groups_block,
        "assumptions": {"normality": normality, "variance": lev},
        "warnings": warnings,
        "plot_path": plot_path,
    }
    return result
