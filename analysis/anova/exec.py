# analysis/anova/exec.py
"""
One-way ANOVA pipeline with assumption checks and fallbacks.

Flow:
- Assumes missing-data handling upstream (missing_data_node).
- Shapiro–Wilk per group → if any non-normal => Kruskal–Wallis (nonparametric).
- Else Levene (Brown–Forsythe, center='median') → equal var ? One-way ANOVA : Welch's ANOVA.
- Effect sizes included.
- Returns a violin plot path for visualization.

Outputs a dict suitable for LLM interpretation or a later narrative node.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import os
import tempfile
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from analysis.shared.stats_tests import shapiro_by_group, levene_test


# -----------------------------
# Plot helpers
# -----------------------------
def _violin_multi_plot_to_tempfile(group_arrays: List[np.ndarray],
                                   group_names: List[str],
                                   ylabel: str,
                                   title: str) -> str:
    """Violin plot (+ jitter) for k groups; returns temp PNG path."""
    k = len(group_arrays)
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * k), 4), dpi=150)

    parts = ax.violinplot(group_arrays, positions=list(range(1, k + 1)),
                          showmeans=False, showmedians=True, widths=0.8)
    for pc in parts.get("bodies", []):
        pc.set_alpha(0.4)

    rng = np.random.default_rng(0)
    for i, arr in enumerate(group_arrays, start=1):
        if arr.size == 0:
            continue
        x = i + rng.normal(0, 0.04, size=arr.size)
        ax.plot(x, arr, "o", ms=3, alpha=0.5)
        ax.hlines(np.mean(arr), i - 0.2, i + 0.2, linestyles="--")

    ax.set_xticks(list(range(1, k + 1)))
    ax.set_xticklabels(group_names, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

    fd, path = tempfile.mkstemp(suffix=".png", prefix="anova_", text=False)
    os.close(fd)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------
# Effect size helpers (local)
# -----------------------------
def _anova_effect_sizes(groups: List[np.ndarray]) -> Dict[str, Optional[float]]:
    """Compute eta^2 and omega^2 for one-way ANOVA (parametric)."""
    # Flatten for totals
    all_values = np.concatenate(groups)
    grand_mean = np.mean(all_values)
    # Between-groups SS
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    # Within-groups SS
    ss_within = sum(sum((g - np.mean(g)) ** 2) for g in groups)
    ss_total = ss_between + ss_within

    k = len(groups)
    n = sum(len(g) for g in groups)

    eta2 = ss_between / ss_total if ss_total > 0 else None
    # Omega^2 = (SSB - (k-1)*MS_within) / (SST + MS_within)
    df_within = n - k
    if df_within > 0:
        ms_within = ss_within / df_within
        omega2 = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within) if (ss_total + ms_within) > 0 else None
    else:
        omega2 = None

    return {"eta_squared": float(eta2) if eta2 is not None else None,
            "omega_squared": float(omega2) if omega2 is not None else None}


def _kruskal_epsilon_squared(H: float, n: int, k: int) -> Optional[float]:
    """Epsilon-squared effect size for Kruskal–Wallis: (H - k + 1)/(n - k)."""
    denom = (n - k)
    if denom <= 0:
        return None
    return float((H - k + 1) / denom)


# -----------------------------
# Welch's ANOVA (Welch 1951)
# -----------------------------
def _welch_anova(groups: List[np.ndarray]) -> Tuple[float, float, float, float]:
    """
    Return F, p, df1, df2 for Welch's ANOVA (unequal variances).
    Implementation per standard formulas (uses Satterthwaite df2).
    """
    # Means, variances, sizes
    means = np.array([np.mean(g) for g in groups], dtype=float)
    vars_ = np.array([np.var(g, ddof=1) if len(g) > 1 else 0.0 for g in groups], dtype=float)
    ns = np.array([len(g) for g in groups], dtype=float)
    k = len(groups)

    # Handle zero variances (add tiny epsilon to avoid div by 0)
    eps = 1e-12
    vars_safe = np.where(vars_ <= 0, eps, vars_)
    w = ns / vars_safe  # weights

    # Weighted mean
    y_bar = np.sum(w * means) / np.sum(w)
    # Numerator
    num = np.sum(w * (means - y_bar) ** 2) / (k - 1)
    # Denominator
    denom = 1 + (2 * (k - 2) / (k**2 - 1)) * np.sum(((1 / (ns - 1)) * (1 - (w / np.sum(w)))**2))
    F = num / denom

    # df1, df2 (Satterthwaite)
    df1 = k - 1
    # Compute df2 using standard Welch ANOVA approximation
    # Following: df2 = (k^2 - 1) / (3 * sum( (1/(n_i - 1)) * (1 - w_i/sum_w)^2 ))
    denom_df = np.sum(((1 / (ns - 1)) * (1 - (w / np.sum(w)))**2))
    df2 = (k**2 - 1) / (3 * denom_df) if denom_df > 0 else np.inf

    # p-value from F distribution
    p = 1 - stats.f.cdf(F, df1, df2)
    return float(F), float(p), float(df1), float(df2)


# -----------------------------
# Main implementation
# -----------------------------
def anova_impl(
    df: pd.DataFrame,
    metadata: Dict[str, str],
    *,
    group_col: str,
    value_col: str,
    equal_var: bool = False,         # user override: force classic ANOVA (ignore Levene for choice)
    nonparametric: bool = False,     # user override: force Kruskal–Wallis
) -> Dict[str, Any]:
    """
    One-way comparison across 3+ groups with assumption checks and branching.
    """
    # --- Validate types ---
    if group_col not in metadata:
        return {"error": f"Column '{group_col}' not found in dataset"}
    if value_col not in metadata:
        return {"error": f"Column '{value_col}' not found in dataset"}
    if metadata[group_col] != "categorical":
        return {"error": f"'{group_col}' should be categorical, but is {metadata[group_col]}"}
    if metadata[value_col] != "numerical":
        return {"error": f"'{value_col}' should be numerical, but is {metadata[value_col]}"}

    # Required data (assume upstream missing-data handled)
    data = df[[group_col, value_col]].copy()
    na_counts = data.isna().sum().to_dict()
    if int(sum(na_counts.values())) > 0:
        return {
            "error": "Missing values detected in required columns after preprocessing.",
            "details": {"missing_by_column": {k: int(v) for k, v in na_counts.items()}},
        }

    # Group arrays
    levels = list(pd.Index(data[group_col].unique()))
    if len(levels) < 3:
        return {"error": f"ANOVA expects 3+ groups; found {len(levels)}: {levels}"}

    # Build arrays in deterministic order
    levels = sorted(levels)
    groups = [data.loc[data[group_col] == lvl, value_col].astype(float).to_numpy() for lvl in levels]

    # Check minimal sizes
    too_small = [lvl for lvl, arr in zip(levels, groups) if arr.size < 2]
    if too_small:
        return {"error": "Each group needs at least 2 observations.",
                "too_small_groups": too_small}

    # Group summaries block
    def _summary(arr: np.ndarray) -> Dict[str, float]:
        q1, q3 = np.percentile(arr, [25, 75]) if arr.size > 0 else (np.nan, np.nan)
        return {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan"),
            "median": float(np.median(arr)),
            "iqr": float(q3 - q1) if np.isfinite(q1) and np.isfinite(q3) else float("nan"),
        }

    groups_block = {lvl: _summary(arr) for lvl, arr in zip(levels, groups)}
    warnings: List[str] = []

    # Forced nonparametric?
    if nonparametric:
        try:
            H, p = stats.kruskal(*groups, nan_policy="omit")
            eps2 = _kruskal_epsilon_squared(H, n=sum(len(g) for g in groups), k=len(groups))
            title = f"Kruskal–Wallis: {group_col} on {value_col}"
            plot_path = _violin_multi_plot_to_tempfile(groups, levels, ylabel=value_col, title=title)
            return {
                "chosen_test": "kruskal_wallis",
                "test_name": "Kruskal–Wallis H",
                "stats": {"H": float(H), "p_value": float(p), "df": len(groups) - 1},
                "effect_size": {"name": "epsilon_squared", "value": eps2},
                "groups": groups_block,
                "assumptions": {"normality": {"forced_nonparametric": True},
                                "variance": {"note": "Not applicable for Kruskal–Wallis."}},
                "warnings": warnings,
                "plot_path": plot_path,
            }
        except Exception as e:
            return {"error": f"Kruskal–Wallis failed: {e.__class__.__name__}", "details": str(e)}

    # Normality per group
    stacked = pd.DataFrame({group_col: np.repeat(levels, [len(g) for g in groups]),
                            value_col: np.concatenate(groups)})
    normality = shapiro_by_group(stacked, group_col, value_col, min_n=3)
    go_nonparametric = bool(normality.get("any_non_normal", False))
    tested_groups = int(normality.get("tested_groups", 0))
    if tested_groups == 0:
        go_nonparametric = True
        warnings.append("Normality not testable for one or more groups; defaulting to nonparametric test.")

    if go_nonparametric:
        try:
            H, p = stats.kruskal(*groups, nan_policy="omit")
            eps2 = _kruskal_epsilon_squared(H, n=sum(len(g) for g in groups), k=len(groups))
            title = f"Kruskal–Wallis: {group_col} on {value_col}"
            plot_path = _violin_multi_plot_to_tempfile(groups, levels, ylabel=value_col, title=title)
            return {
                "chosen_test": "kruskal_wallis",
                "test_name": "Kruskal–Wallis H",
                "stats": {"H": float(H), "p_value": float(p), "df": len(groups) - 1},
                "effect_size": {"name": "epsilon_squared", "value": eps2},
                "groups": groups_block,
                "assumptions": {"normality": normality,
                                "variance": {"note": "Not applicable for Kruskal–Wallis."}},
                "warnings": warnings,
                "plot_path": plot_path,
            }
        except Exception as e:
            return {"error": f"Kruskal–Wallis failed: {e.__class__.__name__}",
                    "details": str(e), "assumptions": {"normality": normality}}

    # Variance homogeneity: Levene/Brown–Forsythe
    # Build tall DF for shared test
    lev = levene_test(stacked, group_col, value_col, center="median")

    # Decide classic vs Welch ANOVA
    if equal_var:
        if lev.get("p") is not None:
            warnings.append("equal_var=True override: Levene/Brown–Forsythe result was ignored for test choice.")
        use_equal_var = True
    else:
        use_equal_var = bool(lev.get("equal_var", False))

    if use_equal_var:
        # Classic ANOVA
        try:
            F, p = stats.f_oneway(*groups)
        except Exception as e:
            return {"error": f"ANOVA failed: {e.__class__.__name__}",
                    "assumptions": {"normality": normality, "variance": lev}}

        # Effect sizes
        eff = _anova_effect_sizes(groups)
        title = f"One-way ANOVA: {group_col} on {value_col}"
        plot_path = _violin_multi_plot_to_tempfile(groups, levels, ylabel=value_col, title=title)

        return {
            "chosen_test": "anova",
            "test_name": "One-way ANOVA",
            "stats": {"F": float(F), "p_value": float(p), "df1": len(groups) - 1, "df2": int(sum(len(g) for g in groups) - len(groups))},
            "effect_size": {"name": "eta_squared/omega_squared",
                            "eta_squared": eff["eta_squared"],
                            "omega_squared": eff["omega_squared"]},
            "groups": groups_block,
            "assumptions": {"normality": normality, "variance": lev},
            "warnings": warnings,
            "plot_path": plot_path,
        }
    else:
        # Welch's ANOVA
        try:
            F, p, df1, df2 = _welch_anova(groups)
        except Exception as e:
            return {"error": f"Welch's ANOVA failed: {e.__class__.__name__}",
                    "assumptions": {"normality": normality, "variance": lev}}

        # Approximate eta^2 using between/total from group means (not exact under Welch)
        eff = _anova_effect_sizes(groups)
        title = f"Welch's ANOVA: {group_col} on {value_col}"
        plot_path = _violin_multi_plot_to_tempfile(groups, levels, ylabel=value_col, title=title)

        return {
            "chosen_test": "welch_anova",
            "test_name": "Welch's ANOVA",
            "stats": {"F": float(F), "p_value": float(p), "df1": float(df1), "df2": float(df2)},
            "effect_size": {"name": "eta_squared", "eta_squared": eff["eta_squared"], "note": "Approximate under Welch."},
            "groups": groups_block,
            "assumptions": {"normality": normality, "variance": lev},
            "warnings": warnings,
            "plot_path": plot_path,
        }
