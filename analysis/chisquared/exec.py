# analysis/chi_square/exec.py
"""
Chi-square / Fisher exact test for categorical association with plotting.

Flow:
- Assumes missing-data handling upstream; returns error if NA remains.
- Build contingency table for var1 x var2 (both categorical).
- Compute expected counts via chi2_contingency.
- If table is 2x2 and any expected < 5 -> Fisher's Exact Test.
- Else -> Chi-square test (Yates correction automatically for 2x2).
- Effect size: Cramér's V (and odds ratio for Fisher).
- Plot: Clustered bar chart (var1 on x-axis, bars for each level of var2).
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats
import os
import tempfile
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from analysis.shared.schema import SCHEMA_VERSION  


# -----------------------------
# Helpers
# -----------------------------
def _cramers_v(chi2: float, table: np.ndarray) -> Optional[float]:
    """Cramér's V = sqrt( chi2 / (n * (min(r-1, c-1))) )."""
    if table.size == 0:
        return None
    n = table.sum()
    if n <= 0:
        return None
    r, c = table.shape
    denom_df = min(r - 1, c - 1)
    if denom_df <= 0:
        return None
    v = np.sqrt(chi2 / (n * denom_df))
    return float(v)


def _clustered_bar_to_tempfile(ct: pd.DataFrame, var1: str, var2: str, title: str) -> str:
    """
    Clustered bar chart:
      - x-axis: levels of var1 (index)
      - bars: levels of var2 (columns)
    Returns temp PNG path.
    """
    # Ensure deterministic order
    ct = ct.copy()
    ct = ct.sort_index(axis=0).sort_index(axis=1)

    k = ct.shape[1]
    fig_w = max(6, 1.5 * max(k, ct.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5), dpi=150)

    x = np.arange(ct.shape[0])  # positions for var1 levels
    width = 0.8 / max(1, k)     # total width ~0.8 split among columns

    for i, col in enumerate(ct.columns):
        ax.bar(x + i * width - 0.4 + width * k / 2, ct[col].values, width=width, label=str(col), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(ix) for ix in ct.index], rotation=0)
    ax.set_xlabel(var1)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(title=var2, fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    fd, path = tempfile.mkstemp(suffix=".png", prefix="chisq_", text=False)
    os.close(fd)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------
# Main implementation
# -----------------------------
def chi_square_impl(
    df: pd.DataFrame,
    metadata: Dict[str, str],
    *,
    var1: str,
    var2: str,
    alpha: float = 0.05,
    missing_report: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Association test between two categorical variables.
    Chooses Fisher exact (2x2, expected < 5) or Chi-square accordingly.
    """
    # Validate inputs
    for v in (var1, var2):
        if v not in metadata:
            return {"schema_version": SCHEMA_VERSION, "test_family": "chi_square",
                    "error": f"Column '{v}' not found in dataset"}
        if metadata[v] != "categorical":
            return {"schema_version": SCHEMA_VERSION, "test_family": "chi_square",
                    "error": f"'{v}' must be categorical, but is {metadata[v]}"}

    # Required data (assume upstream missing-data handled)
    data = df[[var1, var2]].copy()
    na_counts = data.isna().sum().to_dict()
    if int(sum(na_counts.values())) > 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "test_family": "chi_square",
            "error": "Missing values detected in required columns after preprocessing.",
            "details": {"missing_by_column": {k: int(v) for k, v in na_counts.items()}},
        }

    # Contingency table (rows=var1 levels, cols=var2 levels)
    ct = pd.crosstab(data[var1], data[var2])
    if ct.empty or ct.values.sum() == 0:
        return {"schema_version": SCHEMA_VERSION, "test_family": "chi_square",
                "error": "Unable to form a valid contingency table (no counts)."}

    # Expected counts via chi2_contingency (no correction here; we just want expected)
    try:
        chi2_tmp, p_tmp, dof_tmp, expected = stats.chi2_contingency(ct.values, correction=False)
        expected = np.asarray(expected, dtype=float)
    except Exception as e:
        return {"schema_version": SCHEMA_VERSION, "test_family": "chi_square",
                "error": f"Failed to compute expected counts: {e.__class__.__name__}", "details": str(e)}

    min_expected = float(expected.min()) if expected.size else None
    r, c = ct.shape
    is_2x2 = (r == 2 and c == 2)

    warnings: List[str] = []

    # Decide test
    use_fisher = False
    if is_2x2 and (min_expected is not None) and (min_expected < 5):
        use_fisher = True

    title = f"{'Fisher exact' if use_fisher else 'Chi-square'}: {var1} × {var2}"
    plot_path = _clustered_bar_to_tempfile(ct, var1, var2, title=title)

    # Run chosen test
    if use_fisher:
        try:
            # SciPy fisher_exact supports only 2x2
            oddsratio, p = stats.fisher_exact(ct.values, alternative="two-sided")
            # For effect size, compute chi2 from observed/expected to get V
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2_stat = ((ct.values - expected) ** 2 / expected).sum()
                chi2_stat = float(chi2_stat) if np.isfinite(chi2_stat) else None
            cramer_v = _cramers_v(chi2_stat if chi2_stat is not None else 0.0, ct.values) if chi2_stat is not None else None

            result = {
                "schema_version": SCHEMA_VERSION,
                "test_family": "chi_square",
                "chosen_test": "fisher_exact",
                "test_name": "Fisher's Exact Test",
                "stats": {
                    "p_value": float(p),
                    "odds_ratio": float(oddsratio) if np.isfinite(oddsratio) else None,
                    "table_shape": [r, c],
                    "min_expected": min_expected,
                },
                "effect_size": {
                    "name": "Cramér's V",
                    "value": cramer_v,
                    "note": "Computed from observed vs expected; Fisher itself has no V.",
                },
                "contingency_table": ct.to_dict(),
                "assumptions": {
                    "expected_frequencies": {"min": min_expected, "rule_of_5_triggered": True, "alpha": alpha}
                },
                "warnings": warnings,
                "plot_path": plot_path,
            }
            if missing_report:
                result["missing_data_report"] = missing_report
            return result
        except Exception as e:
            return {"schema_version": SCHEMA_VERSION, "test_family": "chi_square",
                    "error": f"Fisher exact failed: {e.__class__.__name__}", "details": str(e)}

    # Else Chi-square
    try:
        # Yates correction only for 2x2; otherwise False
        correction = True if is_2x2 else False
        chi2, p, dof, expected2 = stats.chi2_contingency(ct.values, correction=correction)
        cramer_v = _cramers_v(chi2, ct.values)
        result = {
            "schema_version": SCHEMA_VERSION,
            "test_family": "chi_square",
            "chosen_test": "chi_square",
            "test_name": "Chi-square Test of Independence",
            "stats": {
                "chi2": float(chi2),
                "p_value": float(p),
                "df": int(dof),
                "table_shape": [r, c],
                "min_expected": min_expected,
                "yates_correction": bool(correction) if is_2x2 else False,
            },
            "effect_size": {"name": "Cramér's V", "value": cramer_v},
            "contingency_table": ct.to_dict(),
            "assumptions": {
                "expected_frequencies": {
                    "min": min_expected,
                    "rule_of_5_triggered": bool((min_expected is not None) and (min_expected < 5)),
                    "alpha": alpha,
                }
            },
            "warnings": warnings,
            "plot_path": plot_path,
        }
        if (not is_2x2) and (min_expected is not None) and (min_expected < 5):
            warnings.append(
                "Some expected counts < 5 in a table larger than 2×2; chi-square used (Fisher not available for >2×2). "
                "Interpret with caution."
            )
        if missing_report:
            result["missing_data_report"] = missing_report
        return result
    except Exception as e:
        return {"schema_version": SCHEMA_VERSION, "test_family": "chi_square",
                "error": f"Chi-square failed: {e.__class__.__name__}", "details": str(e)}
