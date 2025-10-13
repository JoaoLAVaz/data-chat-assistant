# analysis/correlation/exec.py
"""
Correlation test (auto Pearson vs Spearman) with normality checks and plotting.

Flow:
- Assumes missing-data handling upstream (missing_data_node) -> no NaNs here.
- Shapiro–Wilk per variable:
    * both normal -> Pearson
    * else        -> Spearman
- Returns a dict suitable for LLM interpretation (or a later narrative node),
  including a temp-file scatter plot path.
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


# -----------------------------
# Small helpers
# -----------------------------
def _shapiro_safe(x: np.ndarray, min_n: int = 3) -> Dict[str, Optional[float]]:
    """Shapiro–Wilk with guards; returns {'W', 'p', 'note'}."""
    out = {"W": None, "p": None, "note": None}
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < min_n:
        out["note"] = f"n<{min_n}; normality not testable."
        return out
    # Shapiro fails on constant arrays
    if np.allclose(x, x[0]):
        out["note"] = "constant array; normality not testable."
        return out
    try:
        W, p = stats.shapiro(x)
        out["W"] = float(W)
        out["p"] = float(p)
        return out
    except Exception as e:
        out["note"] = f"Shapiro failed: {e.__class__.__name__}"
        return out


def _pearson_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    """Approximate two-sided CI for Pearson's r using Fisher z transform."""
    if n <= 3 or not np.isfinite(r):
        return None, None
    try:
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        lo, hi = z - z_crit * se, z + z_crit * se
        r_lo, r_hi = np.tanh([lo, hi])
        return float(r_lo), float(r_hi)
    except Exception:
        return None, None


def _scatter_to_tempfile(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str,
                         add_ls_line: bool = True) -> str:
    """Scatter plot (optionally with least-squares line) -> temp PNG path."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    ax.plot(x, y, "o", ms=4, alpha=0.6)

    if add_ls_line and x.size >= 2 and np.std(x) > 0 and np.std(y) > 0:
        # simple least squares line on raw data
        try:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(np.min(x), np.max(x), 100)
            ax.plot(xs, m * xs + b, "-", lw=1.5)
        except Exception:
            pass

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)

    fd, path = tempfile.mkstemp(suffix=".png", prefix="corr_", text=False)
    os.close(fd)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------
# Main implementation
# -----------------------------
def correlation_impl(
    df: pd.DataFrame,
    metadata: Dict[str, str],
    *,
    var1: str,
    var2: str,
    method: str = "auto",  # "auto" | "pearson" | "spearman"
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Auto-select Pearson vs Spearman based on normality (unless method overrides).
    Assumes upstream missing-data handling; returns error if NA detected here.
    """

    # --- Validate inputs/types ---
    for v in (var1, var2):
        if v not in metadata:
            return {"error": f"Column '{v}' not found in dataset"}
        if metadata[v] != "numerical":
            return {"error": f"'{v}' must be numerical, but is {metadata[v]}"}

    data = df[[var1, var2]].copy()

    # Enforce no-missing invariant
    na_counts = data.isna().sum().to_dict()
    if int(sum(na_counts.values())) > 0:
        return {
            "error": "Missing values detected in required columns after preprocessing.",
            "details": {"missing_by_column": {k: int(v) for k, v in na_counts.items()}},
        }

    x = data[var1].astype(float).to_numpy()
    y = data[var2].astype(float).to_numpy()
    n = int(len(data))

    # Basic sanity: need at least 3 points
    if n < 3:
        return {"error": "Correlation requires at least 3 paired observations.", "n": n}

    # Normality checks (only used when method == 'auto')
    norm_x = _shapiro_safe(x)
    norm_y = _shapiro_safe(y)

    # Decide method
    chosen = method.lower()
    warnings: List[str] = []

    if chosen not in {"auto", "pearson", "spearman"}:
        chosen = "auto"

    if chosen == "auto":
        x_normal = (norm_x.get("p") is not None) and (norm_x["p"] >= alpha)
        y_normal = (norm_y.get("p") is not None) and (norm_y["p"] >= alpha)
        # if either normality test not available (e.g., constant or n<3), lean to Spearman
        if norm_x.get("p") is None or norm_y.get("p") is None:
            warnings.append("Normality not testable for one or both variables; defaulting to Spearman.")
            chosen = "spearman"
        else:
            chosen = "pearson" if (x_normal and y_normal) else "spearman"

    # Compute correlation
    if chosen == "pearson":
        try:
            r, p = stats.pearsonr(x, y)
        except Exception as e:
            # Fallback to Spearman if Pearson fails (e.g., constant series)
            warnings.append(f"Pearson failed ({e.__class__.__name__}); falling back to Spearman.")
            chosen = "spearman"
            r, p = stats.spearmanr(x, y, alternative="two-sided")
    else:
        r, p = stats.spearmanr(x, y, alternative="two-sided")

    # Confidence interval for Pearson only
    ci_lo, ci_hi = (None, None)
    if chosen == "pearson" and np.isfinite(r):
        ci_lo, ci_hi = _pearson_ci(r, n, alpha=alpha)

    # Plot
    title = f"{'Pearson' if chosen=='pearson' else 'Spearman'} correlation: {var1} vs {var2}"
    plot_path = _scatter_to_tempfile(x, y, xlabel=var1, ylabel=var2, title=title, add_ls_line=True)

    # Package results
    result: Dict[str, Any] = {
        "chosen_test": "pearson" if chosen == "pearson" else "spearman",
        "test_name": "Pearson correlation" if chosen == "pearson" else "Spearman correlation",
        "stats": {
            "r": float(r) if np.isfinite(r) else None,
            "p_value": float(p) if np.isfinite(p) else None,
            "n": n,
        },
        "effect_size": {"name": "correlation_r", "value": float(r) if np.isfinite(r) else None},
        "assumptions": {
            "normality": {
                var1: norm_x,
                var2: norm_y,
                "alpha": alpha,
                "both_normal": bool(
                    (norm_x.get("p") is not None) and (norm_y.get("p") is not None) and
                    (norm_x["p"] >= alpha) and (norm_y["p"] >= alpha)
                ),
            }
        },
        "warnings": warnings,
        "plot_path": plot_path,
    }

    if chosen == "pearson":
        result["confidence_interval_95"] = {"lower": ci_lo, "upper": ci_hi, "method": "Fisher z"}

    return result
