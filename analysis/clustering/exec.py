# analysis/clustering/exec.py
"""
Clustering pipeline (K-Means) with local imputation, scaling, model selection, and plotting.

Design:
- This module performs its OWN feature-level missing handling (independent from the test-oriented missing node).
- Numeric: median imputation; Categorical (optional, low-cardinality): mode imputation + one-hot.
- Standardization: default True (StandardScaler). Applied AFTER encoding.
- Model selection:
    * n_clusters: int -> use directly
    * n_clusters: "auto" -> sweep k in [k_min, k_max], pick best silhouette (ties -> smallest k).
- Outputs:
    * schema_version, test_family="clustering", method="kmeans"
    * preprocessing_report: selected/dropped features, imputation, scaling
    * model_report: k chosen, silhouette, cluster sizes
    * cluster_profiles: per-cluster means on ORIGINAL (imputed) feature scale
    * plot_paths: {"pca_scatter": ..., "centroid_heatmap": ...}

Note:
- Upstream missing-data report (if present) can be attached by the caller as `missing_report`; we include it as
  `missing_data_report` for transparency but do not rely on it for clustering imputations.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import os
import tempfile

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


SCHEMA_VERSION = "1.0"

# -----------------------------
# Helpers: plotting
# -----------------------------
def _pca_scatter_to_tempfile(X_scaled: np.ndarray,
                             labels: np.ndarray,
                             title: str = "K-Means (PCA 2D)") -> str:
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    # one color per cluster (matplotlib default cycle)
    for k in np.unique(labels):
        mask = labels == k
        ax.plot(Z[mask, 0], Z[mask, 1], "o", ms=4, alpha=0.7, label=f"Cluster {int(k)}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fd, path = tempfile.mkstemp(suffix=".png", prefix="cluster_pca_", text=False)
    os.close(fd)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _centroid_heatmap_to_tempfile(cluster_profiles: pd.DataFrame,
                                  title: str = "Cluster profiles (feature means)") -> str:
    # cluster_profiles: index = cluster id, columns = features (original units)
    data = cluster_profiles.copy().T  # features as rows, clusters as columns for readability
    fig_h = max(4, 0.3 * data.shape[0] + 1.5)
    fig_w = max(6, 1.2 * data.shape[1] + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    im = ax.imshow(data.values, aspect="auto")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(list(data.index))
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels([str(c) for c in data.columns])
    ax.set_xlabel("Cluster")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fd, path = tempfile.mkstemp(suffix=".png", prefix="cluster_heatmap_", text=False)
    os.close(fd)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------
# Helpers: feature selection & imputation
# -----------------------------
def _select_features(df: pd.DataFrame,
                     metadata: Dict[str, str],
                     features: Optional[List[str]],
                     include_categoricals: bool,
                     max_cat_cardinality: int) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Returns (selected_features, dropped_with_reasons).
    - If `features` provided: filter by availability & type constraints.
    - Else: select all numeric features; optionally include low-cardinality categoricals.
    """
    dropped: List[Dict[str, str]] = []

    if features:
        candidates = [c for c in features if c in df.columns]
    else:
        # default: all numericals
        candidates = [c for c, t in metadata.items() if t == "numerical" and c in df.columns]
        if include_categoricals:
            cats = [c for c, t in metadata.items() if t == "categorical" and c in df.columns]
            for c in cats:
                card = int(df[c].nunique(dropna=True))
                if card <= max_cat_cardinality:
                    candidates.append(c)
                else:
                    dropped.append({"name": c, "reason": f"high_cardinality({card})"})

    selected: List[str] = []
    for c in candidates:
        t = metadata.get(c, "other")
        if t == "numerical":
            selected.append(c)
        elif t == "categorical" and include_categoricals:
            card = int(df[c].nunique(dropna=True))
            if card <= max_cat_cardinality:
                selected.append(c)
            else:
                dropped.append({"name": c, "reason": f"high_cardinality({card})"})
        else:
            dropped.append({"name": c, "reason": f"unsupported_type({t})"})

    # deduplicate preserving order
    sel_unique = []
    seen = set()
    for c in selected:
        if c not in seen:
            sel_unique.append(c)
            seen.add(c)

    return sel_unique, dropped


def _impute_and_encode(df: pd.DataFrame,
                       metadata: Dict[str, str],
                       selected: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Impute and encode the selected features.
    - Numeric: median
    - Categorical: mode, then one-hot (drop_first=True), NaN category not created (already imputed)
    Returns:
      X_raw_imputed (original scale, numerics + original cat columns),
      imputation_report (per feature strategy + value),
      X_model (numeric-only, one-hot encoded and imputed)
    """
    X_raw = df[selected].copy()
    imputation_report: Dict[str, Any] = {}

    # First, impute raw
    for c in selected:
        t = metadata.get(c, "other")
        if t == "numerical":
            med = float(X_raw[c].median())
            X_raw[c] = X_raw[c].fillna(med)
            imputation_report[c] = {"type": "numerical", "strategy": "median", "value": med}
        elif t == "categorical":
            modes = X_raw[c].mode(dropna=True)
            val = None if modes.empty else modes.iloc[0]
            X_raw[c] = X_raw[c].fillna(val)
            imputation_report[c] = {"type": "categorical", "strategy": "mode", "value": val}
        else:
            # Shouldn't happen due to selection, but guard anyway
            X_raw[c] = X_raw[c].fillna(X_raw[c].mode(dropna=True).iloc[0] if X_raw[c].mode(dropna=True).size else None)
            imputation_report[c] = {"type": "other", "strategy": "mode"}

    # Build model matrix: numerics unchanged, categoricals one-hot (drop_first=True)
    num_cols = [c for c in selected if metadata.get(c) == "numerical"]
    cat_cols = [c for c in selected if metadata.get(c) == "categorical"]

    X_model = pd.DataFrame(index=X_raw.index)
    if num_cols:
        X_model[num_cols] = X_raw[num_cols].astype(float)
    if cat_cols:
        dummies = pd.get_dummies(X_raw[cat_cols], drop_first=True, dummy_na=False)
        X_model = pd.concat([X_model, dummies], axis=1)

    return X_raw, imputation_report, X_model


# -----------------------------
# Main implementation
# -----------------------------
def clustering_impl(
    df: pd.DataFrame,
    metadata: Dict[str, str],
    *,
    features: Optional[List[str]] = None,
    n_clusters: Optional[int | str] = "auto",   # int or "auto"
    k_min: int = 2,
    k_max: int = 8,
    include_categoricals: bool = False,
    max_cat_cardinality: int = 8,
    standardize: bool = True,
    random_state: int = 0,
    missing_report: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    K-Means clustering over a selected feature set with local imputation & scaling.

    Parameters:
      - features: explicit list to use; if None -> auto-select numerics (+ optional low-card cats)
      - n_clusters: integer K or "auto" for silhouette search in [k_min, k_max]
    """

    # ---- Feature selection
    selected, dropped = _select_features(
        df=df,
        metadata=metadata,
        features=features,
        include_categoricals=include_categoricals,
        max_cat_cardinality=max_cat_cardinality,
    )

    if len(selected) < 2:
        return {
            "schema_version": SCHEMA_VERSION,
            "test_family": "clustering",
            "error": "At least 2 usable features are required for clustering.",
            "selected_features": selected,
            "dropped_features": dropped,
        }

    # ---- Impute + encode
    X_raw, impute_rep, X_model = _impute_and_encode(df, metadata, selected)

    if X_model.shape[1] < 2:
        return {
            "schema_version": SCHEMA_VERSION,
            "test_family": "clustering",
            "error": "Model matrix after encoding has fewer than 2 columns.",
            "selected_features": selected,
            "encoded_columns": list(X_model.columns),
        }

    # ---- Scale
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_model.values)
        scaling_rep = {"standardize": True, "scaler": "StandardScaler", "n_features": int(X_model.shape[1])}
    else:
        X_scaled = X_model.values
        scaler = None
        scaling_rep = {"standardize": False, "scaler": None, "n_features": int(X_model.shape[1])}

    # ---- Choose K
    def _fit_kmeans(k: int) -> Tuple[KMeans, np.ndarray, float]:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X_scaled)
        # silhouette requires > 1 cluster and < n_samples clusters; guard for tiny datasets
        sil = silhouette_score(X_scaled, labels) if (k > 1 and k < X_scaled.shape[0]) else np.nan
        return km, labels, float(sil) if np.isfinite(sil) else np.nan

    if isinstance(n_clusters, str) and n_clusters.lower() == "auto":
        k_candidates = [k for k in range(max(2, k_min), max(k_min, k_max) + 1)]
        fits = []
        for k in k_candidates:
            km, labels, sil = _fit_kmeans(k)
            fits.append((k, km, labels, sil))
        # pick best silhouette; tie -> smallest k
        best = max(fits, key=lambda t: (t[3], -t[0]))  # (k, model, labels, silhouette)
        k_best, km_model, labels, sil_best = best
    else:
        k_best = int(n_clusters) if n_clusters is not None else 3
        km_model, labels, sil_best = _fit_kmeans(k_best)

    # ---- Profiles in original (imputed) feature space
    # Build original-scale table (numerics in original units, categoricals as original categories)
    # For profiles, use only original selected columns in X_raw
    cluster_ids = np.unique(labels)
    profiles = []
    for k in cluster_ids:
        idx = (labels == k)
        means = X_raw.loc[idx, selected].apply(
            lambda s: float(np.mean(s)) if pd.api.types.is_numeric_dtype(s) else None
        )
        profiles.append(pd.Series(means, name=int(k)))
    cluster_profiles = pd.DataFrame(profiles).T  # features x clusters? we’ll keep clusters as columns later for heatmap

    # ---- Summaries
    sizes = {int(k): int(np.sum(labels == k)) for k in cluster_ids}

    # ---- Plots
    pca_path = _pca_scatter_to_tempfile(X_scaled, labels, title=f"K-Means (k={k_best}) — PCA 2D")
    heatmap_path = _centroid_heatmap_to_tempfile(cluster_profiles.T, title="Cluster profiles (feature means)")

    # ---- Result envelope
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "test_family": "clustering",
        "method": "kmeans",
        "method_params": {"k": int(k_best), "selection": "auto" if (isinstance(n_clusters, str) and n_clusters == "auto") else "fixed"},
        "preprocessing_report": {
            "selected_features": selected,
            "dropped_features": dropped,
            "imputation": impute_rep,
            "scaling": scaling_rep,
            "encoded_columns": list(X_model.columns),
            "n_samples": int(X_model.shape[0]),
            "n_model_features": int(X_model.shape[1]),
        },
        "model_report": {
            "silhouette": float(sil_best) if np.isfinite(sil_best) else None,
            "cluster_sizes": sizes,
        },
        "cluster_profiles": {feat: {str(col): (None if pd.isna(val) else float(val))
                                    for col, val in cluster_profiles.loc[feat].items()}
                             for feat in cluster_profiles.index},
        "plot_paths": {
            "pca_scatter": pca_path,
            "centroid_heatmap": heatmap_path,
        },
        "plot_path": pca_path,
        # keep assignments sample small to avoid huge payloads
        "cluster_assignments_sample": [
            {"index": int(i) if isinstance(i, (int, np.integer)) else str(i), "cluster": int(lbl)}
            for i, lbl in list(zip(X_raw.index, labels))[:50]
        ],
    }

    if missing_report:
        # Attach upstream test-oriented missing-data report for transparency
        result["missing_data_report"] = missing_report

    return result
