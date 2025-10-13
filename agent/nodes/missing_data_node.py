"""
Missing-data preprocessing node.

Policy (high-level):
- Prefer ADVANCED IMPUTATION (hybrid) by default.
- Only do LISTWISE DELETION if:
    (a) MCAR test is available and p >= alpha (unbiased to drop), OR
    (b) missingness is extreme (> extreme_threshold; default 50%).
- A proxy test (missingness vs group) is computed for 1 numeric + 1 categorical
  (typical t-test); if proxy shows dependence (p < alpha), we bias to IMPUTE.

Adds a moderate-warning tier:
- If missing_rate is in (impute_threshold, extreme_threshold], we still IMPUTE
  but add a warning to the report.

Scopes to the columns required by the pending tool call, with two modes:

- scope == "local":
    * Only the two required columns are analyzed/cleaned.
    * Imputation is median/mode (simple).

- scope == "hybrid" (default):
    * Only CLEAN the two required columns, but may use the ENTIRE DATASET
      as predictors to impute the numeric target.
    * Diagnostics:
        - Little's MCAR on numeric columns among [target] + selected numeric predictors (>=2).
        - Proxy test: independence of missing(target) vs group (for t-tests).
    * Imputation (numeric target only):
        - Prefer IterativeImputer (BayesianRidge); fallback to KNNImputer; fallback to median.
        - Categorical predictors are one-hot encoded (only if low-cardinality).
        - High-cardinality or highly-missing predictors are skipped to keep things light.

Writes to analysis_context:
  - "working_df": cleaned subset dataframe (only the two required columns)
  - "missing_report": pure-Python dict describing decisions & stats
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from langchain_core.messages import AIMessage
from ..state import AgentState

NEEDS_MISSING_PIPELINE = {"t_test", "anova_test", "correlation_test", "chi_square_test"}

# ----------------- Helpers: config & column picking ----------------- #

def _required_cols(tool_name: str, args: Dict[str, Any]) -> List[str]:
    if tool_name in {"t_test", "anova_test"}:
        return [args.get("group_col"), args.get("value_col")]
    if tool_name in {"correlation_test", "chi_square_test"}:
        return [args.get("var1"), args.get("var2")]
    return []

def _col_kind(col: str, metadata: Dict[str, str]) -> str:
    # 'numerical' | 'categorical' | 'datetime' | 'other'
    return metadata.get(col, "other")

def _config(state: AgentState) -> Dict[str, Any]:
    cfg = (state.get("config") or {}).get("missing") or {}
    return {
        "alpha": float(cfg.get("alpha", 0.05)),
        "impute_threshold": float(cfg.get("impute_threshold", 0.20)),      # ≤20% → impute (no warning)
        "extreme_threshold": float(cfg.get("extreme_threshold", 0.50)),    # >50% → delete
        "force_impute": bool(cfg.get("force_impute", False)),
        "scope": str(cfg.get("scope", "hybrid")),                          # "local" | "hybrid"
        "max_cat_cardinality": int(cfg.get("max_cat_cardinality", 50)),
        "max_pred_missing": float(cfg.get("max_pred_missing", 0.50)),      # drop predictors >50% missing
    }

# ----------------- MCAR (Little) & Proxy tests ----------------- #

def _mcar_pingouin_numeric_only(df_numeric: pd.DataFrame) -> Tuple[bool, Optional[float], Dict[str, Any]]:
    """
    Try Little's MCAR via pingouin on numeric columns only.
    Supports Pingouin 0.5.5 (little_mcar) and newer alias (mcar) if present.
    Returns (available, p_value, extra_info_dict).
    available=False → could not compute (no function, <2 numeric cols, or error).
    """
    info = {"stat": None, "df": None, "p": None, "note": None, "numeric_cols_used": list(df_numeric.columns)}
    if df_numeric.shape[1] < 2:
        info["note"] = "MCAR requires ≥2 numeric columns; skipped."
        return False, None, info
    try:
        # Try importing little_mcar directly (Pingouin 0.5.5 may not export it; handle gracefully)
        try:
            from pingouin import little_mcar as _little_mcar  # type: ignore
            res = _little_mcar(df_numeric)
        except Exception:
            import pingouin as pg
            if hasattr(pg, "little_mcar"):
                res = pg.little_mcar(df_numeric)
            elif hasattr(pg, "mcar"):
                res = pg.mcar(df_numeric)
            else:
                info["note"] = "No MCAR function found in this Pingouin version."
                return False, None, info

        chi2 = float(res.loc[0, "chi2"])
        dof = int(res.loc[0, "dof"])
        pval = float(res.loc[0, "pval"])
        info.update({"stat": chi2, "df": dof, "p": pval})
        return True, pval, info

    except Exception as e:
        info["note"] = f"MCAR test failed: {e.__class__.__name__}"
        return False, None, info

def _missingness_vs_group_proxy(subset: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    """
    Proxy (not Little's): test independence between missing(value_col) and group_col categories.
    """
    from scipy.stats import chi2_contingency, fisher_exact
    out = {"available": False, "method": None, "p": None, "note": None}
    if group_col not in subset.columns or value_col not in subset.columns:
        out["note"] = "Required columns not present."
        return out
    miss = subset[value_col].isna()
    tab = pd.crosstab(subset[group_col], miss)
    if tab.shape[1] != 2:
        out["note"] = "Missingness is all-missing or all-observed; proxy test not applicable."
        return out
    out["available"] = True
    if tab.shape == (2, 2) and (tab.values < 5).any():
        try:
            _, p = fisher_exact(tab.values)
            out.update({"method": "fisher", "p": float(p)})
        except Exception as e:
            out.update({"available": False, "note": f"Fisher test failed: {e.__class__.__name__}"})
    else:
        try:
            _, p, _, _ = chi2_contingency(tab, correction=True)
            out.update({"method": "chi2", "p": float(p)})
        except Exception as e:
            out.update({"available": False, "note": f"Chi-square test failed: {e.__class__.__name__}"})
    return out

# ----------------- Hybrid-mode predictor assembly & imputation ----------------- #

def _select_predictors_for_hybrid(df: pd.DataFrame,
                                  target_col: str,
                                  metadata: Dict[str, str],
                                  max_cat_card: int,
                                  max_pred_missing: float) -> Dict[str, Any]:
    """
    Select lightweight predictors from the whole dataset to impute a numeric target.
    - Keep numeric predictors with <= max_pred_missing missing rate.
    - Keep low-cardinality categoricals (<= max_cat_card) with <= max_pred_missing missing rate.
    - Skip datetime/other/high-cardinality/noisy features.
    Returns dict with:
      - X_enc (DataFrame): encoded predictors (all numeric, NaNs present)
      - selected_predictors (list)
      - dropped_predictors (list of {"name":..., "reason":...})
    """
    selected, dropped = [], []
    cols = [c for c in df.columns if c != target_col]
    for c in cols:
        kind = metadata.get(c, "other")
        miss_rate = float(df[c].isna().mean())
        if miss_rate > max_pred_missing:
            dropped.append({"name": c, "reason": f"missing_rate>{max_pred_missing:.2f}"})
            continue
        if kind == "numerical":
            selected.append(c)
        elif kind == "categorical":
            card = int(df[c].nunique(dropna=True))
            if card <= max_cat_card:
                selected.append(c)
            else:
                dropped.append({"name": c, "reason": f"high_cardinality({card})"})
        else:
            dropped.append({"name": c, "reason": f"unsupported_type({kind})"})

    if not selected:
        return {"X_enc": pd.DataFrame(index=df.index), "selected_predictors": [], "dropped_predictors": dropped}

    cats = [c for c in selected if metadata.get(c, "other") == "categorical"]
    nums = [c for c in selected if metadata.get(c, "other") == "numerical"]

    X = pd.DataFrame(index=df.index)
    if nums:
        X[nums] = df[nums]
    if cats:
        X = pd.concat([X, pd.get_dummies(df[cats], drop_first=True, dummy_na=False)], axis=1)

    return {"X_enc": X, "selected_predictors": selected, "dropped_predictors": dropped}

def _impute_numeric_target_hybrid(df: pd.DataFrame,
                                  target_col: str,
                                  X_enc: pd.DataFrame) -> Dict[str, Any]:
    """
    Impute only the target numeric column using predictors X_enc.
    Try IterativeImputer (BayesianRidge) → KNNImputer → median.
    Returns dict with:
      - y_imputed (Series with imputed target, index-aligned)
      - method ('iterative'|'knn'|'median'|'none')
      - details (dict)
    """
    y = df[target_col].copy()
    details: Dict[str, Any] = {}

    if not y.isna().any():
        return {"y_imputed": y, "method": "none", "details": {"note": "no missing in target"}}

    if X_enc.shape[1] == 0:
        med = float(y.median())
        y_filled = y.fillna(med)
        return {"y_imputed": y_filled, "method": "median", "details": {"value": med}}

    X_clean = X_enc.copy()
    dummy_cols = [c for c in X_clean.columns if X_clean[c].dropna().isin([0, 1]).all()]
    if dummy_cols:
        X_clean[dummy_cols] = X_clean[dummy_cols].fillna(0)
    other_cols = [c for c in X_clean.columns if c not in dummy_cols]
    for c in other_cols:
        X_clean[c] = X_clean[c].fillna(X_clean[c].median())

    # IterativeImputer
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        mat = pd.concat([y, X_clean], axis=1)
        imp = IterativeImputer(random_state=0, sample_posterior=False, max_iter=10, initial_strategy="median")
        imputed = imp.fit_transform(mat.values)
        y_imp = pd.Series(imputed[:, 0], index=y.index, name=target_col)
        details.update({"n_features": int(mat.shape[1]), "max_iter": 10})
        return {"y_imputed": y_imp, "method": "iterative", "details": details}
    except Exception as e_iter:
        details["iterative_error"] = e_iter.__class__.__name__

    # KNNImputer
    try:
        from sklearn.impute import KNNImputer

        mat = pd.concat([y, X_clean], axis=1)
        imp = KNNImputer(n_neighbors=5, weights="uniform")
        imputed = imp.fit_transform(mat.values)
        y_imp = pd.Series(imputed[:, 0], index=y.index, name=target_col)
        details.update({"n_features": int(mat.shape[1]), "n_neighbors": 5})
        return {"y_imputed": y_imp, "method": "knn", "details": details}
    except Exception as e_knn:
        details["knn_error"] = e_knn.__class__.__name__

    # Median fallback
    med = float(y.median())
    y_filled = y.fillna(med)
    return {"y_imputed": y_filled, "method": "median", "details": {"value": med, **details}}

# ----------------- Main node ----------------- #

def missing_data_node(state: AgentState) -> AgentState:
    msgs = state["messages"]
    last = msgs[-1] if msgs else None

    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return {}

    tool_call = None
    for c in last.tool_calls:
        if c["name"] in NEEDS_MISSING_PIPELINE:
            tool_call = c
            break
    if tool_call is None:
        return {}

    tool_name = tool_call["name"]
    args = tool_call.get("args", {}) or {}
    cols = [c for c in _required_cols(tool_name, args) if c]
    if len(cols) != 2:
        return {}

    df = state["df"]
    metadata = state["metadata"]
    cfg = _config(state)

    subset = df[cols].copy()
    rows_before = int(len(subset))

    total_missing = int(subset.isna().sum().sum())
    missing_by_col = {c: int(subset[c].isna().sum()) for c in cols}
    missing_rate = float(total_missing / subset.size) if subset.size > 0 else 0.0

    report: Dict[str, Any] = {
        "columns": cols,
        "rows_before": rows_before,
        "total_missing": total_missing,
        "missing_by_column": missing_by_col,
        "missing_rate": missing_rate,
        "policy": "none",
        "alpha": cfg["alpha"],
        "scope": cfg["scope"],
        "mcar": {"available": False, "stat": None, "df": None, "p": None, "note": None, "numeric_cols_used": []},
        "mcar_proxy": {"available": False, "method": None, "p": None, "note": None},
        "imputations": {},
        "warning": None,
        "config": {
            "impute_threshold": cfg["impute_threshold"],
            "extreme_threshold": cfg["extreme_threshold"],
            "force_impute": cfg["force_impute"],
            "max_cat_cardinality": cfg["max_cat_cardinality"],
            "max_pred_missing": cfg["max_pred_missing"],
        },
        "predictors": {"selected": [], "dropped": [], "encoded_features": 0},
        "imputer_details": {},
    }

    # No missing → pass-through
    if total_missing == 0:
        working = subset
        report["policy"] = "none"
    else:
        kinds = [_col_kind(c, metadata) for c in cols]

        # Numeric pool for MCAR (depends on scope)
        if cfg["scope"] == "hybrid":
            target_numeric = cols[kinds.index("numerical")] if "numerical" in kinds else None
            if target_numeric is not None:
                pred_sel = _select_predictors_for_hybrid(
                    df, target_numeric, metadata, cfg["max_cat_cardinality"], cfg["max_pred_missing"]
                )
                report["predictors"]["selected"] = pred_sel["selected_predictors"]
                report["predictors"]["dropped"] = pred_sel["dropped_predictors"]
                report["predictors"]["encoded_features"] = int(pred_sel["X_enc"].shape[1])

                df_numeric_for_mcar = pd.DataFrame(index=df.index)
                df_numeric_for_mcar[target_numeric] = df[target_numeric]
                num_preds_original = [c for c in pred_sel["selected_predictors"] if metadata.get(c) == "numerical"]
                if num_preds_original:
                    df_numeric_for_mcar = pd.concat([df_numeric_for_mcar, df[num_preds_original]], axis=1)
            else:
                df_numeric_for_mcar = subset.select_dtypes(include=["number"]).copy()
        else:
            df_numeric_for_mcar = subset.select_dtypes(include=["number"]).copy()

        # MCAR (numeric-only) if possible
        mcar_available, p_mcar, mcar_info = _mcar_pingouin_numeric_only(df_numeric_for_mcar)
        report["mcar"].update(mcar_info)
        if mcar_available:
            report["mcar"]["available"] = True

        # Proxy test (1 numeric + 1 categorical)
        if set(kinds) == {"numerical", "categorical"}:
            num_col = cols[kinds.index("numerical")]
            cat_col = cols[kinds.index("categorical")]
            proxy = _missingness_vs_group_proxy(subset[[cat_col, num_col]].copy(), cat_col, num_col)
            report["mcar_proxy"].update(proxy)
        else:
            proxy = {"available": False, "p": None}

        # -------- Decision policy (IMPUTE by default; warn for moderate; delete if MCAR or extreme) --------
        alpha = cfg["alpha"]
        prefer_impute_due_to_proxy = bool(proxy.get("available") and (proxy.get("p") is not None) and (proxy["p"] < alpha))

        if cfg["force_impute"]:
            decision = "simple_impute"
        elif (mcar_available and (p_mcar is not None) and (p_mcar >= alpha)):
            decision = "listwise_delete"           # MCAR supported → deletion is unbiased
        elif missing_rate > cfg["extreme_threshold"]:
            decision = "listwise_delete_high"      # extreme missing → delete with strong warning
        elif missing_rate > cfg["impute_threshold"]:
            decision = "simple_impute_warn"        # moderate missing → impute but warn
        else:
            decision = "simple_impute"             # small missing → impute silently

        # If proxy suggests dependence on group, bias to impute (override deletions)
        if decision.startswith("listwise_delete") and prefer_impute_due_to_proxy:
            decision = "simple_impute_warn"
            report["warning"] = "Proxy test suggests missingness depends on group; preferring imputation over deletion."

        # Apply decision
        if decision.startswith("listwise_delete"):
            working = subset.dropna(subset=cols)
            report["policy"] = "listwise_delete"
            if decision == "listwise_delete_high":
                report["warning"] = (
                    "High missingness (> extreme_threshold) in required columns; used listwise deletion. "
                    "Consider advanced/multiple imputation."
                )
        else:
            working = subset.copy()
            report["policy"] = "simple_impute"
            if decision == "simple_impute_warn" and report["warning"] is None:
                report["warning"] = (
                    "Moderate missingness (> impute_threshold). Used imputation; "
                    "results may be sensitive to imputation assumptions."
                )
            for c in cols:
                kind = _col_kind(c, metadata)
                if kind == "numerical":
                    if cfg["scope"] == "hybrid":
                        pred_sel = _select_predictors_for_hybrid(
                            df, c, metadata, cfg["max_cat_cardinality"], cfg["max_pred_missing"]
                        )
                        X_enc = pred_sel["X_enc"]
                        imp_res = _impute_numeric_target_hybrid(df[[c]].join(X_enc), c, X_enc)
                        working[c] = imp_res["y_imputed"].reindex(working.index)
                        report["imputations"][c] = {"strategy": imp_res["method"]}
                        report["imputer_details"] = imp_res.get("details", {})
                        report["predictors"]["selected"] = pred_sel["selected_predictors"]
                        report["predictors"]["dropped"] = pred_sel["dropped_predictors"]
                        report["predictors"]["encoded_features"] = int(X_enc.shape[1])
                    else:
                        val = float(working[c].median())
                        working[c] = working[c].fillna(val)
                        report["imputations"][c] = {"strategy": "median", "value": val}
                elif kind == "categorical":
                    modes = working[c].mode(dropna=True)
                    val = None if modes.empty else modes.iloc[0]
                    working[c] = working[c].fillna(val)
                    report["imputations"][c] = {"strategy": "mode", "value": val}
                else:
                    # conservative fallback
                    working = working.dropna(subset=[c])

    rows_after = int(len(working))
    report["rows_after"] = rows_after
    report["rows_dropped"] = int(rows_before - rows_after)

    # Write to analysis_context (do NOT mutate original df)
    ctx = dict(state.get("analysis_context") or {})
    ctx["working_df"] = working
    ctx["missing_report"] = report

    print(report)  # debug log

    return {"analysis_context": ctx}
