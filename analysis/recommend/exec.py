# analysis/recommend/exec.py
"""
Recommendation engine for feasible & interesting statistical tests.

Flow:
- Build deterministic, statistically valid candidate pairs/tuples from df+metadata.
- Rank + phrase with an LLM (small, swappable function).
- Return a JSON-serializable dict of recommendations.

Design notes:
- The ranking LLM call is isolated in `_rank_with_llm(...)`. Replace that
  with OSS+RAG later without touching the candidate builders.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import math

SCHEMA_VERSION = "1.0"

# ---------------------------
# Utilities: column stats
# ---------------------------

def _basic_col_profile(df: pd.DataFrame, metadata: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Compute simple profiles used by candidate builders.
    Returns: {col: {"kind","n_nonnull","missing_rate","n_unique","std"}}
    """
    prof: Dict[str, Dict[str, Any]] = {}
    for c in df.columns:
        kind = metadata.get(c, "other")
        s = df[c]
        n_nonnull = int(s.notna().sum())
        miss_rate = float(1 - n_nonnull / len(df)) if len(df) else 0.0
        nunq = int(s.nunique(dropna=True))
        std = float(s.std(ddof=1)) if kind == "numerical" else None
        prof[c] = {
            "kind": kind,
            "n_nonnull": n_nonnull,
            "missing_rate": miss_rate,
            "n_unique": nunq,
            "std": std,
        }
    return prof


def _missing_rate(series: pd.Series) -> float:
    return float(series.isna().mean())


def _safe_std(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    return float(np.nanstd(s, ddof=1)) if s.notna().sum() >= 2 else 0.0


def _unique_count(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").nunique(dropna=True))


def _pairwise_abs_corr(dfnum: pd.DataFrame) -> float:
    """Mean absolute pairwise correlation (excluding diagonal). Returns 0 if not computable."""
    try:
        if dfnum.shape[1] < 2:
            return 0.0
        c = dfnum.corr(numeric_only=True)
        m = c.values
        n = m.shape[0]
        idx = np.triu_indices(n, k=1)
        vals = np.abs(m[idx])
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else 0.0
    except Exception:
        return 0.0


def _score_feature_set(df: pd.DataFrame, feats: List[str]) -> float:
    """
    Heuristic score for a candidate clustering feature set.
    - higher average std is better
    - moderate correlations (not all 0.99) preferred
    - penalize missingness
    """
    sub = df[feats].apply(pd.to_numeric, errors="coerce")
    stds = [_safe_std(sub[c]) for c in feats]
    avg_std = float(np.mean(stds)) if stds else 0.0

    mean_abs_corr = _pairwise_abs_corr(sub)  # 0..1
    # prefer some correlation but not extremely high; transform to reward ~0.5
    corr_bonus = 1.0 - abs(mean_abs_corr - 0.5)  # peak at 0.5

    miss_rates = [_missing_rate(sub[c]) for c in feats]
    miss_penalty = float(np.mean(miss_rates))  # 0..1 (lower is better)

    return max(0.0, avg_std * (0.5 + 0.5 * corr_bonus) * (1.0 - miss_penalty))


def _prefix_buckets(cols: List[str]) -> Dict[str, List[str]]:
    """
    Group columns by simple prefix before first underscore.
    Example: 'lab_ldh' and 'lab_crp' -> bucket 'lab'.
    """
    buckets: Dict[str, List[str]] = {}
    for c in cols:
        if "_" in c:
            pfx = c.split("_", 1)[0]
            buckets.setdefault(pfx, []).append(c)
    return {k: v for k, v in buckets.items() if len(v) >= 2}


# ---------------------------
# Candidate builders
# ---------------------------

def _candidates_t_test(df: pd.DataFrame, meta: Dict[str, str], prof: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    t-test: 1 categorical (exactly 2 levels, reasonable counts) x 1 numerical (non-constant).
    """
    cats = [c for c, p in prof.items() if p["kind"] == "categorical" and p["n_unique"] == 2 and p["n_nonnull"] >= 10]
    nums = [c for c, p in prof.items() if p["kind"] == "numerical" and p["n_nonnull"] >= 10 and (p["std"] or 0) > 0]

    out: List[Dict[str, Any]] = []
    for g in cats:
        # group balance check (each level at least 2 obs)
        vc = df[g].dropna().value_counts()
        if (vc.min() if not vc.empty else 0) < 2:
            continue
        for v in nums:
            # make sure some overlap (rows with both non-null)
            nn = df[[g, v]].dropna().shape[0]
            if nn < 6:
                continue
            out.append({
                "test_family": "t_test",
                "group_col": g,
                "value_col": v,
                "n_pairs": nn,
                "group_levels": vc.index.tolist(),
                "group_sizes": vc.astype(int).to_dict(),
            })
    return out


def _candidates_anova(df: pd.DataFrame, meta: Dict[str, str], prof: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    ANOVA: 1 categorical (3-8 levels) x 1 numerical. Avoid tiny groups.
    """
    cats = [c for c, p in prof.items() if p["kind"] == "categorical" and 3 <= p["n_unique"] <= 8 and p["n_nonnull"] >= 15]
    nums = [c for c, p in prof.items() if p["kind"] == "numerical" and p["n_nonnull"] >= 15 and (p["std"] or 0) > 0]

    out: List[Dict[str, Any]] = []
    for g in cats:
        vc = df[g].dropna().value_counts()
        if (vc.min() if not vc.empty else 0) < 3:
            continue
        for v in nums:
            nn = df[[g, v]].dropna().shape[0]
            if nn < 9:  # loose floor
                continue
            out.append({
                "test_family": "anova",
                "group_col": g,
                "value_col": v,
                "n_pairs": nn,
                "group_levels": vc.index.tolist(),
                "group_sizes": vc.astype(int).to_dict(),
            })
    return out


def _candidates_correlation(df: pd.DataFrame, meta: Dict[str, str], prof: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Correlation: 2 numerical, non-constant, enough pairs.
    """
    nums = [c for c, p in prof.items() if p["kind"] == "numerical" and p["n_nonnull"] >= 10 and (p["std"] or 0) > 0]
    out: List[Dict[str, Any]] = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            a, b = nums[i], nums[j]
            nn = df[[a, b]].dropna().shape[0]
            if nn < 10:
                continue
            out.append({
                "test_family": "correlation",
                "var1": a,
                "var2": b,
                "n_pairs": nn,
            })
    return out


def _candidates_chisq(df: pd.DataFrame, meta: Dict[str, str], prof: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chi-square: 2 categorical, manageable cardinalities, not ultra-sparse (quick screen).
    """
    cats = [c for c, p in prof.items() if p["kind"] == "categorical" and 2 <= p["n_unique"] <= 8 and p["n_nonnull"] >= 20]
    out: List[Dict[str, Any]] = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            a, b = cats[i], cats[j]
            tab = pd.crosstab(df[a], df[b])
            if tab.values.sum() < 20:
                continue
            # simple sparsity screen: allow but penalize later if >30% zeros
            zero_frac = float((tab.values == 0).sum() / tab.size)
            out.append({
                "test_family": "chi_square",
                "var1": a,
                "var2": b,
                "n_pairs": int(tab.values.sum()),
                "levels_var1": tab.index.astype(str).tolist(),
                "levels_var2": tab.columns.astype(str).tolist(),
                "zero_fraction": zero_frac,
            })
    return out


# --------- NEW: clustering candidates ----------------------------------------

def _prefix_groups(cols: List[str]) -> Dict[str, List[str]]:
    """Same as _prefix_buckets but kept separate for clarity in this section."""
    return _prefix_buckets(cols)

def _candidates_clustering(df: pd.DataFrame, meta: Dict[str, str], prof: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clustering (K-means):
    - Select numeric features with acceptable missingness (<70%) and variation (unique >= 3).
    - Propose several candidate feature sets:
        * top-variance sets (size 3..5)
        * thematic prefix buckets (>=2 features)
        * a domain-smart set if present (age, pack_years, performance_status, bmi, weight, height)
    - Score with a simple heuristic to rank later by LLM.
    """
    numeric_cols = [c for c, t in meta.items() if t == "numerical" and c in df.columns]
    if len(numeric_cols) < 2:
        return []

    # Filter by missingness & variation
    filtered: List[str] = []
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if _missing_rate(s) >= 0.70:
            continue
        if _unique_count(s) < 3:
            continue
        filtered.append(c)

    if len(filtered) < 2:
        return []

    # (a) top-variance sets
    var_sorted = sorted(filtered, key=lambda c: _safe_std(df[c]), reverse=True)
    top_names = var_sorted[:8]
    cand_sets: List[List[str]] = []
    for size in (3, 4, 5):
        if len(top_names) >= size:
            cand_sets.append(top_names[:size])

    # (b) thematic prefix buckets
    buckets = _prefix_groups(filtered)
    for _, cols in buckets.items():
        cols_sorted = sorted(cols, key=lambda c: _safe_std(df[c]), reverse=True)[:5]
        if len(cols_sorted) >= 2:
            cand_sets.append(cols_sorted)

    # (c) domain-smart
    domain = ["age", "pack_years", "performance_status", "bmi", "weight", "height"]
    domain_set = [c for c in domain if c in filtered]
    if len(domain_set) >= 2:
        cand_sets.append(domain_set[:5])

    # de-duplicate
    seen = set()
    uniq_sets: List[List[str]] = []
    for s in cand_sets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            uniq_sets.append(list(key))

    out: List[Dict[str, Any]] = []
    for feats in uniq_sets:
        score = _score_feature_set(df, feats)
        out.append({
            "test_family": "clustering",
            "features": feats,
            "score": float(score),
            "n_features": len(feats),
        })
    return out


# ---------------------------
# Ranking / phrasing (LLM)
# ---------------------------

def _rank_with_llm(
    model_name: str,
    dataset_hint: Optional[str],
    test_family: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Calls a small LLM to rank + phrase. Swappable later with OSS+RAG.
    Returns a list of items with fields:
      - short_title
      - rationale
      - variables (family-specific)
      - (optional) suggested_params for clustering
    """
    if not candidates:
        return []

    family = test_family.lower()

    # Compact candidate summary sent to the model
    brief_items = []
    for c in candidates:
        if family in ("t_test", "anova"):
            brief_items.append({
                "group_col": c["group_col"],
                "value_col": c["value_col"],
                "group_levels": c.get("group_levels", []),
                "n_pairs": c.get("n_pairs"),
            })
        elif family == "correlation":
            brief_items.append({
                "var1": c["var1"], "var2": c["var2"], "n_pairs": c.get("n_pairs")
            })
        elif family == "chi_square":
            brief_items.append({
                "var1": c["var1"], "var2": c["var2"],
                "levels_var1": c.get("levels_var1", []),
                "levels_var2": c.get("levels_var2", []),
                "n_pairs": c.get("n_pairs"),
            })
        else:  # clustering
            brief_items.append({
                "features": c["features"],
                "n_features": c.get("n_features", len(c["features"])),
                "score": float(c.get("score", 0.0)),
            })

    # Use OpenAI (or whatever your environment provides). Kept isolated.
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        import json, re

        sys = SystemMessage(content=(
            "You are a data analysis assistant. You will receive a brief dataset topic/industry hint and a list of "
            "feasible candidate analyses for a specified test family. Rank the candidates by (1) potential insight for the "
            "stated context, (2) clarity/interpretability, and (3) data adequacy.\n\n"
            "General guidance:\n"
            "- Prefer analyses that relate predictors to meaningful outcomes or KPIs (e.g., revenue, conversion, churn, "
            "defect rate, lead time, satisfaction, survival) when applicable.\n"
            "- Prefer domain-relevant predictors (e.g., treatment/stage in healthcare; campaign/segment in marketing; "
            "category/price in retail; device/region/feature-use in product analytics; grade/attendance in education; "
            "sensor/line settings in manufacturing). Avoid trivial demographic-only analyses unless context suggests value.\n"
            "- Penalize very small overlap/sample sizes, ultra-sparse contingency tables, and extremely high-cardinality "
            "categoricals that harm interpretability.\n"
            "- When the family is 'clustering', favor small (2–5), interpretable, non-redundant numeric feature sets with "
            "reasonable variation and low missingness. Prefer sets that could map to actionable segments in the given context.\n\n"
            "Output: Return the top K recommendations as JSON (a list). Each item must include:\n"
            "  - short_title: concise, human-friendly test description\n"
            "  - rationale: ≤25 words on why it's interesting/useful\n"
            "  - variables: the exact variable mapping required by the family (do NOT invent names)\n"
            "For clustering items, also include:\n"
            "  - suggested_params: {\"method\":\"kmeans\",\"n_clusters\":\"auto\",\"k_range\":[2,4]} "
            "(adjust k_range only if the feature set clearly implies a different small range).\n"
            "Do not output anything except the JSON list."
        ))
        user = HumanMessage(content=(
            f"Dataset topic: {dataset_hint or 'N/A'}\n"
            f"Test family: {test_family}\n"
            f"Top K: {top_k}\n"
            f"Candidates (JSON): {brief_items}"
        ))

        llm = ChatOpenAI(model=model_name, temperature=0.2, max_tokens=700)
        resp = llm.invoke([sys, user])
        text = resp.content or ""

        m = re.search(r"\[.*\]", text, flags=re.S)
        parsed = json.loads(m.group(0)) if m else json.loads(text)
        recs = parsed if isinstance(parsed, list) else []
        return recs[:top_k]
    except Exception:
        # Fallback: crude scoring
        def score_item(it: Dict[str, Any]) -> float:
            blob = " ".join(map(str, it.values())).lower()
            boosts = ["survival", "response", "treatment", "stage", "histology", "progression"]
            base = sum(w in blob for w in boosts) + float(it.get("n_pairs", 0)) / 50.0
            if family == "clustering":
                # prefer fewer features (2–4), and use given 'score' as heuristic
                nf = it.get("n_features", 3)
                feat_bonus = 1.0 if 2 <= nf <= 4 else 0.5
                base += feat_bonus + float(it.get("score", 0.0))
            return base

        ranked = sorted(brief_items, key=score_item, reverse=True)[:top_k]
        out = []
        for it in ranked:
            if family in ("t_test", "anova"):
                title = f"{it['group_col']} → {it['value_col']}"
                vars_map = {"group_col": it["group_col"], "value_col": it["value_col"]}
                extra = {}
            elif family == "correlation":
                title = f"{it['var1']} ↔ {it['var2']}"
                vars_map = {"var1": it["var1"], "var2": it["var2"]}
                extra = {}
            elif family == "chi_square":
                title = f"{it['var1']} × {it['var2']}"
                vars_map = {"var1": it["var1"], "var2": it["var2"]}
                extra = {}
            else:
                feats = it.get("features", [])
                title = f"K-Means on {', '.join(feats[:3])}" + ("" if len(feats) <= 3 else f" + {len(feats)-3} more")
                vars_map = {"features": feats}
                extra = {
                    "suggested_params": {"method": "kmeans", "n_clusters": "auto", "k_range": [2, 4]}
                }
            out.append({
                "short_title": title,
                "rationale": "Likely interpretable; adequate sample/variation.",
                "variables": vars_map,
                **extra
            })
        return out


# ---------------------------
# Public entry point (for tools_exec_node)
# ---------------------------

def recommend_tests_impl(
    df: pd.DataFrame,
    metadata: Dict[str, str],
    *,
    test_family: str,
    top_k: int = 5,
    # Optional knobs for later OSS+RAG swap:
    model_name: str = "gpt-4o-mini",
    dataset_hint: Optional[str] = None,
    missing_report: Optional[Dict[str, Any]] = None,  # <--- added (ignored)
) -> Dict[str, Any]:
    """
    Build feasible candidates and rank + phrase them.
    """
    # missing_report is accepted to keep a uniform tool-call signature; not used here.
    _ = missing_report

    prof = _basic_col_profile(df, metadata)

    family = test_family.lower()
    if family == "t_test":
        cands = _candidates_t_test(df, metadata, prof)
    elif family == "anova":
        cands = _candidates_anova(df, metadata, prof)
    elif family == "correlation":
        cands = _candidates_correlation(df, metadata, prof)
    elif family == "chi_square":
        cands = _candidates_chisq(df, metadata, prof)
    elif family == "clustering":
        cands = _candidates_clustering(df, metadata, prof)
    else:
        return {"error": f"Unknown test_family: {test_family}"}

    if not cands:
        return {
            "schema_version": SCHEMA_VERSION,
            "test_family": family,
            "recommendations": [],
            "items": [],
            "note": "No feasible candidates found under basic screening rules."
        }

    recs = _rank_with_llm(
        model_name=model_name,
        dataset_hint=dataset_hint,
        test_family=family,
        candidates=cands,
        top_k=top_k
    )

    # Normalize output structure
    normed: List[Dict[str, Any]] = []
    for r in recs:
        title = r.get("short_title") or r.get("title") or ""
        rationale = r.get("rationale") or ""
        variables = r.get("variables") or {}
        item: Dict[str, Any] = {
            "test_family": family,
            "short_title": title,
            "rationale": rationale,
            "variables": variables,
        }
        # pass through clustering suggested params if present
        if family == "clustering" and "suggested_params" in r:
            item["suggested_params"] = r["suggested_params"]
        normed.append(item)

    return {
        "schema_version": SCHEMA_VERSION,
        "test_family": family,
        "recommendations": normed,
        "items": normed,
        "candidates_considered": len(cands),
        "top_k": top_k,
    }
