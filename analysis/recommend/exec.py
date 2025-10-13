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
    """
    if not candidates:
        return []

    # Compact candidate summary sent to the model
    family = test_family.lower()
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
        else:  # chi_square
            brief_items.append({
                "var1": c["var1"], "var2": c["var2"],
                "levels_var1": c.get("levels_var1", []),
                "levels_var2": c.get("levels_var2", []),
                "n_pairs": c.get("n_pairs"),
            })

    # Use OpenAI (or whatever your environment provides). Keep isolated.
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        sys = SystemMessage(content=(
            "You are a biostatistics assistant. You will receive a dataset topic and a list of "
            "feasible candidate analyses for a specified test family. Rank the candidates by potential "
            "scientific interest and clarity of interpretation for exploratory analysis. Prefer clinically "
            "meaningful outcomes (survival, response), established prognostic features (stage, histology, "
            "treatment), and avoid trivial/demographic-only analyses unless none other exist. "
            "Return top K as JSON with fields: short_title, rationale (<=25 words), and the variable mapping. "
            "Do not invent variables not present."
        ))
        user = HumanMessage(content=(
            f"Dataset topic: {dataset_hint or 'N/A'}\n"
            f"Test family: {test_family}\n"
            f"Top K: {top_k}\n"
            f"Candidates (JSON): {brief_items}"
        ))

        llm = ChatOpenAI(model=model_name, temperature=0.2, max_tokens=600)
        resp = llm.invoke([sys, user])
        text = resp.content or ""

        # Try to parse JSON-like list. Be forgiving.
        import json, re
        m = re.search(r"\[.*\]", text, flags=re.S)
        parsed = json.loads(m.group(0)) if m else json.loads(text)
        # Ensure list and clamp to top_k
        recs = parsed if isinstance(parsed, list) else []
        return recs[:top_k]
    except Exception:
        # Fallback: crude lexical ranking and simple phrasing
        def score_item(it: Dict[str, Any]) -> int:
            name_blob = " ".join(map(str, it.values())).lower()
            # Boost common clinical outcomes & predictors
            boosts = ["survival", "response", "treatment", "stage", "histology", "progression"]
            return sum(word in name_blob for word in boosts) + int(it.get("n_pairs", 0) // 50)

        ranked = sorted(brief_items, key=score_item, reverse=True)[:top_k]
        out = []
        for it in ranked:
            if family in ("t_test", "anova"):
                title = f"{it['group_col']} → {it['value_col']}"
                vars_map = {"group_col": it["group_col"], "value_col": it["value_col"]}
            elif family == "correlation":
                title = f"{it['var1']} ↔ {it['var2']}"
                vars_map = {"var1": it["var1"], "var2": it["var2"]}
            else:
                title = f"{it['var1']} × {it['var2']}"
                vars_map = {"var1": it["var1"], "var2": it["var2"]}
            out.append({
                "short_title": title,
                "rationale": "Likely clinically meaningful; adequate sample size.",
                "variables": vars_map,
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
) -> Dict[str, Any]:
    """
    Build feasible candidates and rank + phrase them.
    """
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
    else:
        return {"error": f"Unknown test_family: {test_family}"}

    if not cands:
        return {"recommendations": [], "note": "No feasible candidates found under basic screening rules."}

    recs = _rank_with_llm(model_name=model_name, dataset_hint=dataset_hint,
                          test_family=family, candidates=cands, top_k=top_k)

    # Normalize output structure
    normed: List[Dict[str, Any]] = []
    for r in recs:
        title = r.get("short_title") or r.get("title") or ""
        rationale = r.get("rationale") or ""
        variables = r.get("variables") or {}
        normed.append({
            "test_family": family,
            "short_title": title,
            "rationale": rationale,
            "variables": variables,
        })

    return {
        "test_family": family,
        "recommendations": normed,
        "candidates_considered": len(cands),
        "top_k": top_k,
    }
