"""
Tool Schema Definitions

This module defines the tool schemas that the LLM can see and call.
These are the "interfaces" for statistical analysis tools - the actual
implementations are handled in the analysis/ directory.

The LLM sees these tool schemas to understand what statistical analyses
are available and what parameters they need. Only high-level statistical
tests are exposed; diagnostic checks are handled internally within each test.
"""

from typing import List, Optional, Literal
from langchain_core.tools import tool, BaseTool


@tool
def recommend_tests(
    test_family: Literal["t_test", "anova", "correlation", "chi_square", "clustering"],
    top_k: int = 5
) -> str:
    """
    Recommend interesting AND feasible tests for a given family.

    This is a schema-only tool (the body doesn't run). The actual logic runs in a
    state-aware node so it can access df/metadata and (later) an OSS LLM + RAG.

    Args:
        test_family: One of "t_test", "anova", "correlation", "chi_square", "clustering".
        top_k: Maximum number of recommendations to return (default 5).

    Returns:
        JSON string with a list of recommendations. Each item will typically include:
          - test_family
          - variables (e.g., {"group_col": "...", "value_col": "..."} or {"features": [...]})
          - short_title
          - rationale
          - feasibility_notes (optional)
    """
    return "OK"  # schema only


@tool
def t_test(
    group_col: str,
    value_col: str,
    equal_var: bool = False,
    group_a: Optional[str] = None,
    group_b: Optional[str] = None
) -> str:
    """
    Perform a two-sample t-test between two groups in a DataFrame.
    
    This tool will automatically handle all statistical best practices:
    - Check for missing data and handle appropriately
    - Test normality assumptions (Shapiro-Wilk)
    - Test variance equality (Levene's test) 
    - Choose appropriate test variant (Student's vs Welch's vs Mann-Whitney U)
    - Calculate effect sizes and provide interpretations
    
    Args:
        group_col (str): The categorical column to group by (must have exactly 2 groups)
        value_col (str): The numerical column containing values to compare
        equal_var (bool): Force equal variance assumption (default: False, let algorithm decide)
        group_a (str, optional): Name of the first group level to include.
        group_b (str, optional): Name of the second group level to include.
    
    Returns:
        JSON string with comprehensive test results including effect sizes and interpretations
    """
    return "OK"  # Body never executes - this is just for LLM schema


@tool
def anova_test(group_col: str, value_col: str) -> str:
    """
    Perform one-way ANOVA to compare means across multiple groups.
    
    This tool will automatically handle all statistical best practices:
    - Check for missing data and handle appropriately
    - Test normality of residuals (Shapiro-Wilk)
    - Test homogeneity of variances (Levene's test)
    - Choose appropriate test (parametric ANOVA vs Kruskal-Wallis)
    - Perform post-hoc tests if significant (Tukey HSD)
    - Calculate effect sizes and provide interpretations
    
    Args:
        group_col (str): The categorical column defining groups (must have 3+ groups)
        value_col (str): The numerical column containing values to compare
        
    Returns:
        JSON string with ANOVA results, post-hoc tests, and effect sizes
    """
    return "OK"


@tool
def correlation_test(var1: str, var2: str, method: str = "auto") -> str:
    """
    Test correlation between two numerical variables.
    
    This tool will automatically handle all statistical best practices:
    - Check for missing data and handle appropriately
    - Test normality assumptions
    - Choose appropriate correlation method (Pearson vs Spearman)
    - Test significance and provide confidence intervals
    - Calculate effect sizes and provide interpretations
    
    Args:
        var1 (str): First numerical column
        var2 (str): Second numerical column
        method (str): Correlation method - "auto" (default), "pearson", or "spearman"
        
    Returns:
        JSON string with correlation coefficient, significance test, and interpretation
    """
    return "OK"


@tool
def chi_square_test(var1: str, var2: str) -> str:
    """
    Test independence between two categorical variables using chi-square test.
    
    This tool will automatically handle all statistical best practices:
    - Check for missing data and handle appropriately
    - Validate expected frequencies (>5 rule)
    - Choose appropriate test variant if needed (Fisher's exact for small samples)
    - Calculate effect sizes (Cramér's V)
    - Provide contingency table analysis
    
    Args:
        var1 (str): First categorical column
        var2 (str): Second categorical column
        
    Returns:
        JSON string with chi-square results, contingency table, and effect sizes
    """
    return "OK"


# ----------------------------
# NEW: Clustering (KMeans + PCA)
# ----------------------------
@tool
def clustering_kmeans(
    features: Optional[List[str]] = None,
    include_categoricals: bool = False,
    max_cat_cardinality: int = 8,
    n_clusters: str = "auto",
    k_min: int = 2,
    k_max: int = 8
) -> str:
    """
    Cluster rows with K-Means and return an evaluation summary plus a PCA(2D) scatter plot.

    Notes for the planner:
    - If the user lists variables (e.g., "age, pack_years, performance_status"), pass them via `features`.
    - If `features` is null, the implementation will auto-select all numerical columns.
    - If `include_categoricals=True`, low-cardinality categoricals (<= max_cat_cardinality) are one-hot encoded.
    - `n_clusters` can be an integer encoded as a string (e.g., "3") or "auto" to pick k via silhouette between k_min..k_max.

    Args:
        features: Optional list of column names to use. If None, use all numeric columns.
        include_categoricals: Whether to include low-cardinality categorical variables via one-hot.
        max_cat_cardinality: Max categories allowed to one-hot when include_categoricals=True.
        n_clusters: "auto" (default) to select k via silhouette, or a stringified integer like "3".
        k_min: Minimum k when auto-selecting.
        k_max: Maximum k when auto-selecting.

    Returns:
        JSON string with clustering diagnostics (chosen k, silhouette, cluster sizes, feature summary)
        and a temp file path to a PCA(2D) scatter plot.
    """
    return "OK"


ALL_TOOLS = [
    t_test,
    anova_test,
    correlation_test,
    chi_square_test,
    recommend_tests,
    clustering_kmeans,  
]


def get_all_tools() -> List[BaseTool]:
    """
    Get all available statistical analysis tools.
    
    Returns:
        List of all tool schemas for binding to LLM
    """
    return ALL_TOOLS
