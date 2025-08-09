import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from typing import Union, Dict, List
from scipy.stats import ttest_ind, f_oneway, chi2_contingency, pearsonr, spearmanr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os



def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate descriptive statistics for a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing summary statistics.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    return df.describe()



def calculate_correlation(
    df: pd.DataFrame,
    var1: str,
    var2: str
) -> pd.DataFrame:
    """
    Calculate the correlation matrix between two variables in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        var1 (str): The name of the first variable.
        var2 (str): The name of the second variable.

    Returns:
        pd.DataFrame: A 2x2 correlation matrix.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if var1 not in df.columns or var2 not in df.columns:
        raise ValueError(f"Columns '{var1}' and/or '{var2}' not found in DataFrame.")
    return df[[var1, var2]].corr()




def run_t_test(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    group_filter: List[str] = None
) -> Union[str, Dict[str, float]]:
    """
    Perform a two-sample t-test between two groups in a DataFrame.

    Args:
        df (pd.DataFrame): The dataset.
        group_col (str): The categorical column to group by.
        value_col (str): The quantitative column to test.
        group_filter (List[str], optional): Restrict the group_col to exactly two values.

    Returns:
        Dict with t-test results or error message.
    """
    if group_col not in df.columns or value_col not in df.columns:
        return f"Columns '{group_col}' or '{value_col}' not found."

    # Check that value_col is numeric
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return f"Cannot run t-test: '{value_col}' is not numeric."

    # Check that group_col is not numeric (must be categorical)
    if pd.api.types.is_numeric_dtype(df[group_col]):
        return f"Cannot run t-test: '{group_col}' should be a categorical variable, not numeric."

    # Apply filter if given
    if group_filter:
        df = df[df[group_col].isin(group_filter)]

    # Get unique groups
    groups = df[group_col].unique()

    if len(groups) != 2:
        return f"T-test requires exactly 2 groups, found {len(groups)}: {groups.tolist()}"

    group1 = df[df[group_col] == groups[0]][value_col]
    group2 = df[df[group_col] == groups[1]][value_col]

    stat, pval = ttest_ind(group1, group2)

    return {
        "group_col": group_col,
        "value_col": value_col,
        "group_filter": group_filter or groups.tolist(),
        "group_1": str(groups[0]),
        "group_2": str(groups[1]),
        "group_1_n": len(group1),
        "group_2_n": len(group2),
        "statistic": stat,
        "p_value": pval
    }




def run_anova(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    return_plot: bool = False
) -> dict:
    """
    Run one-way ANOVA and optionally generate a box plot.

    Args:
        df (pd.DataFrame): The dataset.
        group_col (str): The categorical column to group by.
        value_col (str): The numeric outcome variable.
        return_plot (bool): If True, also return a box plot as an image path.

    Returns:
        dict: ANOVA results, post-hoc Tukey test (if applicable), and optional plot metadata.
    """

    if group_col not in df.columns or value_col not in df.columns:
        return {"error": f"Columns '{group_col}' or '{value_col}' not found."}

    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return {"error": f"Column '{value_col}' must be numeric for ANOVA."}

    if not pd.api.types.is_categorical_dtype(df[group_col]):
        df[group_col] = df[group_col].astype("category")

    df_clean = df[[group_col, value_col]].dropna()

    grouped = [
        group[value_col].values
        for _, group in df_clean.groupby(group_col)
        if len(group) > 1
    ]

    if len(grouped) < 2:
        return {"error": "ANOVA requires at least 2 groups with more than 1 observation each."}

    stat, pval = f_oneway(*grouped)

    result = {
        "test": "anova",
        "group_col": group_col,
        "value_col": value_col,
        "statistic": stat,
        "p_value": pval,
        "post_hoc": None,
        "show_plot": False,
        "plot_type": "box",
        "plot_path": None
    }

    if pval < 0.05:
        try:
            tukey = pairwise_tukeyhsd(endog=df_clean[value_col], groups=df_clean[group_col])
            result["post_hoc"] = tukey.summary().as_text()
        except Exception as e:
            result["post_hoc"] = f"Post-hoc test failed: {str(e)}"

    if return_plot:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df_clean, x=group_col, y=value_col, ax=ax)
            ax.set_title(f"{value_col} by {group_col}")
            ax.set_xlabel(group_col)
            ax.set_ylabel(value_col)
            plt.xticks(rotation=45)

            tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            fig.savefig(tmpfile.name, bbox_inches="tight")
            plt.close(fig)

            result["show_plot"] = True
            result["plot_path"] = tmpfile.name
        except Exception as e:
            result["plot_error"] = f"Failed to generate box plot: {str(e)}"
            result["show_plot"] = False

    return result






def run_tukey_posthoc(
    df: pd.DataFrame,
    group_col: str,
    value_col: str
) -> dict:
    """
    Run Tukey HSD post-hoc test between groups of a categorical variable.

    Args:
        df (pd.DataFrame): The dataset.
        group_col (str): The categorical column.
        value_col (str): The numerical column to compare.

    Returns:
        dict: Results of Tukey test or error.
    """
    if group_col not in df.columns or value_col not in df.columns:
        return {"error": f"Columns '{group_col}' or '{value_col}' not found."}

    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return {"error": f"Column '{value_col}' must be numeric for Tukey HSD."}

    if not pd.api.types.is_categorical_dtype(df[group_col]):
        df[group_col] = df[group_col].astype("category")

    df_clean = df[[group_col, value_col]].dropna()

    try:
        tukey = pairwise_tukeyhsd(endog=df_clean[value_col], groups=df_clean[group_col])
        return {
            "group_col": group_col,
            "value_col": value_col,
            "post_hoc_summary": tukey.summary().as_text()
        }
    except Exception as e:
        return {"error": f"Tukey HSD failed: {str(e)}"}



def run_chi_squared_test(
    df,
    group_col,
    value_col,
    return_plot: bool = False
) -> dict:
    """
    Run a chi-squared test of independence between two categorical variables.
    Optionally generates a bar plot comparing group distributions.

    Args:
        df (pd.DataFrame): The dataset.
        group_col (str): One of the categorical variables
        value_col (str): The other categorical variable
        return_plot (bool): Whether to generate a bar plot (default: False)

    Returns:
        dict: Test result (and plot info if return_plot is True)
    """

    print("=== CHI-SQUARED TEST FUNCTION CALLED ===")
    print(f"Running chi-squared test for: {group_col} vs {value_col}")

    # Step 1: Filter missing values
    filtered = df[[group_col, value_col]].dropna()
    print(f"Filtered data:\n{filtered.head()}")

    # Step 2: Check category diversity
    if filtered[group_col].nunique() < 2 or filtered[value_col].nunique() < 2:
        print("Not enough unique values for Chi-squared test.")
        return {"error": f"Not enough category diversity in '{group_col}' or '{value_col}' to perform test."}

    # Step 3: Create contingency table
    contingency = pd.crosstab(filtered[group_col], filtered[value_col])
    print(f"Contingency table:\n{contingency}")

    if contingency.empty or contingency.values.sum() == 0:
        print("Empty or invalid contingency table.")
        return {"error": "Contingency table is empty or invalid."}

    # Step 4: Run test
    try:
        chi2, p, dof, expected = chi2_contingency(contingency)
        print(f"Chi-squared: {chi2}, p-value: {p}, dof: {dof}")

        result = {
            "test": "chi_squared",
            "group_col": group_col,
            "value_col": value_col,
            "chi2_statistic": chi2,
            "p_value": p,
            "degrees_of_freedom": dof,
            "expected_freq": expected.tolist(),
            "observed_freq": contingency.values.tolist(),
            "categories": {
                "row_labels": contingency.index.tolist(),
                "col_labels": contingency.columns.tolist()
            }
        }

        # Step 5: Optional plot
        if return_plot:
            plt.figure(figsize=(8, 5))
            contingency.div(contingency.sum(1), axis=0).plot(kind='bar', stacked=True)
            plt.title(f"Distribution of {value_col} by {group_col}")
            plt.ylabel("Proportion")
            plt.tight_layout()

            tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            plt.savefig(tmpfile.name)
            plt.close()

            result["show_plot"] = True
            result["plot_type"] = "bar"
            result["plot_path"] = tmpfile.name

        return result

    except Exception as e:
        print(f"Chi-squared test failed with exception: {e}")
        return {"error": f"Chi-squared test failed: {str(e)}"}




def run_correlation_test(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    method: str = "pearson",
    return_plot: bool = False
) -> dict:
    """
    Run a correlation test between two quantitative variables and optionally return a scatter plot.
    Args:
        df (pd.DataFrame): The dataset.
        col1 (str): First numeric column.
        col2 (str): Second numeric column.
        method (str): 'pearson' (default) or 'spearman'.
        return_plot (bool): Should it return a scater plot

    Returns:
        dict: Correlation coefficient, p-value, and number of observations.
    """

    method = method.lower()
    if method not in {"pearson", "spearman"}:
        return {"error": f"Invalid method '{method}'. Use 'pearson' or 'spearman'."}

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": f"Columns '{col1}' or '{col2}' not found."}

    df[col1] = pd.to_numeric(df[col1], errors="coerce")
    df[col2] = pd.to_numeric(df[col2], errors="coerce")
    df_clean = df[[col1, col2]].dropna()

    if df_clean.shape[0] < 3:
        return {"error": "Not enough data after dropping missing values (need at least 3 rows)."}

    corr_func = spearmanr if method == "spearman" else pearsonr

    try:
        corr, pval = corr_func(df_clean[col1], df_clean[col2])

        result = {
            "test": "correlation",
            "method": method,
            "col1": col1,
            "col2": col2,
            "correlation_coefficient": corr,
            "p_value": pval,
            "n_observations": len(df_clean)
        }

        if return_plot:
            fig, ax = plt.subplots()
            ax.scatter(df_clean[col1], df_clean[col2], alpha=0.6)
            ax.set_title(f"{method.title()} Correlation: {col1} vs {col2}")
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.savefig(tmpfile.name)
                result["show_plot"] = True
                result["plot_type"] = "scatter"
                result["plot_path"] = tmpfile.name

            plt.close(fig)

        return result

    except Exception as e:
        return {"error": f"Correlation test failed: {str(e)}"}




def summarize_variable_statistics(
    df: pd.DataFrame,
    variable_summary: list
) -> dict:
    """
    Computes summary statistics for each variable in the dataset,
    depending on whether it's categorical or quantitative.

    Args:
        df (pd.DataFrame): The dataset.
        variable_summary (list): List of {"name": str, "status": "categorical"/"quantitative"} dicts.

    Returns:
        dict: Summary statistics for each variable

    """
    results = []

    for var in variable_summary:
        name = var["name"]
        status = var["status"]

        if name not in df.columns:#LLM may hallucinate!
            continue

        col_data = df[name]
        stats = {
            "name": name,
            "type": status,
            "missing": int(col_data.isna().sum())
        }

        if status == "categorical":
            top_value = col_data.value_counts(dropna=False).head(1)
            if not top_value.empty:
                stats["top_value"] = {top_value.index[0]: int(top_value.iloc[0])}

        elif status == "quantitative":
            col_numeric = pd.to_numeric(col_data, errors="coerce")
            stats.update({
                "mean": col_numeric.mean(),
                "median": col_numeric.median(),
                "std": col_numeric.std(),
                "min": col_numeric.min(),
                "max": col_numeric.max(),
                "non_missing_count": int(col_numeric.notna().sum())
            })

        results.append(stats)

    return {"variable_summary": results}

