import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI()



system_prompt = """
    You are a data science assistant.
    
    You receive:
    - A small sample of a dataset (as CSV rows).
    
    Your task:
    - Suggest possible **two-sample t-tests** that would make sense on this data.
    - For each t-test:
        - Identify the **grouping variable** (must be categorical).
        - Identify the **numeric outcome variable** (must be quantitative).
        - Write a **natural language question** that this t-test would answer.
    
    Return your suggestions as a JSON list where each item includes:
    - group_col (string)
    - value_col (string)
    - question (string)
    
    Do not include any markdown formatting, backticks, or explanations.
    Make your suggestions as interesting as possible, as the data you are receiving is likely from a thesis or academic related. (the variables being compared should have meaning)
    """


def format_messages(csv_sample: str) -> list:
    """
    Format the LLM messages including variable summary and a CSV sample.
    """
    user_prompt = (
        f"Here is a sample of the dataset:\n{csv_sample}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def suggest_t_tests(csv_sample: str) -> list:
    """
    Asks GPT-4o-mini to suggest t-tests for the given dataset.
    Returns a list of suggestions with group_col, value_col, and a natural language question.
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=format_messages(csv_sample)
    )

    raw_content = response.choices[0].message.content

    try:
        suggestions = json.loads(raw_content)
        return suggestions
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse LLM response as JSON:\n{raw_content}")



def clean_json_from_llm(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text



def parse_user_request_for_ttest(
    user_message: str,
    variable_summary: dict
    , csv_sample: str
) -> dict:
    """
    Uses GPT-4o to extract group_col and value_col for a t-test from a natural language user request.
    """
    system_prompt = """
    You are an assistant that receives:
    - A user question about a dataset
    - A list of variables in that dataset (with types)
    - A small CSV sample of the dataset
    
    Your task:
    - Identify if the user is asking for a valid two-sample t-test
    - If yes, extract:
        - group_col (must be categorical)
        - value_col (must be quantitative)
        - question: a natural language explanation of what this test will answer
        - group_filter (optional): a list of exactly two categories from the group_col if the user specifies them
    
    Only include group_filter if:
    - The group_col has more than two unique values
    - The user specifies which two categories to compare (e.g., "compare Multimodal and Systemic")
    
    Return a valid JSON object like:
    {
      "group_col": "gender",
      "value_col": "overall_survival_months",
      "question": "Do male and female patients have different survival outcomes?",
      "group_filter": ["Male", "Female"]
    }
    
    If no valid t-test can be inferred, return:
    {
      "group_col": null,
      "value_col": null,
      "question": "not applicable"
    }
    
    Do not include any markdown or explanations in your response.
    """


    user_prompt = f"""
    User request: {user_message}
    
    Here is the dataset variable summary:
    {json.dumps(variable_summary, indent=2)}
    
    Here is a sample of the dataset:
    {csv_sample}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    raw_content = response.choices[0].message.content
    cleaned = clean_json_from_llm(raw_content)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse response as JSON:\n{cleaned}")




anova_system_prompt = """
You are a data science assistant.

You receive:
- A small sample of a dataset (as CSV rows).

Your task:
- Suggest possible one-way ANOVA tests that would make sense on this data.
- For each ANOVA:
    - The grouping variable (categorical) must have **3 or more unique values**.
    - The outcome variable must be **quantitative**.
    - Write a **natural language question** that this ANOVA would answer.

Return your suggestions as a JSON list where each item includes:
- group_col (string)
- value_col (string)
- question (string)

Do not include markdown, backticks, or commentary.
Make your suggestions as interesting as possible, as the data you are receiving is likely from a thesis or academic related. (the variables being compared should have meaning)
"""


def suggest_anovas(csv_sample: str) -> list:
    """
    Asks GPT-4o-mini to suggest ANOVA tests for the given dataset.
    Returns a list of suggestions with group_col, value_col, and a natural language question.

    Args:
        csv_sample (str): A sample of the dataset in CSV format.

    Returns:
        list: A list of suggested ANOVA tests, where each item is a dictionary containing:
            - group_col (str): The categorical grouping variable.
            - value_col (str): The numeric dependent variable.
            - question (str): A natural language question describing the comparison.

    Raises:
        ValueError: If the LLM response cannot be parsed as valid JSON.
    """
    messages = [
        {"role": "system", "content": anova_system_prompt},
        {"role": "user", "content": f"Here is a sample of the dataset:\n{csv_sample}"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(clean_json_from_llm(raw))
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse ANOVA suggestions:\n{raw}")




def parse_user_request_for_anova(
    user_message: str,
    variable_summary: dict,
    csv_sample: str
) -> dict:
    """
    Uses GPT to determine whether an ANOVA test is requested and extract its parameters.
    """
    system_prompt = """
    You are an assistant that receives:
    - A user question about a dataset
    - A list of variables in that dataset (with types)
    - A small CSV sample of the dataset
    
    Your task:
    - Identify if the user is asking for a valid one-way ANOVA
    - If yes, extract:
        - group_col (must be categorical with 3+ levels)
        - value_col (must be quantitative)
        - question: a natural language explanation of what this test will answer
    
    Return a valid JSON object like:
    {
      "group_col": "treatment_type",
      "value_col": "overall_survival_months",
      "question": "Do survival times differ across treatment types?"
    }

    If the request is invalid for ANOVA, return:
    {
      "group_col": null,
      "value_col": null,
      "question": "not applicable"
    }

    Do not include markdown or commentary.
    """

    user_prompt = f"""
    User request: {user_message}

    Here is the dataset variable summary:
    {json.dumps(variable_summary, indent=2)}

    Here is a sample of the dataset:
    {csv_sample}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    raw_content = response.choices[0].message.content
    cleaned = clean_json_from_llm(raw_content)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse ANOVA response as JSON:\n{cleaned}")




def parse_user_request_for_tukey(
    user_message: str,
    variable_summary: dict,
    csv_sample: str
) -> dict:
    """
    Uses GPT to determine whether a Tukey HSD test is requested and extract its parameters.
    """

    system_prompt = """
    You are an assistant that receives:
    - A user request about a dataset
    - A list of dataset variables (with types: categorical or quantitative)
    - A small CSV sample from the dataset

    Your job:
    - Determine if the user is asking to run a Tukey HSD post-hoc test
    - If yes, extract:
        - group_col: a categorical variable with 2+ groups
        - value_col: a quantitative variable to compare
        - question: natural language explanation of what this test answers

    If the request is valid, return JSON like:
    {
      "group_col": "treatment_type",
      "value_col": "progression_free_survival_months",
      "question": "Does progression-free survival differ across treatment types?"
    }

    If it's not a valid Tukey request, return:
    {
      "group_col": null,
      "value_col": null,
      "question": "not applicable"
    }

    Do not include any explanation, only return JSON.
    """

    user_prompt = f"""
    User request: {user_message}

    Variable summary:
    {json.dumps(variable_summary, indent=2)}

    CSV sample:
    {csv_sample[:2000]}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    raw_content = response.choices[0].message.content
    cleaned = clean_json_from_llm(raw_content)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse Tukey response as JSON:\n{cleaned}")



def parse_user_request_for_chi_squared(
    user_message: str,
    variable_summary: dict,
    csv_sample: str
) -> dict:
    """
    Uses GPT to determine whether a Chi-squared test of independence is requested and extract its parameters.
    """
    system_prompt = """
    You are an assistant that receives:
    - A user question about a dataset
    - A list of variables in that dataset (with types)
    - A small CSV sample of the dataset

    Your task:
    - Determine if the user is requesting a valid Chi-squared test of independence
    - If yes, extract:
        - col1 (must be categorical)
        - col2 (must be categorical)
        - question: a natural language description of what the test is analyzing

    Return a valid JSON object like:
    {
      "col1": "gender",
      "col2": "smoking_history",
      "question": "Is there a relationship between gender and smoking history?"
    }

    If the request is invalid for a Chi-squared test, return:
    {
      "col1": null,
      "col2": null,
      "question": "not applicable"
    }

    Do not include markdown or commentary.
    """

    user_prompt = f"""
    User request: {user_message}

    Here is the dataset variable summary:
    {json.dumps(variable_summary, indent=2)}

    Here is a sample of the dataset:
    {csv_sample[:2000]}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    raw_content = response.choices[0].message.content
    cleaned = clean_json_from_llm(raw_content)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse Chi-squared response as JSON:\n{cleaned}")



def suggest_chi_squared_tests(csv_sample: str) -> list:
    """
    Asks GPT to suggest valid Chi-squared tests of independence from the dataset.
    Returns a list of test suggestions.

    Each suggestion includes:
        - col1: first categorical variable
        - col2: second categorical variable
        - question: natural language explanation of what the test investigates

    Args:
        csv_sample (str): Sample of the dataset in CSV format.

    Returns:
        list of dicts with keys: col1, col2, question

    Raises:
        ValueError if response can't be parsed as valid JSON.
    """

    system_prompt = """
    You are a data science assistant.
    Your task is to suggest valid Chi-squared tests of independence for a given dataset.

    Each test should involve two different categorical variables and answer a meaningful question about their relationship.

    Return a JSON list like:
    [
      {
        "col1": "gender",
        "col2": "smoking_history",
        "question": "Is there a relationship between gender and smoking history?"
      },
      ...
    ]

    Only suggest tests where both variables are clearly categorical based on the sample data.
    Do not include commentary, markdown, or explanations — only the raw JSON.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is a sample of the dataset:\n{csv_sample[:2000]}"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(clean_json_from_llm(raw))
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse Chi-squared suggestions:\n{raw}")



def parse_user_request_for_correlation(
    user_message: str,
    variable_summary: dict,
    csv_sample: str
) -> dict:
    """
    Uses GPT to determine whether a correlation test is requested and extract its parameters.
    """
    system_prompt = """
    You are an assistant that receives:
    - A user question about a dataset
    - A list of variables in that dataset (with types)
    - A small CSV sample of the dataset

    Your task:
    - Determine if the user is requesting a valid correlation test
    - If yes, extract:
        - col1 (must be quantitative)
        - col2 (must be quantitative)
        - method: "pearson" or "spearman" (default to pearson if not specified)
        - question: a natural language description of what the test is analyzing

    Return a valid JSON object like:
    {
      "col1": "age",
      "col2": "pack_years",
      "method": "pearson",
      "question": "Is there a linear correlation between age and pack years of smoking?"
    }

    If the request is invalid for a correlation test, return:
    {
      "col1": null,
      "col2": null,
      "method": null,
      "question": "not applicable"
    }

    Do not include markdown or commentary.
    """

    user_prompt = f"""
    User request: {user_message}

    Here is the dataset variable summary:
    {json.dumps(variable_summary, indent=2)}

    Here is a sample of the dataset:
    {csv_sample[:2000]}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    raw_content = response.choices[0].message.content
    cleaned = clean_json_from_llm(raw_content)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse correlation response as JSON:\n{cleaned}")



def suggest_correlation_tests(csv_sample: str) -> list:
    """
    Asks GPT to suggest valid correlation tests from the dataset.
    Returns a list of test suggestions.

    Each suggestion includes:
        - col1: first quantitative variable
        - col2: second quantitative variable
        - method: "pearson" (default) or "spearman" (if appropriate)
        - question: natural language explanation of what the test investigates

    Args:
        csv_sample (str): Sample of the dataset in CSV format.

    Returns:
        list of dicts with keys: col1, col2, method, question

    Raises:
        ValueError if response can't be parsed as valid JSON.
    """

    system_prompt = """
    You are a data science assistant.
    Your task is to suggest valid correlation tests between two quantitative variables in the dataset.

    Choose pairs of numeric variables and suggest whether Pearson or Spearman correlation is more appropriate based on their expected distribution or relationships. Default to Pearson unless the relationship is clearly non-linear or ordinal.

    Return a JSON list like:
    [
      {
        "col1": "age",
        "col2": "pack_years",
        "method": "pearson",
        "question": "Is there a linear correlation between age and smoking pack years?"
      },
      ...
    ]

    Only include valid numeric column pairs. Do not include commentary, markdown, or explanations — only the raw JSON.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is a sample of the dataset:\n{csv_sample[:2000]}"}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(clean_json_from_llm(raw))
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse correlation suggestions:\n{raw}")
