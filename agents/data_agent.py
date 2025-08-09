import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

#example on how to call: lung_cancer = pd.read_csv('datasets/lung_cancer_sample.csv')
#                        csv_head_string = lung_cancer.head().to_csv(index=False)
#                        from agents.data_agent import summarize_csv_structure
#                        summary_2 = summarize_csv_structure(csv_head_string)

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI()

# System prompt for structure + instructions
system_prompt = """
You are assistant that receives a sample of a csv file and you summarize what the csv is about.
You should be able to tell if a variable is categorical or quantitative, and provide an explanation of what the dataset is about.

Your response should be in a json object with 3 categories:
- variables : a list of dictionaries, each with the name of a variable and its status ("categorical" or "quantitative")
- explanation : a natural language summary of what the dataset is about
- potential_analyses : a list of suggested statistical or data science analyses appropriate for this dataset

Do not use any markdown.
"""


def create_user_prompt(csv_sample: str) -> str:
    return f"Here is a sample of a csv, what can you tell me about it: {csv_sample}"


def format_messages(csv_sample: str) -> list:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": create_user_prompt(csv_sample)}
    ]


def clean_json_response(response_str: str) -> str:
    """
    Cleans model output by removing ```json blocks, stray newlines, and extra backslashes.
    """
    # Remove Markdown triple backticks and optional "json" hint
    cleaned = re.sub(r"^```(?:json)?\s*", "", response_str.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # Replace common escaped newlines (\\n) with actual newlines if needed
    cleaned = cleaned.replace("\\n", "\n").replace('\\"', '"')

    return cleaned


def summarize_csv_structure(csv_sample: str) -> dict:
    """
    Analyzes the structure and content of a CSV sample and returns a structured summary. Gets called on the app.
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=format_messages(csv_sample)
    )

    raw_content = response.choices[0].message.content
    cleaned_content = clean_json_response(raw_content)
    #return cleaned_content

    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        raise ValueError("Could not parse response as JSON:\n" + cleaned_content)
