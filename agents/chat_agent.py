# agents/chat_agent.py

import json
import numpy as np
import pandas as pd
#import openai
from openai import OpenAI
from agents.analysis_planner_agent import (
    suggest_t_tests as suggest_t_test_func,
    parse_user_request_for_ttest as parse_user_ttest_func,
    suggest_anovas as suggest_anova_func,
    parse_user_request_for_anova as parse_user_anova_func,
    parse_user_request_for_tukey as parse_user_tukey_func,
    suggest_chi_squared_tests as suggest_chi_squared_func,
    parse_user_request_for_chi_squared as parse_user_chi_func,
    suggest_correlation_tests as suggest_correlation_func,
    parse_user_request_for_correlation as parse_user_correlation_func
)
from agents.result_interpreter_agent import (
    interpret_test_result,
    interpret_variable_summary
)
from utils.analysis_tools import (
    run_t_test as run_t_test_func,
    run_anova as run_anova_func,
    run_tukey_posthoc as run_tukey_posthoc_func,
    run_chi_squared_test as run_chi_squared_func,
    run_correlation_test as run_correlation_func,
    summarize_variable_statistics 
)
from agents.chat_agent_tools import tools

openai = OpenAI()


def safe_json(obj): # to json.dumps safely
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return str(obj)



def chat(
    message: str,
    history: list,
    df,
    dataset_summary: dict,
    csv_sample: str
):
    """
    Core function logic of user messages and system replies, calling tools as needed.
    """

    system_prompt = (
        "You are a data science assistant. The user has uploaded a dataset. "
        "You have access to tools for analysis. When a user asks a question about the data, "
        "you should prefer using the tools to determine columns, run analyses, or interpret results."
    )

    messages = [{"role": "system", "content": system_prompt}] + history

    messages += [
        {"role": "user", "content": f"Here is the dataset summary:\n{dataset_summary.get('explanation', '')}"},
        {"role": "user", "content": f"Here are the dataset variables:\n{json.dumps(dataset_summary.get('variables', []), indent=2)}"},
        {"role": "user", "content": f"Here is a sample of the CSV data:\n{csv_sample}"},
        {"role": "user", "content": message}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    choice = response.choices[0]
    print("[GPT finish reason]:", choice.finish_reason)

    plot_path = None

    if choice.finish_reason == "tool_calls":
        tool_calls = choice.message.tool_calls

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls
        })

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            print("[Tool Call]:", tool_name)
            print("[Arguments]:", args)

            try:
                if tool_name == "suggest_t_tests":
                    result = suggest_t_test_func(csv_sample=args["csv_sample"])

                elif tool_name == "parse_user_request_for_ttest":
                    test_info = parse_user_ttest_func(
                        user_message=args["user_message"],
                        variable_summary=args["variable_summary"],
                        csv_sample=args["csv_sample"]
                    )
                    result_raw = run_t_test_func(
                        df,
                        test_info["group_col"],
                        test_info["value_col"],
                        test_info.get("group_filter")
                    )
                    result = interpret_test_result(
                        "t-test",
                        dataset_summary,
                        test_info,
                        result_raw
                    )

                elif tool_name == "run_t_test":
                    result_raw = run_t_test_func(
                        df,
                        group_col=args["group_col"],
                        value_col=args["value_col"],
                        group_filter=args.get("group_filter")
                    )
                    result = interpret_test_result(
                        "t-test",
                        dataset_summary,
                        args,
                        result_raw
                    )

                elif tool_name == "suggest_anovas":
                    result = suggest_anova_func(csv_sample=args["csv_sample"])

                elif tool_name == "parse_user_request_for_anova":
                    test_info = parse_user_anova_func(
                        user_message=args["user_message"],
                        variable_summary=args["variable_summary"],
                        csv_sample=args["csv_sample"]
                    )
                    result_raw = run_anova_func(
                        df,
                        test_info["group_col"],
                        test_info["value_col"],
                        return_plot=True
                    )
                    plot_path = result.get("plot_path")
                    result = interpret_test_result(
                        "anova",
                        dataset_summary,
                        test_info,
                        result_raw
                    )

                elif tool_name == "run_anova":
                    result_raw = run_anova_func(
                        df,
                        args["group_col"],
                        args["value_col"],
                        return_plot=True
                    )
                    plot_path = result.get("plot_path")
                    result = interpret_test_result(
                        "anova",
                        dataset_summary,
                        args,
                        result_raw
                     )

                elif tool_name == "parse_user_request_for_tukey":
                    test_info = parse_user_tukey_func(
                        user_message=args["user_message"],
                        variable_summary=args["variable_summary"],
                        csv_sample=args["csv_sample"]
                    )
                    result_raw = run_tukey_posthoc_func(
                        df,
                        test_info["group_col"],
                        test_info["value_col"]
                    )
                    result = interpret_test_result(
                        "tukey",
                        dataset_summary,
                        test_info,
                        result_raw
                    )

                elif tool_name == "run_tukey_posthoc":
                    result_raw = run_tukey_posthoc_func(
                        df,
                        args["group_col"],
                        args["value_col"]
                    )
                    result = interpret_test_result(
                        "tukey",
                        dataset_summary,
                        args,
                        result_raw
                    )

                elif tool_name == "suggest_chi_squared_tests":
                    result = suggest_chi_squared_func(csv_sample=args["csv_sample"])

                elif tool_name == "parse_user_request_for_chi_squared":
                    test_info = parse_user_chi_func(
                        user_message=args["user_message"],
                        variable_summary=args["variable_summary"],
                        csv_sample=args["csv_sample"]
                    )
                    result_raw = run_chi_squared_func(
                        df, 
                        group_col=test_info["group_col"],
                        value_col=test_info["value_col"],
                        return_plot = True
                    )
                    plot_path = result.get("plot_path")
                    result = interpret_test_result(
                        "chi-squared",
                        dataset_summary,
                        test_info,
                        result_raw
                    )

                elif tool_name == "run_chi_squared_test":
                    result_raw = run_chi_squared_func(
                        df, 
                        group_col=args["group_col"], 
                        value_col=args["value_col"],
                        return_plot = True
                    )
                    plot_path = result.get("plot_path")
                    result = interpret_test_result(
                        "chi-squared",
                        dataset_summary,
                        args
                        result_raw
                    )

                elif tool_name == "suggest_correlation_tests":
                    result = suggest_correlation_func(csv_sample=args["csv_sample"])

                elif tool_name == "parse_user_request_for_correlation":
                    test_info = parse_user_correlation_func(
                        user_message=args["user_message"],
                        variable_summary=args["variable_summary"],
                        csv_sample=args["csv_sample"]
                    )
                    result_raw = run_correlation_func(
                        df, 
                        test_info["col1"], 
                        test_info["col2"],
                        method=test_info.get("method", "pearson"),
                        return_plot = True
                    )
                    plot_path = result.get("plot_path")
                    result = interpret_test_result(
                        "correlation",
                        dataset_summary,
                        test_info,
                        result_raw
                    )

                elif tool_name == "run_correlation_test":
                    result_raw = run_correlation_func(
                        df, 
                        col1=args["col1"], 
                        col2=args["col2"], 
                        method=args.get("method", "pearson"),
                        return_plot = True
                    )
                    plot_path = result.get("plot_path")
                    result = interpret_test_result(
                        "correlation",
                        dataset_summary,
                        args,
                        result_raw
                    )

                elif tool_name == "interpret_result":
                    result = interpret_test_result(
                        test_type=args["test_type"],
                        dataset_summary=dataset_summary,
                        test_info=args["test_info"],
                        result=args["result"]
                    )

                elif tool_name == "summarize_variable_statistics":
                    result_summary = summarize_variable_statistics(
                        df=df,
                        variable_summary = dataset_summary["variables"]
                    )
                    #print(result_summary)
                    result = interpret_variable_summary(result_summary)
                    #print(result)
 

                else:
                    result = {"error": f"[\u26a0\ufe0f Unknown tool: {tool_name}]"}

            except Exception as e:
                result = {"error": f"\u274c Tool execution failed: {str(e)}"}

    
            #print(result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                #"content": json.dumps(result)
                "content": json.dumps(result, default=safe_json)
            })

        #try to stop the rendering of the img on the chat
        system_prompt_addicional = (
        "You are an AI assistant embedded in a data science app. "
        "You may receive results from tools, including statistical tests "
        "If a plot was generated, DO NOT include markdown or image links — just say a plot is available. "
        "Focus on explaining the test result clearly and concisely for a technical user. "
        )
        messages.insert(0, {
            "role":"system",
            "content": system_prompt_addicional
        })
        
        followup = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        final_reply = followup.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": final_reply})

        # Build structured response for Gradio
        response_payload = {
        "text": final_reply
        }
        if plot_path:
            response_payload["show_plot"] = True
            response_payload["plot_type"] = "box"
            response_payload["plot_path"] = plot_path

        #return json.dumps(response_payload), messages
        return response_payload, messages


    assistant_reply = choice.message.content.strip()
    messages.append({"role": "assistant", "content": assistant_reply})
    return {"text": assistant_reply}, messages
    