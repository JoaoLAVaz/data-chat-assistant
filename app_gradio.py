import os
import json
import gradio as gr
import pandas as pd

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent.graph import create_agent
from analysis.shared.metadata import extract_metadata, create_dataset_summary_message, get_dataset_info

# ---- silence joblib/loky core-detection warning on Windows ----
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    # set core count to a small safe number
    try:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
    except Exception:
        os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# ---- avoid MKL KMeans threading issue on Windows ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Compile the agent once at import time
AGENT = create_agent()

# ------------- Helpers -------------

def _init_agent_state(df: pd.DataFrame):
    """Build initial AgentState dict with dataset + summary message.

    Also seeds config to force the missing-data node to run in HYBRID mode.
    """
    metadata = extract_metadata(df)
    summary_msg = create_dataset_summary_message(metadata, df, n_rows=5)

    state = {
        "messages": [summary_msg],   # important: seed history so LLM sees the dataset
        "df": df,
        "metadata": metadata,
        "analysis_context": {},
        "config": {
            "missing": {
                # Force HYBRID scope (use whole dataset as predictors, impute only needed target)
                "scope": "hybrid",
                # Sensible defaults 
                "alpha": 0.05,
                "tiny_threshold": 0.05,     # <=5% missing -> listwise delete (kept for completeness)
                "impute_threshold": 0.20,   # <=20% -> impute; >20% -> delete with warning (current policy prefers imputing)
                "force_impute": False,      # set True to always impute when any missing
                "max_cat_cardinality": 50,  # cap for one-hot encoding
                "max_pred_missing": 0.50,   # drop predictors >50% missing
            }
        },
    }
    # Return markdown summary text for the UI and the internal state
    return summary_msg.content, state


def _get_last_ai_and_tool_ids(messages):
    """Return (last_ai_message, set_of_its_tool_call_ids) for the most recent AI that issued tool calls."""
    last_ai = None
    tool_ids = set()
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            last_ai = m
            for tc in m.tool_calls or []:
                tcid = tc.get("id")
                if tcid:
                    tool_ids.add(tcid)
            break
    return last_ai, tool_ids


def _find_plot_path_for_tool_ids(messages, tool_ids):
    """Find the last ToolMessage whose tool_call_id is in tool_ids and has a JSON payload
    with 'plot_path' or 'plot_paths' (list). Returns a single file path string."""
    if not tool_ids:
        return None
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None) in tool_ids:
            try:
                payload = json.loads(m.content)
                if not isinstance(payload, dict):
                    continue
                # Prefer explicit plot_path
                if payload.get("plot_path"):
                    return payload["plot_path"]
                # Fallback: first path from plot_paths list
                if isinstance(payload.get("plot_paths"), list) and payload["plot_paths"]:
                    return payload["plot_paths"][0]
            except Exception:
                pass
    return None


# ------------- Gradio Callbacks -------------

def load_csv(file):
    """Handle CSV upload: read file, create metadata + summary, seed agent state.
       UI shows only the compact 4-line summary."""
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return "There was an error processing the CSV file. Please try again.", None, [], None

    # Seed agent state (full summary goes to the LLM, not to the UI)
    _, state = _init_agent_state(df)

    # Build the compact UI summary (4 lines)
    info = get_dataset_info(df)
    ui_summary = (
        f"Total rows: {info['n_rows']:,}\n"
        f"Total columns: {info['n_columns']}\n"
        f"Missing values: {info['missing_data']['total_missing']:,}\n"
        f"Complete cases: {info['missing_data']['complete_cases']:,}"
    )

    # Reset chat history UI and last_plot_path
    return ui_summary, state, [], None


def respond(message, chat_history_display, agent_state, last_plot_path):
    """Main chat handler: append user msg, run agent, return assistant reply + optional plot."""
    if not agent_state or "df" not in agent_state:
        # OpenAI-style dict messages
        chat_history_display.append({"role": "user", "content": message})
        chat_history_display.append({"role": "assistant", "content": "Please upload a dataset first."})
        # Clear input, keep chat, hide image, keep textbox interactive, leave last_plot_path unchanged
        return "", chat_history_display, gr.update(visible=False), gr.update(interactive=True), last_plot_path

    # Optimistic UI: show "Thinking..."
    chat_history_display.append({"role": "user", "content": message})
    chat_history_display.append({"role": "assistant", "content": "Thinking..."})
    # Hide image while processing; lock textbox
    yield (
        "",
        chat_history_display,
        gr.update(visible=False),
        gr.update(interactive=False),
        last_plot_path,  # unchanged
    )

    # Append the human message to state and invoke the agent
    agent_state["messages"].append(HumanMessage(content=message))
    result_state = AGENT.invoke(agent_state)

    # Grab the final assistant text (the last AIMessage in the conversation)
    final_text = ""
    for m in reversed(result_state["messages"]):
        if isinstance(m, AIMessage):
            final_text = m.content or ""
            break

    # Find the most recent AI that issued tool calls, then find its matching ToolMessage plot
    _, tool_ids = _get_last_ai_and_tool_ids(result_state["messages"])
    plot_path = _find_plot_path_for_tool_ids(result_state["messages"], tool_ids)

    # Clean up previous plot file (if any)
    if last_plot_path and last_plot_path != plot_path:
        try:
            if os.path.exists(last_plot_path):
                os.remove(last_plot_path)
        except Exception as e:
            print(f"Failed to remove previous plot: {e}")

    # Replace the placeholder "Thinking..." with the real assistant text
    # (last item should be the assistant placeholder we just added)
    if chat_history_display and chat_history_display[-1].get("role") == "assistant":
        chat_history_display[-1] = {"role": "assistant", "content": final_text or "(No response)"}
    else:
        # Fallback, ensure we always add an assistant message
        chat_history_display.append({"role": "assistant", "content": final_text or "(No response)"})

    # Re-enable textbox; set image if available; update last_plot_path state
    yield (
        "",
        chat_history_display,
        gr.update(value=plot_path, visible=bool(plot_path and os.path.exists(plot_path))),
        gr.update(interactive=True),
        plot_path,
    )


# ------------- Gradio UI -------------

with gr.Blocks(title="LLM + Data Science Assistant") as demo:
    gr.Markdown("#  Data Science Chat Assistant")
    gr.Markdown(
        "Upload a CSV file, then ask questions about your dataset.\n\n"
        "**Supported analyses:**\n"
        "- T-tests\n"
        "- ANOVA (Welch / Kruskal–Wallis fallback)\n"
        "- Chi-squared / Fisher's Exact\n"
        "- Correlation (Pearson / Spearman)\n"
        "- Clustering (K-means with PCA visualization)\n"
    )

    with gr.Row():
        file_upload = gr.File(label="Upload CSV", file_types=[".csv"])
        summary_output = gr.Markdown()

    # Internal agent state + last plot path live here across turns
    agent_state = gr.State(value=None)
    last_plot_state = gr.State(value=None)

    # Chat widgets (switch to OpenAI-style messages)
    chatbot = gr.Chatbot(label="Chat with your dataset", type="messages")
    user_input = gr.Textbox(placeholder="Ask a question about your data...")
    plot_output = gr.Image(label="Generated Plot", visible=False)

    # Wire CSV upload -> load_csv (also resets chat and last plot)
    file_upload.change(
        fn=load_csv,
        inputs=file_upload,
        outputs=[summary_output, agent_state, chatbot, last_plot_state],
    )

    # Wire chat submit -> respond (streaming via generator)
    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot, agent_state, last_plot_state],
        outputs=[user_input, chatbot, plot_output, user_input, last_plot_state],
        queue=True,
    )

# Run the app
if __name__ == "__main__":
    demo.launch()
