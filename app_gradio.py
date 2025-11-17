import os
import json
import gradio as gr
import pandas as pd

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent.graph import create_agent
from analysis.shared.metadata import extract_metadata, create_dataset_summary_message, get_dataset_info

# ---- silence joblib/loky core-detection warning on Windows ----
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    try:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
    except Exception:
        os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# ---- avoid MKL KMeans threading issue on Windows ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ----------------------------------------------------------------
# Sample datasets registry
# ----------------------------------------------------------------
SAMPLE_DATASETS = {
    # label shown in dropdown                 # relative path in repo
    "Lung cancer (toy) with missing values": "datasets/lung_cancer_sample_missingvals_alot.csv",
}

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
                "scope": "hybrid",
                "alpha": 0.05,
                "tiny_threshold": 0.05,
                "impute_threshold": 0.20,
                "force_impute": False,
                "max_cat_cardinality": 50,
                "max_pred_missing": 0.50,
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


def _ui_summary_text(df: pd.DataFrame) -> str:
    info = get_dataset_info(df)
    return (
        f"**Rows:** {info['n_rows']:,}  \n"
        f"**Columns:** {info['n_columns']}  \n"
        f"**Missing values:** {info['missing_data']['total_missing']:,}  \n"
        f"**Complete cases:** {info['missing_data']['complete_cases']:,}"
    )


# ------------- Gradio Callbacks -------------


def load_csv(file):
    """Handle CSV upload: read file, create metadata + summary, seed agent state.
       UI shows only the compact summary and a 5-row preview."""
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return "There was an error processing the CSV file. Please try again.", None, [], None, None

    _, state = _init_agent_state(df)
    ui_summary = _ui_summary_text(df)
    head_preview = df.head(5)

    # Reset chat history UI and last_plot_path
    return ui_summary, state, [], None, head_preview


def load_sample(selected_label, chat_history_display, last_plot_path):
    """Load one of the bundled sample datasets by label."""
    path = SAMPLE_DATASETS.get(selected_label)
    if not path:
        return (
            "Sample not found.",
            None,
            chat_history_display,
            gr.update(visible=False),
            last_plot_path,
            None,
        )

    if not os.path.exists(path):
        msg = (
            f"Sample file not found at `{path}`.\n\n"
            "If you're running on Hugging Face Spaces, make sure this file is committed in the repo."
        )
        return msg, None, chat_history_display, gr.update(visible=False), last_plot_path, None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error loading sample CSV: {e}")
        return (
            "There was an error loading the sample dataset.",
            None,
            chat_history_display,
            gr.update(visible=False),
            last_plot_path,
            None,
        )

    _, state = _init_agent_state(df)
    ui_summary = "**Loaded sample:** " + selected_label + "  \n" + _ui_summary_text(df)
    head_preview = df.head(5)

    # Reset chat & last plot when switching dataset
    return ui_summary, state, [], gr.update(visible=False), None, head_preview


def respond(message, chat_history_display, agent_state, last_plot_path):
    """Main chat handler: append user msg, run agent, return assistant reply + optional plot.

    IMPORTANT: returns updated agent_state so LangGraph history is preserved across turns.
    """
    # No dataset yet
    if not agent_state or "df" not in agent_state:
        chat_history_display.append({"role": "user", "content": message})
        chat_history_display.append(
            {"role": "assistant", "content": "Please upload a dataset or load a sample first."}
        )
        return (
            "",
            chat_history_display,
            gr.update(visible=False),
            gr.update(interactive=True),
            last_plot_path,
            agent_state,  # unchanged
        )

    # Optimistic UI
    chat_history_display.append({"role": "user", "content": message})
    chat_history_display.append({"role": "assistant", "content": "Thinking..."})

    # Hide image while processing; lock textbox
    yield (
        "",
        chat_history_display,
        gr.update(visible=False),
        gr.update(interactive=False),
        last_plot_path,
        agent_state,  # current state while tools/LLM run
    )

    # Append user msg into graph state and invoke agent
    agent_state["messages"].append(HumanMessage(content=message))
    prev_len = len(agent_state.get("messages", []))
    result_state = AGENT.invoke(agent_state)
    new_len = len(result_state.get("messages", []))

    # Debug: what tools were just called?
    last_ai, tool_ids = _get_last_ai_and_tool_ids(result_state["messages"])
    tool_names = []
    if last_ai and getattr(last_ai, "tool_calls", None):
        tool_names = [tc.get("name") for tc in (last_ai.tool_calls or [])]
    print(
        f"[DEBUG] respond: msgs_in={prev_len}, msgs_out={new_len}, "
        f"last_ai_tools={tool_names}"
    )

    # Persist updated agent_state
    agent_state = result_state

    # Grab final assistant text
    final_text = ""
    for m in reversed(result_state["messages"]):
        if isinstance(m, AIMessage):
            final_text = m.content or ""
            break

    # Find plot for the most recent tool-call AI
    _, tool_ids = _get_last_ai_and_tool_ids(result_state["messages"])
    plot_path = _find_plot_path_for_tool_ids(result_state["messages"], tool_ids)

    # Clean up previous plot file (if any)
    if last_plot_path and last_plot_path != plot_path:
        try:
            if os.path.exists(last_plot_path):
                os.remove(last_plot_path)
        except Exception as e:
            print(f"Failed to remove previous plot: {e}")

    # Replace "Thinking..." with final answer
    if chat_history_display and chat_history_display[-1].get("role") == "assistant":
        chat_history_display[-1] = {"role": "assistant", "content": final_text or "(No response)"}
    else:
        chat_history_display.append({"role": "assistant", "content": final_text or "(No response)"})

    # Yield final UI state + updated agent_state
    yield (
        "",
        chat_history_display,
        gr.update(value=plot_path, visible=bool(plot_path and os.path.exists(plot_path))),
        gr.update(interactive=True),
        plot_path,
        agent_state,
    )


# ------------- Gradio UI -------------


with gr.Blocks(title="LLM + Data Science Assistant") as demo:
    gr.Markdown("#  Data Science Chat Assistant")
    gr.Markdown(
        "Upload a CSV file or load a sample, then ask questions about your dataset.\n\n"
        "**Supported analyses:**\n"
        "- T-tests\n"
        "- ANOVA (Welch / Kruskal–Wallis fallback)\n"
        "- Chi-squared / Fisher's Exact\n"
        "- Correlation (Pearson / Spearman)\n"
        "- Clustering (K-means with PCA visualization)\n"
    )

    with gr.Row():
        with gr.Column():
            file_upload = gr.File(label="Upload your CSV", file_types=[".csv"])

            sample_label = gr.Dropdown(
                choices=list(SAMPLE_DATASETS.keys()),
                value=list(SAMPLE_DATASETS.keys())[0] if SAMPLE_DATASETS else None,
                label="Or pick a sample dataset",
                interactive=True,
            )
            load_sample_btn = gr.Button("Use sample dataset", variant="primary")

        with gr.Column():
            summary_output = gr.Markdown()
            preview_table = gr.Dataframe(
                interactive=False,
                wrap=True,
                visible=True,
                label="Preview (first 5 rows)",
            )

    # Internal agent state + last plot path live here across turns
    agent_state = gr.State(value=None)
    last_plot_state = gr.State(value=None)

    # Chat widgets (OpenAI-style dicts)
    chatbot = gr.Chatbot(label="Chat with your dataset", type="messages")
    user_input = gr.Textbox(placeholder="Ask a question about your data...")
    plot_output = gr.Image(label="Generated Plot", visible=False)

    # CSV upload -> load_csv
    file_upload.change(
        fn=load_csv,
        inputs=file_upload,
        outputs=[summary_output, agent_state, chatbot, last_plot_state, preview_table],
    )

    # "Use sample dataset" -> load_sample
    load_sample_btn.click(
        fn=load_sample,
        inputs=[sample_label, chatbot, last_plot_state],
        outputs=[summary_output, agent_state, chatbot, plot_output, last_plot_state, preview_table],
    )

    # Chat submit -> respond (streaming via generator)
    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot, agent_state, last_plot_state],
        outputs=[user_input, chatbot, plot_output, user_input, last_plot_state, agent_state],
        queue=True,
    )

# Run the app
if __name__ == "__main__":
    # For local dev can set share=True, but on HF Spaces it's not needed.
    demo.launch()
