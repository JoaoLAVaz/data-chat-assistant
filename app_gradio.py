import os
import json
import gradio as gr
import pandas as pd

from langchain_core.messages import HumanMessage
from agent.graph import create_agent
from analysis.shared.metadata import extract_metadata, create_dataset_summary_message, get_dataset_info

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
                # Sensible defaults you can tweak later
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
        chat_history_display.append((message, "Please upload a dataset first."))
        # Clear input, keep chat, hide image, keep textbox interactive, leave last_plot_path unchanged
        return "", chat_history_display, gr.update(visible=False), gr.update(interactive=True), last_plot_path

    # Optimistic UI: show "Thinking..."
    chat_history_display.append((message, "Thinking..."))
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

    # Extract final AI text (LLM interpretation after tools)
    final_text = ""
    for m in reversed(result_state["messages"]):
        if m.__class__.__name__ == "AIMessage":
            final_text = m.content or ""
            break

    # Try to find a plot_path from the last ToolMessage JSON payload
    plot_path = None
    for m in reversed(result_state["messages"]):
        if m.__class__.__name__ == "ToolMessage":
            try:
                payload = json.loads(m.content)
                if isinstance(payload, dict) and "plot_path" in payload and payload["plot_path"]:
                    plot_path = payload["plot_path"]
                    break
            except Exception:
                # ignore non-JSON tool outputs
                pass

    # Clean up previous plot file (if any)
    if last_plot_path and last_plot_path != plot_path:
        try:
            if os.path.exists(last_plot_path):
                os.remove(last_plot_path)
        except Exception as e:
            print(f"Failed to remove previous plot: {e}")

    # Update chat bubble
    chat_history_display[-1] = (message, final_text or "(No response)")

    # Re-enable textbox; set image if available; update last_plot_path state
    yield (
        "",
        chat_history_display,
        gr.update(value=plot_path, visible=bool(plot_path)),
        gr.update(interactive=True),
        plot_path,
    )


# ------------- Gradio UI -------------

with gr.Blocks(title="LLM + Data Science Assistant") as demo:
    gr.Markdown("#  Data Science Chat Assistant")
    gr.Markdown(
        "Upload a CSV file, then ask questions about your dataset.\n\n"
        "**Supported analyses (roadmap):**\n"
        "- T-tests (implemented)\n- ANOVA + Tukey (soon)\n- Chi-squared (soon)\n- Correlation (soon)\n"
    )

    with gr.Row():
        file_upload = gr.File(label="Upload CSV", file_types=[".csv"])
        summary_output = gr.Markdown()

    # Internal agent state + last plot path live here across turns
    agent_state = gr.State(value=None)
    last_plot_state = gr.State(value=None)

    # Chat widgets
    chatbot = gr.Chatbot(label="Chat with your dataset")
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
