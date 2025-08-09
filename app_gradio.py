import gradio as gr
import pandas as pd
import json

from agents.data_agent import summarize_csv_structure
from agents.chat_agent import chat

# Global state across interactions
chat_history = []
dataset_df = None
dataset_summary = None
csv_sample_str = None


def load_csv(file):
    global dataset_df, dataset_summary, csv_sample_str, chat_history

    try:
        # Load and sample CSV
        dataset_df = pd.read_csv(file.name)
        csv_sample_str = dataset_df.head(30).to_csv(index=False)
    
        # Generate summary
        dataset_summary = summarize_csv_structure(csv_sample_str)
    
        # Reset chat history
        chat_history = []

        # Show summary message
        return f" Dataset loaded! {len(dataset_df)} rows.\n\n**Summary:** {dataset_summary['explanation']}"
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return "There was an error processing the CSV file. Please try again."



def chat_with_data(user_input):
    global chat_history, dataset_df, dataset_summary, csv_sample_str

    if dataset_df is None:
        return " Please upload a dataset first.", None

    response_dict, chat_history = chat(
        message=user_input,
        history=chat_history,
        df=dataset_df,
        dataset_summary=dataset_summary,
        csv_sample=csv_sample_str
    )

    response_text = response_dict.get("text", str(response_dict))
    plot_path = response_dict.get("plot_path")

    return response_text, plot_path



def respond(
    message,
    chat_history_display):
    # Show user message immediately
    chat_history_display.append((message, " Thinking..."))
    # Disable input while processing
    yield (
        "", 
        chat_history_display, 
        gr.update(visible=False),  # Hide image
        gr.update(interactive=False)  # Disable textbox
    )

    # Assistant reply
    response_text, plot_path = chat_with_data(message)
    visible = bool(plot_path)

    chat_history_display[-1] = (message, response_text)

    # Re-enable textbox
    yield (
        "", 
        chat_history_display, 
        gr.update(value=plot_path, visible=visible),
        gr.update(interactive=True)  # Re-enable textbox
    )



# Gradio UI
with gr.Blocks(title="LLM + Data Science Assistant") as demo:
    gr.Markdown("#  Data Science Chat Assistant")
    gr.Markdown(
        "Upload a CSV file, then ask questions about your dataset.\n\n"
        "**Supported analyses:**\n- T-tests\n- ANOVA + Tukey\n- Chi-squared\n- Correlation\n"
    )

    with gr.Row():
        file_upload = gr.File(label="Upload CSV", file_types=[".csv"])
        summary_output = gr.Markdown()

    # Show summary after CSV upload
    file_upload.change(load_csv, inputs=file_upload, outputs=summary_output)

    chatbot = gr.Chatbot(label="Chat with your dataset")
    user_input = gr.Textbox(placeholder="Ask a question about your data...")
    plot_output = gr.Image(label="Generated Plot", visible=False)

    # Respond to message and show plot if returned
    user_input.submit(
        respond,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot, plot_output, user_input],
        queue=True
    )

# Run the app
if __name__ == "__main__":
    demo.launch()