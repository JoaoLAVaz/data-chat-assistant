# data-chat-assistant
Interactive data science assistant powered by LLMs. Upload CSVs, explore datasets, and run statistical tests via natural language. Built with an agent-based architecture. A personal project to explore data science and LLM tools, including statistics and data science techniques, API and open-source LLMs, RAG, fine-tuning, etc.


# Minimum Viable Product features

CSV upload & automatic structure analysis

Dataset summary generation (variable types, missing values, descriptive statistics)

Natural language commands to run:

    T-tests

    ANOVA + Tukey HSD

    Chi-squared tests

    Correlation tests (Pearson, Spearman)

Automatic result interpretation in plain language

Plot generation where relevant

Agentized workflow for modular, extensible design


# Tech Stack

Python

Gradio – interactive UI

Pandas – data manipulation

Matplotlib – plots

OpenAI API – LLM-powered agents


# Project Roadmap 

This MVP focuses on statistical analysis and dataset exploration on a basic level.
Upcoming version will add to the existing features.
The planned next steps are:

    More statistical tests & EDA tools

    Missing value analysis (MCAR/MAR/MNAR) + automated imputation

    Quality-of-life improvements in UI (loading states, better chat flow)

    Machine learning models (classification & clustering)

    Open-source LLM integration for privacy-friendly offline mode

    RAG (Retrieval-Augmented Generation) for domain-specific Q&A

    QLoRA fine-tuning for specialized analysis


# Project Structure
    .
    agents/                  # LLM agents for specific tasks
    utils/                   # Helper functions
    datasets/                # Sample CSV datasets
    app_gradio.py            # Main Gradio app
    requirements.txt         # Python dependencies
    environment.yml          # Conda environment file
    README.md


Install using:
    git clone https://github.com/JoaoLAVaz/data-chat-assistant

    cd data-science-chat-assistant
    
    pip install -r requirements.txt

and run:
    python app_gradio.py
