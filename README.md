# Data Chat Assistant (v1)

An interactive **data science assistant** that lets users upload CSV datasets, explore them conversationally, and run statistically sound analyses via natural language.

The assistant combines **LLM-driven reasoning**, **formal statistical pipelines**, and an **agent-based workflow** to automatically select appropriate tests, validate assumptions, and explain results clearly.

This project started as an MVP and has evolved into **v1**, featuring a fully modular statistical engine, assumption-aware test selection, clustering, and a fine-tuned open-source explainer model.

---

##  Live Demo (Hosted App)

The application is publicly hosted on **Hugging Face Spaces**:

 **[Open the Data Chat Assistant](https://huggingface.co/spaces/Ozymandias2/data-chat-assistant)**

> Note: The app runs on GPU-backed infrastructure to support the fine-tuned explainer model.  
> Go ahead and restart the space and try it out! **The restart can take a few minutes**.

---

## Demo Video

[![Watch the demo]](https://www.youtube.com/watch?v=kJsbDPVbSEk)

---

## What’s New in v1

Version 1 significantly expands the original MVP with **statistical rigor**, **better architecture**, and **explainability**.

###  Agent Architecture
- Fully agentized workflow built with **LangGraph**
- Clear separation of concerns:
  - Decision-making LLM
  - Missing-data preprocessing
  - Statistical execution
  - Optional detailed explanation
- Modular and extensible by design

###  Statistical Pipelines (Assumption-Aware)
Each analysis automatically performs **pre-test diagnostics** and selects the correct method.

**Supported analyses:**
- **T-tests**
  - Student’s t-test
  - Welch’s t-test
  - Mann–Whitney U (nonparametric fallback)
- **ANOVA**
  - One-way ANOVA
  - Welch’s ANOVA
  - Kruskal–Wallis (nonparametric fallback)
- **Chi-squared analysis**
  - Chi-square test of independence
  - Fisher’s Exact Test (2×2 tables with low expected counts)
- **Correlation**
  - Pearson correlation
  - Spearman correlation
- **Clustering**
  - K-means clustering
  - PCA-based visualizations
  - Cluster profiling

###  Automatic Assumption Checks
Before running a test, the pipeline performs:
- Missing-data validation
- **Normality testing** (Shapiro–Wilk)
- **Variance homogeneity checks** (Levene / Brown–Forsythe)
- **Expected frequency checks** for contingency tables

The final test is chosen **programmatically**, not heuristically.

###  Missing Data Handling
- Dedicated missing-data node
- Summary of missingness
- Optional automated imputation
- Transparent reporting of all preprocessing steps

###  Smart Data Coercion
- Automatic numerical → categorical coercion when statistically safe  
  (e.g. binary 0/1 variables used in categorical tests)
- Explicitly documented in tool outputs for transparency

###  Fine-Tuned Explainer Model
- Optional detailed explanation mode powered by a **fine-tuned open-source LLM**
- Model: `Ozymandias2/qwen3-4b-instruct-stat-qlora-v2`
- Trained specifically to:
  - Interpret statistical pipelines
  - Explain assumptions → test choice → results → interpretation
- Toggleable in the UI (standard vs. detailed explanations)

 **Fine-tuning details:**  
If you’re interested in how the explainer model was trained, evaluated, and validated, check the **`result_explorer/`** directory.  
It documents:
- Model comparisons
- Evaluation methodology
- Dataset construction
- QLoRA fine-tuning process

###  Visualizations
- Automatic plot generation where appropriate
- PCA plots for clustering
- Clean separation between computation and visualization
- Plots shown in the UI without leaking filesystem paths

---

## Tech Stack

### Core
- **Python**
- **Pandas / NumPy / SciPy**
- **Statsmodels / Pingouin**
- **Scikit-learn**

### LLM & Orchestration
- **LangGraph** (agent workflow)
- **LangChain Core** (message abstractions)
- **OpenAI API** (decision LLM)
- **Hugging Face Transformers**
- **QLoRA fine-tuned model for explanations**

### UI
- **Gradio** (web interface)
- **Matplotlib** (plots)

---


## Installation (Local)

```bash
git clone https://github.com/JoaoLAVaz/data-chat-assistant
cd data-chat-assistant
pip install -r requirements.txt
```

Run locally:

```bash
python app.py
```
- Make sure you have a .env file with a working open ai key
---

## Roadmap (Post-v1)

- Multivariate tests and advanced EDA
- Classification models with prediction endpoints
- Model export and reuse
- Domain-specific RAG
- Further fine-tuning of explainer models
- Improved UI/UX and performance optimizations

---

## Motivation

This project is a **personal exploration of applied data science + LLM systems**, focusing on:
- Statistical correctness
- Transparent reasoning
- Modular, production-style architecture
- Practical explainability

It is designed both as a usable tool and as a learning platform for modern AI-assisted data analysis.
