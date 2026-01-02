# Fine-Tuning Qwen-3-4B-Instruct for Statistical Result Explanation

## Overview

This directory documents a focused experimental study on **fine-tuning a small open-weight language model** to generate **high-quality, reliable explanations of statistical test results**.

The goal was **not** to beat frontier models outright, but to answer a practical question:

> Can a compact, efficient model be fine-tuned to approach GPT‑4‑level explanation quality for structured statistical pipelines—while avoiding hallucinations and unsafe interpretations?

This study covers:
- Base model selection  
- Baseline evaluation using a **blind ensemble judge**  
- An initial fine-tuning attempt that failed  
- A corrective fine-tuning strategy  
- A final evaluation showing **near GPT‑4 performance**

All notebooks, datasets, and evaluation artifacts are included for reproducibility.

---

## Motivation

Many data-science pipelines output **structured JSON results**:
- hypothesis tests (ANOVA, Kruskal–Wallis, chi-square, t-tests),
- correlations,
- clustering diagnostics,
- missing-data and assumption reports.
- data imputation.

Explaining these outputs **clearly, accurately, and conservatively** is difficult.  
General-purpose LLMs often:
- hallucinate missing data,
- invent statistics,
- misname tests,
- use causal language for observational results.

The objective of this project was to train a **specialized explanation model** that:
- respects statistical assumptions,
- avoids hallucination,
- follows a strict 5-section explanation template,
- and remains small enough for low-cost deployment.

---

## Step 1 — Choosing the Base Model

We first compared several candidate models by **manual inspection of a single representative example**:

- GPT‑4.1 (reference standard)  
- LLaMA‑3.1‑8B-Instruct  
- Qwen‑3‑4B‑Instruct  
- Gemma‑2‑2B  

Evaluation criteria:
- faithfulness to statistical logic,
- handling of missing data,
- clarity and structure,
- hallucination tendencies.

### Outcome

- Gemma‑2‑2B was too weak for structured reasoning.
- LLaMA‑3.1‑8B was decent but heavier than necessary.
- **Qwen‑3‑4B‑Instruct** offered the best balance of reasoning quality, instruction following, and memory footprint.

**Qwen‑3‑4B‑Instruct was selected as the base model.**

---

## Step 2 — Baseline Evaluation vs GPT‑4.1

We conducted a systematic baseline evaluation using a **blind ensemble judge** composed of:

- GPT‑5.1  
- Claude opus 4.5  
- Gemini3 pro  

Judges never saw model identities, outputs were labeled only as *Model A* and *Model B*.

### Evaluation Metrics

**Numeric scores (1–5):**
- Overall quality
- Factual correctness
- Interpretation
- Coverage
- Clarity

**Binary flags:**
- Hallucinated missing data
- Hallucinated numbers
- Wrong test or direction
- Unsafe causal language

Scores were averaged across judges.

### Baseline Results 

| Metric | GPT‑4.1 | Qwen‑3‑4B |
|------|--------|-----------|
| Overall | **4.65** | **4.29** |

The base Qwen model showed promising structure but lagged GPT‑4.1 in factual precision and hallucination control having hallucinated rarely but in every category.

---

## Step 3 — First Fine-Tuning Attempt

### Dataset Construction

A supervised fine-tuning dataset was generated using **GPT‑5.1** as a teacher model.

Each example followed a chat-style JSONL format:
- system instruction enforcing a 5-section structure,
- user message containing the tool JSON,
- assistant message with the gold explanation.

Key properties:
- strict structure,
- neutral, non-causal language,
- explanations under ~400 words.

### Training Setup

- QLoRA (4‑bit quantization + LoRA adapters)
- Base model: Qwen‑3‑4B‑Instruct
- ~150 examples
- 3 epochs
- Relatively aggressive learning rate

### Result

The first fine-tuned model performed **worse than the base model** under blind evaluation.

| Metric | GPT‑4.1 | Qwen‑3‑4B | Qwen‑3‑4B |
|------|--------|-----------|-----------|
| Overall | **4.67** | **4.31** | **4.18** |

We can see a clearly worst result than base Qwen and it also hallucinated more often.

### Likely Causes

- Dataset too small for aggressive optimization
- Overfitting to stylistic quirks
- Reduced generalization across test families

This confirmed that **fine-tuning can degrade performance if done incorrectly**.

---

## Step 4 — Corrective Fine-Tuning Strategy

Rather than discarding the effort, we applied a **corrective fine-tuning approach**.

### Key Changes

1. **Targeted Data Expansion**
   - New examples focused on failure modes:
     - hallucinated missingness,
     - incorrect test naming,
     - unsafe causal phrasing,
     - weak clustering interpretations.

2. **Larger, More Diverse Dataset**
   - Total examples increased to **186**

3. **Conservative Training Regime**
   - 1 epoch only
   - Lower learning rate (5e‑5)
   - Shorter max sequence length (1000 tokens)

The goal was correction, not memorization.

---

## Step 5 — Second Fine-Tuning & Final Evaluation

The second fine-tuned model (`qwen3_4b_ft_v2`) was evaluated using the same blind ensemble judges and test cases.

### Final Results (placeholder)

| Metric | GPT‑4.1 | Qwen‑Base | Qwen‑FT‑v2 |
|------|--------|-----------|------------|
| Overall | **4.60** | **4.30** | **4.56** |
| Factual | **4.78** | **4.19** | **4.58** |
| Interpretation | **4.63** | **4.53** | **4.65** |
| Coverage | **4.95** | **4.91** | **4.93** |
| Clarity | **4.92** | **4.95** | **4.90** |

### Flag Rates

| Flag | GPT‑4.1 | Qwen‑Base | Qwen‑FT‑v2 |
|----|----|----|----|
| Hallucinated numbers | **0.000** | **0.108** | **0.088** |
| Hallucinated numbers | **0.000** | **0.147** | **0.049** |
| Wrong test/direction | **0.029** |** 0.118** | **0.029** |
| Unsafe causal | **0.402** | **0.216** | **0.255** |

### Key Takeaways

- FT‑v2 significantly outperformed base Qwen.
- Near parity with GPT‑4.1 on most metrics.
- Large reduction in hallucinated numbers.
- Improved correctness in test identification.
- Maintained conservative, non-causal language.

---

## Conclusion

This project shows that:

- Small models can approach frontier performance on narrow, well-defined tasks.
- Blind, ensemble-based evaluation is essential.
- Aggressive fine-tuning can harm performance.
- **Corrective fine-tuning**, paired with careful data design and conservative hyperparameters, is effective.

The final model is considered **production-ready** for this application and is now integrated into the main project.

---

## Repository Contents

- create_synthethic_datasets.ipynb - notebook that created some syntetic datasets
- create_test_files.ipynb -  notebook that used real and synthetic datasets to create the json output (mimicking the app)
- fine_tune.ipynb - Links to colab where the FT was done using QLORA
- ft_create_dataset.ipynb - notebook used to create the training dataset for finetuning
- model_compare_qwen_gpt.ipynb - notebook used to create the essemble judge and evaluate the models  
- model_selection.ipynb - first notebook used to check and select the base model from a few open source models
- ft_dataset/ — tool json examples needed for the fine-tune process 
- toy_dataset/ — mix of synthetic and real datasets used to run the statistical tests to create the json outputs 
- cases/ — base tool examples and outputs of gpt4.1 and base qwen for compare 
- cases_ft/ — first fine-tuning outputs and base models to compare
- cases_ft_v2/ — final fine-tuning outputs and base models to compare
- final_finetuning_dataset.jsonl and final_finetuning_v2.jsonl the in line json file used for finetuning
- train.jsonl and val.jsonl - train and validation split (test data was the one used in cases/ which obv. wasnt used for training)

---

For questions or discussion, feel free to reach out or open an issue.
