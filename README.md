# Cordis Sentinel

**An agentic heart attack risk prediction pipeline — from raw clinical data to explainable, AI-reasoned risk reports.**

> Status: 🚧 In Development — data cleaning & EDA phase

---

## Overview

Cordis Sentinel predicts an individual's heart attack risk from clinical and lifestyle data, then goes a step further: an agentic reasoning layer interprets the model's output and generates a clear, explainable risk summary — rather than just spitting out a probability score.

The project follows a **Plan-Execute-Reflect (PER)** agentic architecture, the same pattern used in [CreditIQ](#), applied here to the healthcare risk domain.

## Why this project

Most heart attack prediction projects stop at a classifier and an accuracy score. Cordis Sentinel is built around two harder problems instead:

1. **The dataset is genuinely messy** — inconsistent formats, missing values, and encoding issues that need real cleaning decisions, not `dropna()` and move on.
2. **A prediction alone isn't useful** — the agentic layer takes the model's output and its SHAP-based feature attributions and reasons over them to produce a human-readable, clinically-grounded explanation of *why* the risk was flagged.

## Architecture

```
Raw Data → Cleaning → EDA → Feature Engineering → Model Training
                                                        ↓
                                              SHAP Explainability
                                                        ↓
                                         Agentic Layer (Plan-Execute-Reflect)
                                                        ↓
                                          Explainable Risk Report
```

**Planned pipeline stages:**
- **Data Cleaning** — handling missing values, inconsistent categorical encodings, dtype fixes, outlier treatment
- **EDA** — univariate/bivariate analysis, class imbalance check, feature-target relationships
- **Feature Engineering** — encoding, scaling, imbalance handling
- **Modeling** — Logistic Regression & Decision Tree Classifier (baseline comparisons)
- **Explainability** — SHAP values for per-prediction feature attribution
- **Agentic Layer** — LangGraph-based PER agent that consumes model + SHAP output and generates a reasoned, retrieval-grounded risk explanation

## Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Pandas, NumPy |
| Modeling | Scikit-learn (Logistic Regression, Decision Tree) |
| Explainability | SHAP |
| Agentic Framework | LangGraph |
| Vector Store / RAG | ChromaDB |
| LLM | Llama 3 (via Groq) |
| Serving (planned) | Streamlit |

## Project Structure

```
cordis-sentinel/
├── data/
│   ├── raw/               # original dataset, untouched
│   ├── interim/           # cleaning checkpoints
│   └── processed/         # final model-ready dataset
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation_shap.ipynb
│   └── 07_agentic_pipeline.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── agent/
│       ├── tools.py
│       ├── graph.py
│       └── prompts.py
├── models/
├── chroma_db/
├── reports/
│   └── figures/
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## Roadmap

- [ ] Data inspection & diagnostic report
- [ ] Data cleaning pipeline
- [ ] Exploratory data analysis
- [ ] Feature engineering
- [ ] Baseline model training (Logistic Regression, Decision Tree)
- [ ] SHAP-based explainability
- [ ] Agentic reasoning layer (PER architecture)
- [ ] Streamlit demo app

## Disclaimer

This project is for educational and portfolio purposes only. It is **not** a certified medical tool and should not be used for real clinical decision-making.

---

*Part of an ongoing series of agentic ML pipelines — see also: CreditIQ (credit risk assessment).*