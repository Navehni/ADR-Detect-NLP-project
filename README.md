# ADR Detect: Adverse Drug Reaction Classification using LLMs and Embedding-Based Models

**Authors**: Nicole Poliak & Naveh Nissan  

---

## Overview

Adverse Drug Reactions (ADRs) are a major cause of patient harm and hospitalization, yet they are often buried in unstructured clinical notes or patient-written reviews. This project explores automated ADR detection from such texts using two different natural language processing (NLP) paradigms:

- **Zero-/Few-shot Large Language Models (LLMs)**  
- **Sentence Embeddings + Logistic Regression Classifier**

Our aim is to compare their effectiveness in identifying ADRs and evaluate their suitability for scalable, AI-driven medical decision support.

---

## Task Description

- **Input**: Sentence from medical literature (PubMed) or patient reviews (AskAPatient)
- **Output**: Binary label — `1` (ADR present) or `0` (ADR not present)
- **Task**: Sentence-level binary classification

---

## Datasets

We used two publicly available datasets:

- **ADE Corpus V2** – 23,516 expert-annotated biomedical sentences
- **PsyTAR** – 6,009 patient drug review sentences
- **Combined Dataset** – 26,867 cleaned, deduplicated sentences  
  Includes columns: `Text`, `Label`, `Dataset (ADE/PsyTAR)`

> Note: Due to label imbalance (~24% ADR), the majority class was downsampled for balance.

---

## Methods Compared

### Embedding-Based Classifier

- **Embeddings**: BioBERT, SBERT, InstructorXL
- **Classifier**: Logistic Regression
- **Pipeline**: 60/20/20 train/dev/test split (stratified)
- **Baseline**: Naïve Bayes + Bag-of-Words (BoW)

### Large Language Models (LLMs)

- **Models**: GPT-4o-mini, GPT-4o, Phi-4-mini-instruct, LLaMA-3.2-3B-Instruct
- **Strategy**: Zero-/Few-shot prompting (no fine-tuning)
- **Prompting**: Task description + optional labeled examples
- **Baseline**: GPT-4o-mini Zero-Shot Prompt

---

## Evaluation Metrics

- **Accuracy** – Overall classification performance  
- **Precision** – How many predicted ADRs were correct  
- **Recall** – How many true ADRs were identified (most important for safety)  
- **F1-Score** – Harmonic mean of precision and recall  
- **ROC-AUC** – Area under ROC curve (embedding models only)

---

## Baseline Results

| Method                          | Accuracy | Precision | Recall | F1-Score |
|---------------------------------|----------|-----------|--------|----------|
| BoW + Naïve Bayes               | 0.76     | 0.73      | 0.81   | 0.77     |
| GPT-4o-mini (Zero-Shot Prompt)  | 0.83     | 0.79      | 0.90   | 0.84     |

---

## Pipeline Summary

1. **Data Cleaning & Preprocessing** – Lowercasing, punctuation removal, duplicate filtering  
2. **Exploratory Data Analysis** – Distribution, length analysis, imbalance correction  
3. **Baselines** – BoW + Naïve Bayes, GPT-4o-mini (Zero-Shot Prompt)
4. **Embeddings Extraction** – SBERT, BioBERT, InstructorXL  
5. **Classifier Training** – Logistic Regression  
6. **LLM Evaluation** – Prompting via Open LLMs  
7. **Model Comparison** – Metric analysis, confusion matrix, ROC curves

---

## Novelty
- Integration of **diverse data sources** (expert reports and patient reviews) into a unified dataset
- Comparison between general-purpose LLMs and biomedical embedding models

---

## References

- [Simmering.dev Blog (2025)](https://simmering.dev/blog/modernbert-vs-llm/)
- [ACL Anthology 2025 Paper](https://aclanthology.org/2025.insights-1.11.pdf)
- [SCITEPRESS 2025 Study](https://www.scitepress.org/Papers/2025/131607/131607.pdf)

---

