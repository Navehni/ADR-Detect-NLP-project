# ADR Detect – Benchmarking LLMs vs. Embedding-Based Models for Adverse Drug Reaction Detection

**ADR Detect** is an NLP project aimed at evaluating and comparing the effectiveness of large language models (LLMs) and embedding-based classifiers for detecting **Adverse Drug Reactions (ADRs)** in clinical text.

This project was developed as part of an academic NLP course final assignment by **Naveh Nissan & Nicole Poliak**.

---

## Project Objective

To benchmark two distinct strategies for sentence-level ADR classification:

- **LLM-based generalization** – using GPT-4 in zero-shot and few-shot modes without any supervised training.
- **Embedding-based classification** – combining sentence vector representations (TF-IDF, SBERT, OpenAI) with simple classifiers.

---

## Dataset

We use the **ADE Corpus V2 – Classification Subset**:
- **23,517** expert-labeled sentences from ~3,000 PubMed case reports.
- **Labels**:  
  - `1` = ADR present  
  - `0` = No ADR
- **Format**:  
  | Column | Description |
  |--------|-------------|
  | `Text` | Clinical sentence |
  | `Label` | Binary ADR label (1 or 0) |

---

## Evaluation Strategy

### Embedding-Based Models
- Sentence → Embedding (TF-IDF, SBERT, OpenAI) → Classifier (Logistic Regression)
- Evaluation: **80/20 stratified train/test split**
- Baseline: **Naïve Bayes + Bag-of-Words**

### LLM-Based Models (Zero-/Few-Shot)
- Models: **GPT-4**, optionally Claude or GPT-3.5
- Zero-shot prompt (baseline) and few-shot prompts with example sentences
- No training or fine-tuning — evaluated directly on test data

### Metrics
- **Precision**
- **Recall**
- **F1-Score**

---

## ⚖omparison Goals

- Can general-purpose LLMs detect ADRs without training?
- How does the **choice of embedding** affect classifier performance?
- Can LLMs match or outperform supervised models on real-world biomedical data?

---
