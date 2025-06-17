# ADR Detect: Comparing LLM Generalization with Embedding-based Models for Adverse Drug Reaction Detection

**Authors**: Nicole Poliak & Naveh Nissan

---

## Overview

Adverse Drug Reactions (ADRs) are a major cause of patient harm and hospitalization, yet they are often buried in unstructured clinical notes or patient-written reviews. This project explores automated ADR detection from such texts using two different natural language processing (NLP) paradigms:

- **Zero-/Few-shot Large Language Models (LLMs)**  
- **Sentence Embeddings + Logistic Regression Classifier**

Our aim is to compare their effectiveness in identifying ADRs and evaluate their suitability for scalable, AI-driven medical decision support.

---

## Task Definition

- **Input**: A single sentence from a biomedical abstract or patient review  
- **Output**: Binary label  
  - `1`: ADR present  
  - `0`: No ADR present  
- **Task**: Sentence-level binary classification (ADR vs. Non-ADR)

---

## Datasets

We used two publicly available datasets:

| Dataset        | Description                                      | Size     |
|----------------|--------------------------------------------------|----------|
| **ADE Corpus V2** | Expert-annotated PubMed sentences on drug events | 23,516   |
| **PsyTAR**     | Patient-written drug reviews from AskAPatient    | 6,009    |
| **Combined**   | Merged, cleaned, deduplicated dataset             | 26,867 → 12,862 after downsampling |

Each entry has:
- `text`: the sentence
- `label`: binary (1 = ADR, 0 = Non-ADR)
- `dataset`: source origin (ADE or PsyTAR)

> Due to class imbalance (~24% ADR), downsampling was applied to the majority class.

---

## Exploratory Data Analysis (EDA)

- **Final size after balancing**: 6,431 ADR + 6,431 Non-ADR = 12,862 total  
- **Average sentence length**: 17.2 words  
- **Duplicates removed**: 2,639  
- **Total words**: 462,108  
- **Total characters**: 2.6M+  

Text preprocessing included:
- Lowercasing  
- Punctuation removal  
- Rechecking duplicates  

---
## Evaluation Metrics

- **Accuracy** – Overall classification performance  
- **Precision** – How many predicted ADRs were correct  
- **Recall** – How many true ADRs were identified (most important for safety)  
- **F1-Score** – Harmonic mean of precision and recall
- **Confusion Matrix** – Shows counts of correct and incorrect predictions across ADR and non-ADR classes.
- **ROC-AUC** – Area under ROC curve (BoW + embedding models only)

---

## Modeling Approaches

### Baseline Models
- **Approach**: Bag-of-Words + Naïve Bayes & GPT-4o-mini Zero-Shot Prompt

## Baseline Results

| Method                          | Accuracy | Precision | Recall | F1-Score |
|---------------------------------|----------|-----------|--------|----------|
| BoW + Naïve Bayes               | 0.76     | 0.73      | 0.81   | 0.77     |
| GPT-4o-mini (Zero-Shot Prompt)  | 0.84     | 0.81      | 0.89   | 0.85     |

---


### 2. Embedding-Based Models
- **Embeddings**: `SBERT`, `BioBERT`, `InstructorXL`
- **Classifier**: Logistic Regression
- **Split**: Stratified 60/20/20 (train/dev/test)
- **Embedding Model Configurations:** max_iter=1000 in Logistic Regression and batch_size=16 during BioBERT embedding

### 3. LLMs (Zero-/Few-shot)
- **Models**: `GPT-4o`, `GPT-4o-mini`, `Phi-4-mini-instruct`, `LLaMA-3.2-3B-Instruct`
- The models were evaluated using only test data.  
- **Prompting Strategy**:  
  - Zero-shot: no examples  
  - Few-shot: 4–8 examples 
- **Settings**: `max_tokens=5`, `temperature=0.0-0.1`, `top_p=1.0`
- **Inference Platform**: Azure OpenAI (GPTs, Phi), Hugging Face (LLaMA)
- **Platform**: Google Colab Pro+ with A100 GPU.

---

## Results

| Model                    | Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------|-----------|--------|----------|
| BoW + Naive Bayes        | 0.76     | 0.73      | 0.81   | 0.77     |
| SBERT + LR               | 0.78     | 0.77      | 0.80   | 0.78     |
| BioBERT + LR             | 0.82     | 0.82      | 0.82   | 0.82     |
| InstructorXL + LR        | 0.81     | 0.81      | 0.81   | 0.81     |
| Phi-4-mini Zero-Shot     | 0.77     | 0.73      | 0.87   | 0.79     |
| Phi-4-mini Few-Shot      | 0.73     | 0.75      | 0.69   | 0.72     |
| LLaMA Zero-Shot          | 0.71     | 0.73      | 0.66   | 0.70     |
| LLaMA Few-Shot           | 0.60     | 0.74      | 0.36   | 0.49     |
| GPT-4o-mini Zero-Shot    | 0.84     | 0.81      | 0.89   | 0.85     |
| GPT-4o-mini Few-Shot     | 0.85     | 0.85      | 0.85   | 0.85     |
| GPT-4o Zero-Shot         | 0.81     | 0.75      | 0.95   | 0.84     |
| GPT-4o Few-Shot          | 0.84     | 0.81      | 0.89   | 0.85     |   

---

## Full Pipeline

1. **Data Preparation**  
   Merge ADE and PsyTAR datasets

2. **EDA**  
   Checking class distribution, duplicate removal, sentence and word length analysis, text cleaning, handling class imbalance

3. **Baseline Modeling**  
   Bag-of-Words + Naïve Bayes

4. **Embedding Feature Extraction**  
   Generate dense vectors via InstructorXL, SBERT, BioBERT

5. **Classifier Training**  
   Logistic Regression using sentence embeddings

6. **LLM Prompting & Evaluation**  
   Zero-/few-shot testing on multiple LLMs

7. **Evaluation & Visualization**  
   Confusion matrices, ROC curves, metric comparisons

---

## Insights 

- **LLMs** like GPT-4o achieved **top recall and F1** even without training
- **BioBERT embeddings** gave strong performance with efficient training
- **Prompt design** was key—zero-shot often outperformed few-shot
- **LLaMA** performed poorly due to lack of fine-tuning and response quality
- For clinical applications, **recall** is prioritized to avoid missing ADRs

---

## Graphical Abstract

![ADR Detect - Graphical Abstract](./graphical_abstract.png)

---

## Repository Contents

| File | Description |
|------|-------------|
| `combined_dataset.csv` | Final structured dataset |
| `data_preparation.ipynb` | Dataset merging and preprocessing |
| `ADR_classification_pipeline.ipynb` | Modeling, evaluation, and results |
| `graphical_abstract.png` | Project visual summary |
| `overall_results.csv` | Aggregated results of all models |

---

## References

- [Simmering.dev Blog (2025)](https://simmering.dev/blog/modernbert-vs-llm/)
- [ACL Anthology (2025)](https://aclanthology.org/2025.insights-1.11.pdf)
- [SCITEPRESS (2025)](https://www.scitepress.org/Papers/2025/131607/131607.pdf)
- [ADE Corpus V2 on Hugging Face](https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2)
- [PsyTAR Dataset Info](https://www.askapatient.com/store/#!/Psytar-Data-Set/p/449080512/category=129206256)

---

## Novelty

- Integration of **diverse data sources** (expert reports and patient reviews) into a unified dataset
- Comparison between general-purpose LLMs and biomedical embedding models

---
