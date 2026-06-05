# Biomedical Named Entity Recognition using BioBERT and PubMedBERT

## Overview

This project focuses on **Biomedical Named Entity Recognition (BioNER)** using transformer-based language models, specifically **BioBERT** and **PubMedBERT**, to automatically identify and classify biomedical entities from scientific literature.

Biomedical literature contains vast amounts of information related to genes, proteins, diseases, chemicals, drugs, and biological processes. Manually extracting this information is time-consuming and difficult to scale. This project aims to automate the extraction process by leveraging state-of-the-art Natural Language Processing (NLP) techniques.

The project also emphasizes **model evaluation and error analysis**, helping understand not only how well the model performs but also where and why it fails.

---

## Problem Statement

Biomedical researchers generate millions of publications every year. Extracting structured information from these documents is essential for:

* Biomedical knowledge discovery
* Literature mining
* Clinical decision support systems
* Knowledge graph construction
* Drug discovery and genomics research

Traditional rule-based systems struggle with the complexity of biomedical terminology. This project investigates whether transformer-based models can improve entity recognition performance while maintaining robustness across diverse biomedical texts.

---

## Objectives

* Develop a Biomedical Named Entity Recognition system using transformer-based models.
* Fine-tune domain-specific language models on biomedical datasets.
* Evaluate model performance using standard NER metrics.
* Analyze model errors and failure modes.
* Explore challenges associated with reliable biomedical information extraction.

---

## Models Used

### BioBERT

BioBERT is a domain-specific language model based on BERT and pre-trained on large biomedical corpora including PubMed abstracts and PMC full-text articles.

### PubMedBERT

PubMedBERT is trained exclusively on biomedical literature, enabling stronger understanding of domain-specific terminology and contextual relationships.

---

## Methodology

### 1. Data Preparation

* Dataset collection and preprocessing
* Text normalization
* Label formatting using BIO tagging
* Train-validation-test split generation

### 2. Tokenization

* Biomedical text tokenization
* Token-label alignment
* Handling sub-word tokens

### 3. Model Fine-Tuning

* Fine-tuning BioBERT and PubMedBERT models
* Hyperparameter optimization
* Training and validation monitoring

### 4. Evaluation

Models were evaluated using:

* Precision
* Recall
* F1 Score

Additional evaluation focused on:

* Entity-level performance
* Class-wise analysis
* Error categorization

### 5. Error Analysis

Detailed investigation of:

* False positives
* False negatives
* Ambiguous biomedical terminology
* Rare entity types
* Boundary detection errors

---

## Technologies Used

### Programming Languages

* Python

### Libraries and Frameworks

* Transformers (Hugging Face)
* PyTorch
* NumPy
* Pandas
* Scikit-learn

### Development Environment

* Jupyter Notebook
* Google Colab
* Linux

---

## Project Workflow


  Biomedical Text
        │
        ▼
Data Preprocessing
        │
        ▼
  Tokenization
        │
        ▼
BioBERT / PubMedBERT
        │
        ▼
    Fine-Tuning
        │
        ▼
    Prediction
        │
        ▼
    Evaluation
        │
        ▼
  Error Analysis

---

## Key Findings

* Transformer-based models significantly improve biomedical entity recognition performance compared to traditional approaches.
* Domain-specific pretraining provides better understanding of biomedical terminology.
* Model performance varies across entity categories.
* Rare entities and ambiguous terminology remain challenging.
* Evaluation beyond aggregate metrics is necessary to understand real-world model behavior.

---

## Challenges Encountered

* Complex biomedical vocabulary
* Imbalanced entity distributions
* Annotation inconsistencies
* Long scientific sentences
* Domain-specific abbreviations and acronyms

These challenges highlighted the importance of careful evaluation and robust benchmarking when deploying NLP systems in scientific and healthcare domains.

---

## Relevance to AI Reliability and Evaluation

One of the primary lessons from this project is that strong benchmark performance does not always imply reliable behavior.

While the models achieved good overall performance, detailed analysis revealed situations where predictions could fail due to ambiguity, rare terminology, or contextual variation. Understanding these failure modes is critical for developing trustworthy AI systems.

This project strengthened my interest in:

* AI evaluation
* Benchmark design
* Model robustness
* Failure analysis
* Trustworthy machine learning systems

---

## My Contributions

I was responsible for:

* Dataset preparation and preprocessing
* Model implementation and fine-tuning
* Training pipeline development
* Performance evaluation
* Error analysis
* Interpretation and reporting of results

I independently conducted experiments to compare model behavior, investigate failure cases, and identify opportunities for improving biomedical entity recognition performance.

---

## Future Improvements

* Explore larger biomedical language models
* Investigate retrieval-augmented approaches
* Improve handling of rare entities
* Develop more robust evaluation benchmarks
* Study model reliability under distribution shifts

---
