<div align="center">

# 🧬 Biomedical NER — LPMO Knowledge Graph

### Named Entity Recognition with BioBERT & PubMedBERT → Neo4j Knowledge Graph

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge%20Graph-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com)
[![ISMB 2026](https://img.shields.io/badge/ISMB%202026-Accepted-brightgreen?style=flat-square)](https://www.iscb.org/ismb2026)

</div>

---

## 📌 Overview

This project focuses on **Biomedical Named Entity Recognition (BioNER)** using transformer-based language models — **BioBERT** and **PubMedBERT** — to automatically identify and classify biomedical entities from scientific literature, with a downstream goal of constructing a **domain-specific Knowledge Graph** for LPMO *(Lytic Polysaccharide Monooxygenase)* research.

> LPMOs are copper-dependent enzymes critical to biomass conversion and biofuel production. Knowledge about their interactions is scattered across a rapidly growing body of literature — this pipeline makes it structured and queryable.

The project also emphasizes **model evaluation and error analysis**, helping understand not only how well the model performs, but where and why it fails.

---

## 🏆 Results at a Glance

<div align="center">

| Metric | Score |
|---|---|
| **Overall F1** | **74.83%** |
| **Accuracy** | **92.87%** |
| ENZYME F1 | 79.36% |
| SUBSTRATE F1 | 75.60% |
| RELATION F1 | 67.01% |

</div>

<details>
<summary><b>Full per-entity breakdown</b></summary>

| Entity | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ENZYME | 78.89% | 79.84% | **79.36%** | 1,250 |
| SUBSTRATE | 75.70% | 75.49% | **75.60%** | 1,387 |
| RELATION | 67.87% | 66.17% | **67.01%** | 878 |
| **Overall** | **74.94%** | **74.71%** | **74.83%** | 3,515 |

</details>

---

## 🗂️ Repository Structure

```
Bio_NER/
├── main_project.py                    # BioBERT fine-tuning (80/20 split)
├── pubmain_project.py                 # PubMedBERT fine-tuning
├── pubmed_model_combined_70_15_15.py  # PubMedBERT (70/15/15 split) ← best model
├── pubmed_model_70_15_15_2.py
├── pubmedbert_model_v2_improved.py
├── pubmedbert_weighted_v1.py
├── pubmain_project_crf.py             # CRF variant experiments
├── pubmain_project_crf_2.py
├── test_model.py                      # Inference & testing
├── test_2.py
├── final_extractor.py                 # Entity extraction pipeline
├── triples_with_genes.py              # Triple generation for KG
├── generate_graph.py                  # Graph building utilities
├── visualize_graph.py                 # Graph visualization
├── universal_fix.py                   # Dataset normalization utilities
├── verify_data.py                     # Data validation
│
├── train.json                         # Annotated dataset (3,178 samples)
├── train_vennila.json                 # Second annotated dataset (1,960 samples)
├── predicted_ner_tags.json            # Model predictions output
├── test_metrics.json                  # Test set metrics (PubMedBERT best)
├── test_metrics_improved.json         # Metrics from improved run
│
├── triples.csv                        # Extracted entity triples (8,065)
├── nodes.csv / nodes_knet.csv         # KG nodes (119)
├── edges.csv / edges_knet.csv         # KG edges (257)
├── final_knowledge_graph.csv          # Complete KG export
├── neo4j_query_table_data_2026-4-8.csv
├── 1_all_words.csv
├── 2_knowledge_graph.csv
│
├── svein_horn_papers.txt              # Source literature corpus
├── input_data.txt
├── biobert_Model_result.txt           # BioBERT training logs
├── pubmed_model_result.txt            # PubMedBERT training logs
├── pubmed_model_combined_70_15_15_results.txt
├── pubmedbert_weighted_loss_result.txt
│
├── kishore dataset.xlsx               # Dataset overview
├── test_result_Edward_A._Bayer.xlsx   # Per-author test results
├── final_network_diagram.png          # Knowledge graph visualization
│
├── Long_abstract_ISMB_2026.pdf/.docx  # ISMB 2026 submission
└── short_abstract_ISMB_2026.docx
```

---

## 🔬 Problem Statement

Biomedical researchers generate millions of publications every year. Extracting structured information from these documents is essential for:

- 🔍 Biomedical knowledge discovery
- 📚 Literature mining
- 🏥 Clinical decision support systems
- 🕸️ Knowledge graph construction
- 💊 Drug discovery and genomics research

Traditional rule-based systems struggle with the complexity of biomedical terminology. This project investigates whether transformer-based models can improve entity recognition performance while maintaining robustness across diverse biomedical texts.

---

## 🎯 Objectives

- Develop a BioNER system using transformer-based models
- Fine-tune domain-specific language models (BioBERT, PubMedBERT) on a custom LPMO dataset
- Evaluate model performance using standard NER metrics (Precision, Recall, F1)
- Analyze model errors and failure modes
- Construct a domain-specific Knowledge Graph from extracted entities and relationships using Neo4j
- Explore challenges associated with reliable biomedical information extraction

---

## 🤖 Models Used

### BioBERT — `dmis-lab/biobert-v1.1`

Pre-trained on PubMed abstracts and PMC full-text articles. Fine-tuned on the merged dataset (5,138 samples) using an **80/20** train-test split. Achieved ~69.5% overall F1.

### PubMedBERT — `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` ⭐ Best

Trained exclusively on PubMed abstracts — stronger domain vocabulary. Fine-tuned on `train.json` (3,178 samples) using a **70/15/15** train-validation-test split. Achieved **74.83% F1** — the best result.

---

## 🛠️ Methodology

### 1 · Data Preparation

- Dataset curated from titles and abstracts of **14 LPMO-focused authors**
- Manually annotated into four entity categories using **BIO tagging**:

  | Tag | Entity | Examples |
  |---|---|---|
  | `B-ENZYME` / `I-ENZYME` | Enzyme names | LpCel5A, LPMO, GH61 |
  | `B-SUBSTRATE` / `I-SUBSTRATE` | Biochemical substrates | cellulose, chitin, xylan |
  | `B-RELATION` / `I-RELATION` | Biological relationships | oxidizes, cleaves, binds |
  | `O` | All other tokens | — |

- Source files: `train.json` (3,178) + `train_vennila.json` (1,960) → **5,138 total samples**
- Splits: **70 / 15 / 15** (PubMedBERT) · **80 / 20** (BioBERT)

### 2 · Tokenization

- Model-specific fast tokenizers (BertTokenizerFast)
- Sub-word token handling: label assigned to **first sub-token**, `-100` for the rest
- Handles variable-length sequences up to 512 tokens

### 3 · Model Fine-Tuning

- `BertForTokenClassification` with 7 output labels (`O`, `B/I-ENZYME`, `B/I-SUBSTRATE`, `B/I-RELATION`)
- AdamW optimizer with linear learning rate warmup
- Early stopping based on validation F1
- GPU training on **NVIDIA GeForce RTX 2050** (CUDA)

### 4 · Evaluation

Metrics computed using **seqeval** (entity-level, not token-level):

- Precision · Recall · F1 Score · Token-level Accuracy
- Per-class breakdown: ENZYME, SUBSTRATE, RELATION

### 5 · Knowledge Graph Construction

After NER, extracted triples were loaded into **Neo4j**:

```
(ENZYME) --[RELATION]--> (SUBSTRATE)
e.g. (LpAA9A) --[oxidizes]--> (cellulose)
```

| KG Stat | Count |
|---|---|
| Nodes | 119 |
| Edges | 257 |
| Triples | 8,065 |
| Final KG records | 2,874 |

Designed for future integration with [KnetMiner](https://knetminer.com).

### 6 · Error Analysis

- False positives / false negatives per entity class
- Ambiguous biomedical terminology
- Boundary detection errors
- Rare entity types and domain-specific abbreviations

---

## 📊 Training Curves

PubMedBERT (best run, 70/15/15 split):

| Epoch | Train Loss | Val F1 |
|---|---|---|
| 1 | 2.35 | 57.3% |
| 3 | 0.46 | 70.5% |
| 6 | 0.16 | 73.6% |
| 9 | 0.11 | 74.1% |
| 15 | 0.10 | 74.3% |

> Loss decreased from ~2.35 → below 0.10. Performance plateaued after ~9–11 epochs.

---

## ⚙️ Technologies

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | PyTorch, Transformers (HuggingFace) |
| NER Metrics | HuggingFace `evaluate` (seqeval) |
| Data | NumPy, Pandas, Scikit-learn |
| Graph DB | Neo4j |
| Environment | Anaconda (`torch_gpu`), Windows, NVIDIA RTX 2050 |

---

## 🔄 Project Workflow

```
  PubMed Abstracts (14 LPMO authors)
             │
             ▼
  Manual Annotation (BIO tagging: ENZYME / SUBSTRATE / RELATION / O)
             │
             ▼
  Tokenization  ──►  Sub-word alignment
             │
             ▼
  Fine-Tuning  ──►  BioBERT  /  PubMedBERT ⭐
             │
             ▼
  Evaluation  ──►  Precision · Recall · F1 · Accuracy
             │
             ▼
  Relation Extraction  ──►  Triple generation (Subject, Relation, Object)
             │
             ▼
  Neo4j Knowledge Graph  ──►  119 nodes · 257 edges · 8,065 triples
             │
             ▼
  Error Analysis & Reporting
```

---

## ⚠️ Challenges

- **Complex vocabulary** — LPMO-specific terms, abbreviations (LPMO, GH61, AA9)
- **Class imbalance** — unequal distribution of ENZYME, SUBSTRATE, RELATION tags
- **Annotation inconsistencies** — across two independently annotated source files
- **Long sentences** — scientific abstracts with nested entity mentions
- **Relation ambiguity** — same verb can describe different biological events in context

---

## 🔍 Relevance to AI Reliability

> *Strong benchmark performance does not always imply reliable behavior.*

While models achieved good overall performance, error analysis revealed failure modes caused by ambiguity, rare terminology, and contextual variation. Understanding these is critical for deploying trustworthy AI in scientific and healthcare settings.

Key takeaways:

- Careful **benchmark design** matters as much as model architecture
- **Entity boundary errors** are common and often invisible in aggregate metrics
- Domain-specific pretraining (PubMedBERT) outperforms general pretraining (BioBERT) on specialized corpora

---

## 👤 My Contributions

- Dataset preparation, merging, and normalization (`train.json` + `train_vennila.json`)
- Model implementation and fine-tuning for BioBERT and PubMedBERT
- Training pipeline with early stopping and validation monitoring
- Performance evaluation using seqeval (entity-level metrics)
- Knowledge Graph construction in Neo4j from predicted triples
- Error analysis and failure mode interpretation
- Abstract accepted at **ISMB 2026** (International Society for Computational Biology)

---

## 🚀 Future Work

- [ ] Explore larger biomedical LMs — BioGPT, LLaMA-Med
- [ ] Retrieval-augmented approaches for rare entity handling
- [ ] Expand entity types: MICROBE, ORGANISM, REACTION
- [ ] Integrate KG with [KnetMiner](https://knetminer.com) for cross-domain discovery
- [ ] Interactive Neo4j querying interface
- [ ] Extend to broader CAZyme / glycoside hydrolase families
- [ ] Study robustness under distribution shift (full-text articles vs. abstracts)

---

## 📄 Publication

> **Construction of a Domain-Specific Knowledge Graph for LPMO Research Using Machine Learning-Based Named Entity Recognition**
>
> Kishore Matheswaran, Vennila Kanchana Devi Marimuthu, Ragothaman M. Yennamalli
>
> *Accepted — ISMB 2026, International Society for Computational Biology*

---

<div align="center">
<sub>SASTRA Deemed to be University, Thanjavur, India · Dept. of Mathematics & Bioinformatics</sub>
</div>
