# BioIE-LLM

**Biological Information Extraction from Large Language Models (LLMs)**

This is the official code repository for the following paper:

- [Comparative Performance Evaluation of Large Language Models for Extracting Molecular Interactions and Pathway Knowledge](https://arxiv.org/abs/2307.08813)

For the earlier study and preliminary results, please visit the companion repository: [BioIE-LLM-Prelim](https://github.com/boxorange/BioIE-LLM-Prelim), which corresponds to the paper:

- [Automated Extraction of Molecular Interactions and Pathway Knowledge using Large Language Model, Galactica: Opportunities and Challenges](https://aclanthology.org/2023.bionlp-1.22/)


## Installation
The dependencies are listed in requirements.txt


## Datasets
* [STRING DB](https://string-db.org): the human (Homo sapiens) protein network for performing a protein-protein interaction (PPI) recognition task.
* [Negatome DB](https://pubmed.ncbi.nlm.nih.gov/24214996) a specialized repository dedicated to cataloging Non-Interacting Proteins (NIPs).
* [KEGG DB](https://www.genome.jp/kegg): the KEGG human pathways which have been identified as being activated in response to low-dose radiation exposure in [the study](https://arxiv.org/abs/2301.01769).
* [INDRA DB](http://www.indra.bio): a set of human gene regulatory relation statements that represent mechanistic interactions between biological agents.


## Reproduction
To reproduce the results of the experiments, use the bash script [run.sh](scripts/run.sh). You need to change model/data paths accordingly.


## Results
Here are the results of the experiments. The experiments were conducted on 4×NVIDIA A100 80GB GPUs. Note different number of GPUs and batch size can produce slightly different results.


### Recognizing Protein-Protein Interactions with LLMs

This document presents the evaluation results of various Large Language Models (LLMs) on tasks involving the recognition of protein-protein interactions (PPIs) using data from the STRING and Negatome databases.

---

### STRING DB PPI Task (Generative Question)

We evaluated the ability of LLMs to generate lists of proteins that interact with a given protein, based on a human protein network from the STRING database. For this task:

- 1,000 proteins were randomly selected as queries.
- Each model was prompted to generate 10 interacting proteins for each query.
- These 10,000 generated PPI pairs (1,000 proteins × 10 predictions) were compared against known interacting proteins in the STRING DB.
- Due to model generation constraints and inference efficiency, only the top 10 predictions per protein were considered.
- Evaluation metrics include:
  - **Micro F1**: Measures prediction accuracy over all 10,000 PPI pairs.
  - **Macro F1**: Measures average prediction accuracy per individual protein.
  - **# Full Matches**: The number of proteins (out of 1,000) for which all 10 predicted interactors matched the ground truth.

### Results

| Model                              | Micro F1 | Macro F1 | # Full Matches (out of 1,000) |
| ---------------------------------- | -------- | -------- | ----------------------------- |
| BioGPT-Large (1.5B)                | 0.1220   | 0.1699   | 10                            |
| BioMedLM (2.7B)                    | 0.1598   | 0.1992   | 61                            |
| Galactica (6.7B)                   | 0.2110   | 0.2648   | 75                            |
| Galactica (30B)                    | 0.2867   | 0.3516   | 110                           |
| Alpaca (7B)                        | 0.0998   | 0.1388   | 16                            |
| RST (11B)                          | 0.0987   | 0.1523   | 10                            |
| Falcon (7B)                        | 0.0435   | 0.0632   | 7                             |
| Falcon (40B)                       | 0.1246   | 0.1607   | 35                            |
| MPT-Chat (7B)                      | 0.1313   | 0.1658   | 45                            |
| MPT-Chat (30B)                     | 0.2926   | 0.3467   | 144                           |
| LLaMA2-Chat (7B)                   | 0.2807   | 0.3498   | 89                            |
| LLaMA2-Chat (70B)                  | 0.3517   | 0.4187   | 159                           |
| Mistral-Instruct (7B)             | 0.2762   | 0.3299   | 126                           |
| **Mixtral-8x7B-Instruct (46B)**    | **0.3867** | **0.4295** | **258**                      |
| SOLAR-Instruct (10.7B)            | 0.2766   | 0.3260   | 141                           |

> **Note**: A 5-shot prompting strategy was used. Bolded values indicate the best-performing model.

---

### STRING/Negatome DB PPI Task (Yes/No Classification)

In this task, LLMs were evaluated on their ability to determine whether a given protein pair interacts. We used a balanced set of 2,000 pairs (1,000 known positives from STRING and 1,000 negatives from the Negatome DB).

- Models were prompted with yes/no questions.
- Performance was measured using:
  - **Micro F1**: Accuracy across all examples.
  - **Macro F1**: Average F1 score per class.
- The number of shots (example demonstrations) used per model is also noted.

### Results

| Model                             | Micro F1 (#shot)      | Macro F1 (#shot)      |
|----------------------------------|------------------------|------------------------|
| BioGPT-Large (1.5B)              | 0.5700 (1-shot)        | 0.4811 (1-shot)        |
| BioMedLM (2.7B)                  | 0.7125 (2-shot)        | 0.6866 (2-shot)        |
| Galactica (6.7B)                 | 0.5320 (1-shot)        | 0.4568 (1-shot)        |
| Galactica (30B)                  | 0.8585 (5-shot)        | 0.8585 (5-shot)        |
| Alpaca (7B)                      | 0.6660 (5-shot)        | 0.6241 (5-shot)        |
| RST (11B)                        | 0.6990 (0-shot)        | 0.6701 (0-shot)        |
| Falcon (7B)                      | 0.5000 (1-shot)        | 0.3333 (1-shot)        |
| Falcon (40B)                     | 0.5050 (1-shot)        | 0.3443 (1-shot)        |
| **MPT-Chat (7B)**                | **0.9795 (5-shot)**    | **0.9795 (5-shot)**    |
| MPT-Chat (30B)                   | 0.9345 (5-shot)        | 0.9343 (5-shot)        |
| LLaMA2-Chat (7B)                 | 0.8670 (5-shot)        | 0.8662 (5-shot)        |
| LLaMA2-Chat (70B)                | 0.9545 (5-shot)        | 0.9545 (5-shot)        |
| Mistral-Instruct (7B)            | 0.7745 (5-shot)        | 0.7707 (5-shot)        |
| Mixtral-8x7B-Instruct (46B)      | 0.7770 (5-shot)        | 0.7658 (5-shot)        |
| SOLAR-Instruct (10.7B)           | 0.7615 (3-shot)        | 0.7481 (3-shot)        |

---

These results highlight both the opportunities and current limitations of LLMs in extracting biological knowledge, particularly in complex domains like protein interaction networks.



### KEGG Pathway Recognition

This experiment evaluates the performance of Large Language Models (LLMs) in recognizing genes associated with human pathways relevant to low-dose radiation (LDR) exposure using the KEGG database.

#### Task Description

- **KEGG DB Pathways affected by LDR exposure Task (Generative Task)**: For each of the top 100 KEGG pathways associated with LDR exposure, models were prompted to generate 10 genes. These predictions were compared to the actual gene sets associated with each pathway.
- Due to the extensive nature of KEGG pathways, we limited the comparison to 10 predicted genes per pathway for evaluation consistency and efficiency.
- The evaluation metrics include:
  - **Micro F1**: Measures accuracy across all gene-pathway pairs.
  - **Macro F1**: Measures average accuracy per pathway.
  - **# Full Matches**: Number of pathways for which all 10 predicted genes matched the ground truth.

Notably, performance on this task surpassed that of previous generative tasks such as the STRING DB PPI Task. This may be attributed to the more structured and context-specific appearance of pathway names in the literature, compared to the broader and more diffuse mentions of protein names. Domain-specific models such as **Mixtral-8x7B-Instruct**, **BioMedLM**, and **Galactica (30B)** showed particularly strong performance.

---

### KEGG Task – Gene Generation Results

| Model                             | Micro F1 (#shot)    | Macro F1 (#shot)    | # Full Matches (out of 100) |
|----------------------------------|----------------------|----------------------|------------------------------|
| BioGPT-Large (1.5B)              | 0.2435 (3-shot)      | 0.3131 (3-shot)      | 5                            |
| BioMedLM (2.7B)                  | 0.4619 (2-shot)      | 0.5383 (2-shot)      | 22                           |
| Galactica (6.7B)                 | 0.3136 (5-shot)      | 0.3874 (5-shot)      | 8                            |
| Galactica (30B)                  | 0.4609 (5-shot)      | 0.5304 (5-shot)      | 24                           |
| Alpaca (7B)                      | 0.1172 (3-shot)      | 0.1439 (3-shot)      | 4                            |
| RST (11B)                        | 0.1102 (3-shot)      | 0.1238 (3-shot)      | 7                            |
| Falcon (7B)                      | 0.1393 (3-shot)      | 0.1681 (3-shot)      | 5                            |
| Falcon (40B)                     | 0.2004 (3-shot)      | 0.2367 (3-shot)      | 7                            |
| MPT-Chat (7B)                    | 0.1894 (5-shot)      | 0.2482 (5-shot)      | 4                            |
| MPT-Chat (30B)                   | 0.3978 (5-shot)      | 0.4550 (5-shot)      | 18                           |
| LLaMA2-Chat (7B)                 | 0.2936 (5-shot)      | 0.3874 (5-shot)      | 8                            |
| LLaMA2-Chat (70B)                | 0.3098 (5-shot)      | 0.4577 (5-shot)      | 18                           |
| Mistral-Instruct (7B)           | 0.3828 (2-shot)      | 0.4416 (2-shot)      | 19                           |
| **Mixtral-8x7B-Instruct (46B)**  | **0.5962 (2-shot)**  | **0.6479 (2-shot)**  | **39**                       |
| SOLAR-Instruct (10.7B)          | 0.3928 (2-shot)      | 0.4537 (2-shot)      | 19                           |

> **Note**: Bold values indicate the best-performing model in each column.

---

This task highlights the potential of domain-specialized LLMs for accurate biological knowledge extraction when applied to focused, well-defined contexts.


### Evaluating Gene Regulatory Relations

This evaluation assesses the ability of Large Language Models (LLMs) to identify gene regulatory relationships using the INDRA database. The INDRA dataset contains statements extracted from scientific literature that describe gene-gene regulatory interactions. These statements provide rich, contextual information that models must interpret to classify relationships accurately.

---

#### Task Description

- **INDRA DB Gene Regulatory Relation Task**: Models were presented with text snippets and asked to identify the correct gene regulatory relationship between two genes from a set of six options: **Activation, Inhibition, Phosphorylation, Dephosphorylation, Ubiquitination,** and **Deubiquitination**.
- A **multiple-choice format** was used.
- Each class included **500 examples**, totaling **3,000 samples** across six classes.
- Models were evaluated using **Micro F1** and **Macro F1** scores.
- Most evaluations were performed with 1-shot prompting unless otherwise noted.

---

### INDRA Task – Multiple Choice Classification Results

| Model                             | Micro F1 (#shot)    | Macro F1 (#shot)    |
|----------------------------------|----------------------|----------------------|
| BioGPT-Large (1.5B)              | 0.2267 (0-shot)      | 0.1600 (0-shot)      |
| BioMedLM (2.7B)                  | 0.1443 (0-shot)      | 0.1084 (0-shot)      |
| Galactica (6.7B)                 | 0.5593 (1-shot)      | 0.4489 (1-shot)      |
| Galactica (30B)                  | 0.6560 (1-shot)      | 0.5533 (1-shot)      |
| Alpaca (7B)                      | 0.1670 (1-shot)      | 0.0483 (1-shot)      |
| RST (11B)                        | 0.4627 (0-shot)      | 0.4025 (0-shot)      |
| Falcon (7B)                      | 0.1707 (1-shot)      | 0.0557 (1-shot)      |
| Falcon (40B)                     | 0.6503 (1-shot)      | 0.5494 (1-shot)      |
| MPT-Chat (7B)                    | 0.5977 (1-shot)      | 0.5105 (1-shot)      |
| MPT-Chat (30B)                   | 0.6607 (1-shot)      | 0.5737 (1-shot)      |
| LLaMA2-Chat (7B)                 | 0.5767 (1-shot)      | 0.5017 (1-shot)      |
| LLaMA2-Chat (70B)                | 0.6780 (1-shot)      | 0.5906 (1-shot)      |
| Mistral-Instruct (7B)           | 0.6380 (1-shot)      | 0.5571 (1-shot)      |
| **Mixtral-8x7B-Instruct (46B)**  | **0.7553 (1-shot)**  | **0.6436 (1-shot)**  |
| SOLAR-Instruct (10.7B)          | 0.7387 (2-shot)      | 0.6411 (2-shot)      |

> **Note**: Bold values indicate the highest-performing model in each column.

---

These results highlight that **larger and domain-specialized models** outperform smaller, general-purpose ones in understanding and classifying gene regulatory relationships from biomedical text. **Mixtral-8x7B-Instruct (46B)** achieved the best performance, closely followed by **SOLAR-Instruct (10.7B)**. In contrast, models like **Alpaca (7B)** and **Falcon (7B)** demonstrated significant performance limitations, likely due to biased or inconsistent predictions, as also reflected in the confusion matrices (see Appendix F for details).



## Citation
```bibtex
@article{doi:10.1089/cmb.2025.0078,
  title = {Comparative Performance Evaluation of Large Language Models for Extracting Molecular Interactions and Pathway Knowledge},
  author = {Park, Gilchan and Yoon, Byung-Jun and Luo, Xihaier and L\'{o}pez-Marrero, Vanessa and Yoo, Shinjae and Jha, Shantenu},
  journal = {Journal of Computational Biology},
  year = {2025},
  doi = {10.1089/cmb.2025.0078},
  note = {PMID: 40387594},
  url = {https://doi.org/10.1089/cmb.2025.0078},
  eprint = {https://doi.org/10.1089/cmb.2025.0078}
}
```

