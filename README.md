# BioIE-LLM

**Biological Information Extraction from Large Language Models (LLMs)**

This is the official code repository for the following paper:

- [Comparative Performance Evaluation of Large Language Models for Extracting Molecular Interactions and Pathway Knowledge](https://arxiv.org/abs/2307.08813)

For the earlier study and preliminary results, please visit the companion repository: [BioIE-LLM-Prelim](https://github.com/boxorange/BioIE-LLM-Prelim), which corresponds to the paper:

- [Automated Extraction of Molecular Interactions and Pathway Knowledge using Large Language Model, Galactica: Opportunities and Challenges](https://aclanthology.org/2023.bionlp-1.22/)


## Installation
The code was implemented on Python version 3.8, and the versions of the dependencies are listed in requirements.txt


## Datasets
* [STRING DB](https://string-db.org): the human (Homo sapiens) protein network for performing a protein-protein interaction (PPI) recognition task.
* [KEGG DB](https://www.genome.jp/kegg): the KEGG human pathways which have been identified as being activated in response to low-dose radiation exposure in [a recent study](https://arxiv.org/abs/2301.01769).
* [INDRA DB](http://www.indra.bio): a set of human gene regulatory relation statements that represent mechanistic interactions between biological agents.


## Reproduction
To reproduce the results of the experiments, use the bash script [run.sh](scripts/run.sh). You need to change model/data paths accordingly.


## Results
Here are the results of the experiments. The experiments were conducted on 4×NVIDIA A100 80GB GPUs. Note different number of GPUs and batch size can produce slightly different results.

# Recognizing Protein-Protein Interactions with LLMs

This document presents the evaluation results of various Large Language Models (LLMs) on tasks involving the recognition of protein-protein interactions (PPIs) using data from the STRING and Negatome databases.

---

## STRING DB PPI Task (Generative Question)

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

## STRING/Negatome DB PPI Task (Yes/No Classification)

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



### KEGG Pathway Recognition ###
* KEGG Task1 - Precision for the generated genes that belong to the top 20 pathways relevant to low-dose radiation exposure.
* KEGG Task2 - Micro F-scores for randomly selected positive and negative pairs (I.e., 1K = 500 pos + 500 neg).
* Model prediction consistency between Task1 and Task2.
<table>
    <tr>
        <th>Model</th>
        <th>KEGG Task1</th>
		<th>KEGG Task2</th>
		<th>Consistency</th>
    </tr>
	<tr>
        <th>Galactica (6.7B)</th>
		<td>0.256</td>
		<td>0.564</td>
		<td>0.917</td>
    </tr>
	<tr>
        <th>LLaMA (7B)</th>
		<td>0.180</td>
		<td>0.562</td>
		<td>0.881</td>
	</tr>
	<tr>
        <th>Alpaca (7B)</th>
		<td>0.268</td>
		<td>0.522</td>
		<td>1.0</td>
	</tr>
	<tr>
        <th>RST (11B)</th>
		<td>0.255</td>
		<td>0.514</td>
		<td>0.0</td>
	</tr>
	<tr>
        <th>BioGPT-Large (1.5B)</th>
		<td>0.550</td>
		<td>0.497</td>
		<td>0.923</td>
	</tr>
	<tr>
        <th>BioMedLM (2.7B)</th>
		<td>0.514</td>
		<td>0.568</td>
		<td>0.821</td>
	</tr>
</table>

### Evaluating Gene Regulatory Relations ###
* INDRA Task - Micro F-scores with 1K samples for each class.
<table>
    <tr>
        <th>Model</th>
        <th>2 class</th>
        <th>3 class</th>
        <th>4 class</th>
        <th>5 class</th>
        <th>6 class</th>
    </tr>
	<tr>
        <th>Galactica (6.7B)</th>
        <td>0.704</td>
        <td>0.605</td>
        <td>0.567</td>
        <td>0.585</td>
        <td>0.597</td>
    </tr>
    <tr>
        <th>LLaMA (7B)</th>
        <td>0.351</td>
        <td>0.293</td>
        <td>0.254</td>
        <td>0.219</td>
        <td>0.212</td>
    </tr>
	<tr>
        <th>Alpaca (7B)</th>
        <td>0.736</td>
        <td>0.645</td>
        <td>0.556</td>
        <td>0.636</td>
        <td>0.535</td>
    </tr>
	<tr>
        <th>RST (11B)</th>
        <td>0.640</td>
        <td>0.718</td>
        <td>0.597</td>
        <td>0.667</td>
        <td>0.614</td>
    </tr>
	<tr>
        <th>BioGPT-Large (1.5B)</th>
        <td>0.474</td>
        <td>0.390</td>
        <td>0.293</td>
        <td>0.328</td>
        <td>0.288</td>
    </tr>
	<tr>
        <th>BioMedLM (2.7B)</th>
        <td>0.542</td>
        <td>0.408</td>
        <td>0.307</td>
        <td>0.230</td>
        <td>0.195</td>
    </tr>
</table>


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

