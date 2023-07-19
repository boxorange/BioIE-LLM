# BioIE-LLM
Biological Information Extraction from Large Language Models (LLMs)

This is the official code of the papers:
* [Automated Extraction of Molecular Interactions and Pathway Knowledge using Large Language Model, Galactica: Opportunities and Challenges](https://aclanthology.org/2023.bionlp-1.22/)
* [Comparative Performance Evaluation of Large Language Models for Extracting Molecular Interactions and Pathway Knowledge](https://arxiv.org/abs/2307.08813)


## Installation
The code was implemented on Python version 3.8, and the versions of the dependencies are listed in requirements.txt


## Datasets
* [STRING DB](https://string-db.org): the human (Homo sapiens) protein network for performing a protein-protein interaction (PPI) recognition task.
* [KEGG DB](https://www.genome.jp/kegg): the KEGG human pathways which have been identified as being activated in response to low-dose radiation exposure in [a recent study](https://arxiv.org/abs/2301.01769).
* [INDRA DB](http://www.indra.bio): a set of human gene regulatory relation statements that represent mechanistic interactions between biological agents.


## Reproduction
To reproduce the results of the experiments, use the bash script [run.sh](scripts/run.sh). You need to change model/data paths accordingly.


## Results
Here are the results of the experiments. The experiments were conducted on 8×NVIDIA V100 GPUs. Note different number of GPUs and batch size can produce slightly different results.

### Recognizing Protein-Protein Interactions ###
<table>
    <tr>
        <th>Model</th>
        <th>1K proteins</th>
		<th>1K protein pairs</th>
    </tr>
	<tr>
        <th>Galactica (6.7B)</th>
        <th>LLaMA (7B)</th>
        <th>Alpaca (7B)</th>
        <th>RST (11B)</th>
        <th>BioGPT-Large (1.5B)</th>
        <th>BioMedLM (2.7B)</th>
    </tr>
	<tr>
        <td>0.166</td>
        <td>0.043</td>
        <td>0.052</td>
        <td>0.146</td>
        <td>0.100</td>
        <td>0.069</td>
    </tr>
    <tr>
        <td>0.552</td>
        <td>0.484</td>
        <td>0.521</td>
        <td>0.529</td>
        <td>0.504</td>
        <td>0.643</td>
    </tr>
</table>

### KEGG Pathway Recognition ###
<table>
    <tr>
        <th>Model</th>
        <th>Galactica (6.7B)</th>
        <th>LLaMA (7B)</th>
        <th>Alpaca (7B)</th>
        <th>RST (11B)</th>
        <th>BioGPT-Large (1.5B)</th>
        <th>BioMedLM (2.7B)</th>
    </tr>
	<tr>
        <th>Pathways</th>
        <td>0.256</td>
        <td>0.180</td>
        <td>0.268</td>
        <td>0.255</td>
        <td>0.550</td>
        <td>0.514</td>
    </tr>
	<tr>
        <th>Pathways</th>
        <td>0.256</td>
        <td>0.180</td>
        <td>0.268</td>
        <td>0.255</td>
        <td>0.550</td>
        <td>0.514</td>
    </tr>
    <tr>
        <th>1K gene and pathway pairs</th>
        <td>0.564</td>
        <td>0.562</td>
        <td>0.522</td>
        <td>0.514</td>
        <td>0.497</td>
        <td>0.568</td>
    </tr>
</table>

### Evaluating Gene Regulatory Relations ###
<table>
    <tr>
        <th>Model</th>
        <th>Galactica (6.7B)</th>
        <th>LLaMA (7B)</th>
        <th>Alpaca (7B)</th>
        <th>RST (11B)</th>
        <th>BioGPT-Large (1.5B)</th>
        <th>BioMedLM (2.7B)</th>
    </tr>
	<tr>
        <th>2 class</th>
        <td>0.704</td>
        <td>0.351</td>
        <td>0.736</td>
        <td>0.640</td>
        <td>0.474</td>
        <td>0.542</td>
    </tr>
    <tr>
        <th>3 class</th>
        <td>0.605</td>
        <td>0.293</td>
        <td>0.645</td>
        <td>0.718</td>
        <td>0.390</td>
        <td>0.408</td>
    </tr>
	<tr>
        <th>4 class</th>
        <td>0.567</td>
        <td>0.254</td>
        <td>0.556</td>
        <td>0.597</td>
        <td>0.293</td>
        <td>0.307</td>
    </tr>
	<tr>
        <th>5 class</th>
        <td>0.585</td>
        <td>0.219</td>
        <td>0.636</td>
        <td>0.667</td>
        <td>0.328</td>
        <td>0.230</td>
    </tr>
	<tr>
        <th>6 class</th>
        <td>0.597</td>
        <td>0.212</td>
        <td>0.535</td>
        <td>0.614</td>
        <td>0.288</td>
        <td>0.195</td>
    </tr>
</table>


## Citation
```bibtex
@inproceedings{park2023automated,
  title={Automated Extraction of Molecular Interactions and Pathway Knowledge using Large Language Model, Galactica: Opportunities and Challenges},
  author={Park, Gilchan and Yoon, Byung-Jun and Luo, Xihaier and Lpez-Marrero, Vanessa and Johnstone, Patrick and Yoo, Shinjae and Alexander, Francis},
  booktitle={The 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks},
  pages={255--264},
  year={2023}
}
@inproceedings{Park2023ComparativePE,
  title={Comparative Performance Evaluation of Large Language Models for Extracting Molecular Interactions and Pathway Knowledge},
  author={Gilchan Park and Byung-Jun Yoon and Xihaier Luo and Vanessa L'opez-Marrero and Patrick Johnstone and Shinjae Yoo and Francis J. Alexander},
  year={2023}
}
```

