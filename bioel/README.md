# BioEL: A comprehensive package for training, evaluating, and benchmarking biomedical entity linking models.

## Citation

If you have found our manuscript useful in your work, please consider citing:

> @inproceedings{
anonymous2025bioel,
title={Bio{EL}: A Comprehensive Python Package for Biomedical Entity Linking},
author={Anonymous},
booktitle={The 2025 Annual Conference of the Nations of the Americas Chapter of the ACL},
year={2025},
url={https://openreview.net/forum?id=IzF83RNr3d}
}

## Installation

```bash
conda create -n bioel python=3.9
conda activate bioel
pip install bioel

python3 -m pip install pip==24.0

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

pip install numpy==1.23
```

Biobart and Biogenel require transformers version 4.44.2. This specific version of transformers depends on functionality available in numpy version 1.23.

Currently, installing `fairseq` requires cloning the repository directly, as we encountered an issue when attempting installation via pip. We have reported the issue to the `fairseq` team, and once it's resolved, we will update our setup accordingly.

## Development Instructions

1. Install it as an editable package using `pip` as shown above.
1. Add any new dependencies to `setup.py`.
1. Add tests to `tests/` directory.

## Ontologies
Ontologies included in the package : 

- UMLS : UMLS is licensed by the National Library of Medicine and requires a free account to download. You can sign up for an account at https://uts.nlm.nih.gov/uts/signup-login. Once your account has been approved, you can download the UMLS metathesaurus at https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html.

- MeSH : MeSH is derived from UMLS

- NCBI Gene (Entrez) ontololgy. It can be downloaded at https://ftp.ncbi.nih.gov/gene/DATA/
(gene_info.gz)

- MEDIC (MErged DIsease voCabulary) : It can be downloaded at https://ctdbase.org/downloads/

## Resolving abbreviations
As a preprocessing step, we resolve abbreviations in the text using Ab3P, an abbreviation detector created for biomedical text. We ran abbreviation detection on the text of all documents in our benchmark, the results of which are stored in a large dictionary in `data/abbreviations.json`. In order to reproduce our abbreviation detection/resolution pipeline, please run the following:

```
from bioel.utils.solve_abbreviation.solve_abbreviations import create_abbrev
create_abbrev(output_dir, all_dataset)
# output_path : path where to create abbreviations.json
# all_dataset : datasets for which you want the abbreviations.
```

## Example Usage: Model Training, Inference, and Evaluation
```
# Import modules
from bioel.model import BioEL_Model
from bioel.evaluate import Evaluate

# 1) load model
# Look at data/params.json for more information about the config file
krissbert = BioEL_Model.load_krissbert(
        name="krissbert",
        params_file='path/to/params_krissbert.json", # config file
        # checkpoint_path="path/to/checkpoint" # if you have an already trained model
    )

# 2) Training and inference
krissbert.training() # Train
krissbert.inference() # Inference

abbreviations_path = "data/abbreviations.json"
dataset_names = ["ncbi_disease"]
model_names = ["krissbert"]
path_to_result = {
    "ncbi_disease": {
        "krissbert": "results/ncbi_disease.json"
    }
}
eval_strategies=["basic"]

# 3) Results
evaluator = Evaluate(dataset_names=dataset_names, 
                     model_names=model_names, 
                     path_to_result=path_to_result, 
                     abbreviations_path=abbreviations_path, 
                     eval_strategies=eval_strategies,
                     max_k=10, # number of candidates to consider
                     )
evaluator.load_results()
evaluator.process_datasets()
evaluator.evaluate()
evaluator.plot_results()
evaluator.detailed_results()
```
1. Load the Model: Use the `BioEL_Model` class to load your model. Ensure you have a configuration file (params.json) ready.
2. Training and Inference: Perform training and inference using the training() and inference() methods.
3. Evaluation:
        Run evaluations for all models across all datasets using the `Evaluate` class.
        For error analysis with hit index details, use the `evaluator.error_analysis_dfs` attribute.
        For detailed performance metrics such as failure stage breakdown, accuracy per type, recall@k per type, MAP@k, and statistical significance (p-values), refer to the `evaluator.detailed_results_analysis` attribute.

## Config files
Example of configuration files for the various models are provided in the `data/` directory. These can serve as references for users to create or modify their own configuration files.

## Load the different datasets
```
from bioel.dataset import Dataset
abbreviations_path = "data/abbreviations.json"
dataset_name = "bc5cdr" # Specify the desired dataset name from the BigBio collection here.
dataset = Dataset(
    dataset_name = dataset_name, abbreviations_path=abbreviations_path
)

```


## Load the different ontologies

```
from bioel.ontology import BiomedicalOntology

##### ----------------- Load medic ----------------- #####

dataset_name = 'ncbi_disease'
medic_dict = {"name" : "medic",
            "filepath" : "path/to/medic"} # medic.tsv file

ontology = BiomedicalOntology.load_medic(**medic_dict)

##### ----------------- Load entrez ----------------- #####

dataset_name = "gnormplus" # or "nlm_gene"

entrez_dict = {"name" : "entrez",
             "filepath" : "path/to/entrez", # gene_info.tsv file
             "dataset" : f"{dataset_name}",}
ontology = BiomedicalOntology.load_entrez(**entrez_dict)

##### ----------------- Load MESH ----------------- #####

dataset_name = "nlmchem"
mesh_dict = {"name" : "mesh",
             "filepath" : "path/to/umls"}
ontology = BiomedicalOntology.load_mesh(**mesh_dict)

##### ----------------- Load UMLS (st21pv subset) ----------------- #####

dataset_name = "medmentions_st21pv"
umls_dict_st21pv = {
    "name": "umls",
    "filepath": "path/to/umls",
    "path_st21pv_cui": "data/umls_cuis_st21pv.json",
}
ontology = BiomedicalOntology.load_umls(**umls_dict_st21pv)

##### ----------------- Load UMLS (full) ----------------- #####

dataset_name = "medmentions_full"
umls_dict = {
    "name": "umls",
    "filepath": "path/to/umls",
}
ontology = BiomedicalOntology.load_umls(**umls_dict)
```


## ArboEL


ArboEL operates in two stages: First, you need to train the biencoder (`load_arboel_biencoder`). Then, you use the candidate results from the biencoder to train the crossencoder (`load_arboel_crossencoder`) and perform evaluation with the crossencoder.

## SapBERT

To train SapBERT, the first step is to generate the aliases, which can be done using the following method:

```
# Example with medic ontology
from bioel.models.sapbert.data.data_processing import cuis_to_aliases
from bioel.ontology import BiomedicalOntology

ontology = BiomedicalOntology.load_medic(
    filepath="path/to/medic", name="medic"
)

cuis_to_aliases(
    ontology= ontology, # ontology object
    save_dir="path/to/alias.txt", # path where to save the aliases
    dataset_name="ncbi_disease", # name of the dataset
)
```

## BioBART/BioGenEL

BioBART and BioGenEL share the same entity linking module: 
- In order to finetune from BioBART set the `model_load_path` parameter in the .json config file to `GanjinZero/biobart-v2-large`, it will load the pretrained weights from HuggingFace.
  
- In order the finetune from BioGenEL's Knowledge base guided pretrained weights, you first must download the pretrained weights from this link: https://drive.google.com/file/d/1TqvQRau1WPYE9hKfemKZr-9ptE-7USAH/view?usp=sharing and then set the `model_load_path` parameter in the .json config file to the path where you stored the pretrained weights.

There is only one config file to handle the training and the evaluation for BioBART/BioGenEL. You cannot launch the training and the evaluation in the same time, thus first set all the parameters in the config file for both the evaluation and the training. When you run training set `model_load_path` as described above.
Some important information:
- if `model_load_path` is not provided, by default the dataset folder will be `biogenel/data/`
- set `preprocess_data` to `true` during the first training for a given dataset or ontology to generate the preprocessed data. If you want to train with different parameters while using the same preprocessed data you can set it to `false`. During evaluation set it to `false`.
- `trie_path` and `dict_path` will be set to `data/abbreviationmode/datasetname/` folder for a given `data` folder. Thus, if `resolve_abbrevs` is set to `true`, make sure to include `/abbr_res/datasetname/trie.pkl` as a path extension for the trie and `/abbr_res/datasetname/target_kb.json` for the dict.
- For evaluation set `evaluation` to `true` and change the `model_load_path` to the path you provided to `model_save_path` during training. 

