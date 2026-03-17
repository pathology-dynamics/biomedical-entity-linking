import pickle
import ujson
import os

import csv

import numpy as np

from tqdm import tqdm
from typing import Optional

from bioel.utils.umls_utils import UmlsMappings
from bioel.utils.bigbio_utils import (
    CUIS_TO_REMAP,
    CUIS_TO_EXCLUDE,
    VALIDATION_DOCUMENT_IDS,
    dataset_to_documents,
    dataset_to_df,
)

from bioel.utils.bigbio_utils import (
    load_bigbio_dataset,
    resolve_abbreviation,
    add_deabbreviations,
)
from bioel.ontology import BiomedicalOntology

def get_type_gcd(types, type2geneology, cached_types: dict = None):
    """
    Find the most granular single parent type for an entity with multiple types
    If an entity is assigned multiple types with disjoint type hierarchies
    """
    # Initialize cached_types if it is None
    if cached_types is None:
        cached_types = {}

    if len(types) == 1:
        t = types[0]
        if t not in cached_types:
            cached_types[t] = type2geneology[t][-1]
        return cached_types[t]

    type_tuple = tuple(sorted(types))
    if type_tuple in cached_types:
        return cached_types[type_tuple]
    else:
        geneologies = [type2geneology[t] for t in types]
        min_len = min([len(x) for x in geneologies])
        arr = np.array([gen[:min_len] for gen in geneologies])
        unique_types = np.unique(arr[:, 0])
        if len(unique_types) > 1:
            output_types = ", ".join(list(set([x[-1] for x in geneologies])))
            cached_types[type_tuple] = output_types
            return output_types
        else:
            for i in np.arange(1, arr.shape[1]):
                curr_unique = np.unique(arr[:, i])

                if len(curr_unique) > 1 or i >= 3:
                    cached_types[type_tuple] = unique_types[0]
                    return unique_types[0]
                else:
                    unique_types = curr_unique


def process_ontology(
    ontology: BiomedicalOntology, data_path: str, tax2name_filepath: str = None
):
    """
    This function prepares the entity data : dictionary.pickle

    Parameters
    ----------
    - ontology : str (only umls for now)
        Ontology associated with the dataset
    - data_path : str
        Path where to load and save dictionary.pickle
    - tax2name_filepath : str
        Path to the taxonomy to name file
    """

    # Check if equivalent CUIs are present for the first entity
    first_entity_cui = next(iter(ontology.entities))
    equivalant_cuis = bool(ontology.entities[first_entity_cui].equivalant_cuis)
    print("equivalant cuis :", equivalant_cuis)
    "If dictionary already processed, load it else process and load it"
    entity_dictionary_pkl_path = os.path.join(data_path, "dictionary.pickle")

    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, "rb") as read_handle:
            entities = pickle.load(read_handle)

        return entities, equivalant_cuis

    if tax2name_filepath:
        with open(tax2name_filepath, "r") as f:
            tax2name = ujson.load(f)

    ontology_entities = []
    for cui, entity in tqdm(ontology.entities.items()):
        new_entity = {}
        
        if ontology.name.lower() in ["umls"]:
            with open(os.path.join(data_path, "tui2type_hierarchy.json"), "r") as f:
                type2geneology = ujson.load(f)
            entity.types = get_type_gcd(entity.types, type2geneology)

        new_entity["cui"] = entity.cui
        new_entity["title"] = entity.name
        new_entity["types"] = f"{entity.types}"

        if entity.aliases:
            if entity.definition:
                if entity.taxonomy:
                    new_entity["description"] = (
                        f"{entity.name} ( {tax2name[str(entity.taxonomy)]}, {entity.types} : {entity.aliases} ) [{entity.definition}]"
                    )

                else:
                    new_entity["description"] = (
                        f"{entity.name} ( {entity.types} : {entity.aliases} ) [{entity.definition}]"
                    )

            else:
                if entity.taxonomy:
                    new_entity["description"] = (
                        f"{entity.name} ( {tax2name[str(entity.taxonomy)]}, {entity.types} : {entity.aliases} )"
                    )
                else:
                    new_entity["description"] = (
                        f"{entity.name} ( {entity.types} : {entity.aliases} )"
                    )

        else:
            if entity.definition:
                if entity.taxonomy:
                    new_entity["description"] = (
                        f"{entity.name} ( {tax2name[str(entity.taxonomy)]}, {entity.types}) [{entity.definition}]"
                    )

                else:
                    new_entity["description"] = (
                        f"{entity.name} ( {entity.types}) [{entity.definition}]"
                    )
            else:
                if entity.taxonomy:
                    new_entity["description"] = (
                        f"{entity.name} ( {tax2name[str(entity.taxonomy)]}, {entity.types})"
                    )
                else:
                    new_entity["description"] = f"{entity.name} ({entity.types})"

        if hasattr(entity, "metadata") and entity.metadata:
            new_entity["description"] += f" {entity.metadata}"

        if equivalant_cuis:
            new_entity["cuis"] = entity.equivalant_cuis

        ontology_entities.append(new_entity)

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Save entities to pickle file
    with open(os.path.join(data_path, "dictionary.pickle"), "wb") as f:
        pickle.dump(ontology_entities, f)

    entities = pickle.load(open(os.path.join(data_path, "dictionary.pickle"), "rb"))
    return entities, equivalant_cuis


# Function for preparing the mentions in the dataset into the right format for each model
def process_mention_dataset(
    ontology: BiomedicalOntology,
    dataset: str,
    data_path: str,
    path_to_abbrev=None,
    tax2name_filepath=None,
    fold_number=None,
):
    """
    This function prepares the mentions data :  Creates the train.jsonl, valid.jsonl, test.jsonl
    Each .jsonl contains data in the following format :
    {'mention': mention,
    'mention_id': ID of the mention, (optional)
    'context_left': context before mention,
    'context_right': context after mention,
    'context_doc_id': ID of the doc, (optional)
    'type': type
    'label_id': label_id,
    'label': entity description, (optional)
    'label_title': entity title}

    Parameters
    ----------
    - ontology : BiomedicalOntology object
    Ontology associated with the dataset
    - dataset : str
    Name of the dataset
    - data_path : str
    Path where to load and save dictionary.pickle
    - tax2name_filepath : str
    Path to the taxonomy to name file
    """
    data = load_bigbio_dataset(dataset_name=dataset, fold_number=fold_number)
    exclude = CUIS_TO_EXCLUDE[dataset]
    remap = CUIS_TO_REMAP[dataset]

    if path_to_abbrev:
        data = add_deabbreviations(dataset=data, path_to_abbrev=path_to_abbrev)

    entities, equivalant_cuis = process_ontology(ontology, data_path, tax2name_filepath)

    entity_dictionary = {d["cui"]: d for d in tqdm(entities)}  # CC1

    # For ontology with multiples cuis
    if equivalant_cuis:
        # Need to redo this since we have multiple synonymous CUIs for ncbi_disease
        entity_dictionary = {cui: d for d in tqdm(entities) for cui in d["cuis"]}
        cui_synsets = {}
        for subdict in tqdm(entities):
            for cui in subdict["cuis"]:
                if cui in subdict:
                    print(cui, cui_synsets[cui], subdict["cuis"])
                cui_synsets[cui] = subdict["cuis"]
        with open(os.path.join(data_path, "cui_synsets.json"), "w") as f:
            f.write(ujson.dumps(cui_synsets, indent=2))

    if dataset in VALIDATION_DOCUMENT_IDS:
        validation_pmids = VALIDATION_DOCUMENT_IDS[dataset]
    else:
        print("ERROR!!!")

    # Convert BigBio dataset to pandas DataFrame
    df = dataset_to_df(
        data,
        entity_remapping_dict=remap,
        cuis_to_exclude=exclude,
        val_split_ids=validation_pmids,
    )
    # Return dictionary of documents in BigBio dataset
    docs = dataset_to_documents(data, dataset_name=dataset)
    label_len = df["db_ids"].map(lambda x: len(x)).max()
    print("Max labels on one doc:", label_len)
    abbrev_dict = ujson.load(open(path_to_abbrev)) if path_to_abbrev else None

    for split in df.split.unique():
        ents_in_split = []
        for d in tqdm(
            df.query("split == @split").to_dict(orient="records"),
            desc=f"Creating correct mention format for {split} dataset",
        ):
            abbrev_resolved = False
            offsets = d["offsets"]
            doc_id = d["document_id"]
            doc = docs[doc_id]
            mention = d["text"]

            # Resolve abbreviaions if desired
            if abbrev_dict is not None:
                deabbreviated_mention = resolve_abbreviation(
                    doc_id, mention, abbrev_dict
                )
                abbrev_resolved = True

            # Get offsets and context
            start = offsets[0][0]  # start on the mention
            end = offsets[-1][-1]  # end of the mention
            before_context = doc[:start]  # left context
            after_context = doc[end:]  # right context

            # ArboEL can't handle multi-labels, so we randomly choose one.
            if len(d["db_ids"]) == 1:
                label_id = d["db_ids"][0]

            # For ontology with multiples cuis
            elif equivalant_cuis:
                labels = []
                used_cuis = set([])
                choosable_ids = []
                for db_id in d["db_ids"]:
                    if db_id in used_cuis:
                        continue
                    else:
                        used_cuis.update(set(entity_dictionary[db_id]["cuis"]))
                    choosable_ids.append(db_id)

                label_id = np.random.choice(choosable_ids)

            else:
                label_id = np.random.choice(d["db_ids"])

            # Check if we missed something
            if label_id not in entity_dictionary:
                # print(label_id)
                continue

            output = [
                {
                    "mention": mention,
                    "mention_id": d["mention_id"],
                    "context_left": before_context,
                    "context_right": after_context,
                    "context_doc_id": doc_id,
                    "type": d["type"][0],
                    "label_id": label_id,
                    "label": entity_dictionary[label_id]["description"],
                    "label_title": entity_dictionary[label_id]["title"],
                }
            ]

            if abbrev_resolved:
                output = [
                    {
                        "mention": deabbreviated_mention,
                        "mention_id": d["mention_id"] + ".abbr_resolved",
                        "context_left": before_context,
                        "context_right": after_context,
                        "context_doc_id": doc_id,
                        "type": d["type"][0],
                        "label_id": label_id,
                        "label": entity_dictionary[label_id]["description"],
                        "label_title": entity_dictionary[label_id]["title"],
                    }
                ]
                output.append(
                    {
                        "mention": deabbreviated_mention,
                        "mention_id": d["mention_id"] + ".abbr_resolved",
                        "context_left": before_context,
                        "context_right": after_context,
                        "context_doc_id": doc_id,
                        "type": d["type"][0],
                        "label_id": label_id,
                        "label": entity_dictionary[label_id]["description"],
                        "label_title": entity_dictionary[label_id]["title"],
                    }
                )

            ents_in_split.extend(output)

        split_name = split
        if split == "validation":
            split_name = "valid"
        with open(os.path.join(data_path, f"{split_name}.jsonl"), "w") as f:
            f.write("\n".join([ujson.dumps(x) for x in ents_in_split]))
