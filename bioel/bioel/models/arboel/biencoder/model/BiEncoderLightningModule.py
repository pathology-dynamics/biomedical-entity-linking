import os
import json
import ujson
import random
import time
import pickle
import logger
import numpy as np
import torch
import math
import lightning as L
import torch.distributed as dist
from transformers import AutoModel
from pytorch_transformers.optimization import WarmupLinearSchedule
from datetime import datetime
from typing import Optional, Union

from scipy.sparse.csgraph import minimum_spanning_tree

# csgraph = compressed sparse graph
from scipy.sparse import csr_matrix

# csr_matrix = compressed sparse row matrices
from collections import Counter

import bioel.models.arboel.biencoder.data.data_process as data_process
import bioel.models.arboel.biencoder.model.eval_cluster_linking as eval_cluster_linking
from bioel.models.arboel.biencoder.model.biencoder import BiEncoderRanker
from bioel.models.arboel.biencoder.model.common.optimizer import get_bert_optimizer
from bioel.models.arboel.biencoder.model.common.params import BlinkParser
from bioel.models.arboel.biencoder.model.eval_cluster_linking import (
    filter_by_context_doc_id,
)
from bioel.models.arboel.biencoder.model.special_partition.special_partition import (
    cluster_linking_partition,
)
from bioel.utils.bigbio_utils import (
    load_bigbio_dataset,
    dataset_to_df,
    add_deabbreviations,
    CUIS_TO_REMAP,
    CUIS_TO_EXCLUDE,
)

from IPython import embed
from tqdm import tqdm

import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from bioel.models.arboel.biencoder.data.BiEncoderLightningDataModule import (
    ArboelDataModule,
)


def create_versioned_filename(base_path, base_name, extension=".json"):
    """Create a versioned filename to avoid overwriting existing files.
    Params:
    - base_path (str):
        The directory where the file will be saved.
    base_name (str):
        The base name of the file without the extension.
    extension (str):
        The file extension.
    Returns:
        str: A versioned filename that does not exist in the base path.
    """
    version = 1
    while True:
        new_filename = f"{base_name}{version}{extension}"
        full_path = os.path.join(base_path, new_filename)
        if not os.path.exists(full_path):
            return full_path
        version += 1


def evaluate(
    params,
    logger,
    entity_data,
    query_data,
    valid_men_embeds,
    nn_ent_dists,
    nn_ent_idxs,
    nn_men_dists,
    nn_men_idxs,
    n_entities,
    n_mentions,
    max_knn,
    use_types=False,
    within_doc=False,
    context_doc_ids=None,
):
    """
    Description
    -----------
    The first 2 steps are done are done at the beginning of each validation epoch.
    1) Computes embeddings and faiss indexes for entities and mentions.
    2) Performs k-nearest neighbors (k-NN) search to establish relationships between them.
    3) Constructs graphs based on these relationships.
    4) Evaluates the model's accuracy by analyzing how effectively the model can link mentions to the correct entities.

    Parameters
    ----------
    logger : 'Logger' object
        Logging object used to record messages
    entity_data : list of dict
        Contains informations (type, cui, description, tokens, ids) on entities
    query_data : list of dict
        Contains information about mentions (mention_id, mention_name, context, label_idxs, etc…) of validation set
    use_types=False : bool
        A boolean flag that indicates whether or not to use type-specific indexes for entities and mentions
    valid_men_embeds : tensor of dim = (nb of entities, entities length = nb of tokens = 64)
        Contains the embedding of the entity token ID vectors
    nn_ent_dists : float
        Distance of the closest entity for all mentions in "valid_men_embeds"
    nn_ent_idxs : int
        Indice of the closest entity for all mentions in "valid_men_embeds"
    nn_men_dists : list of float
        Distance of the k closest mentions for all mentions in "valid_men_embeds"
    nn_men_idxs : list of int
        Indice of the k closest mentions for all mentions in "valid_men_embeds"
    n_entities : int
        Total number of entities
    n_mentions : int
        Total number of mentions in the validation set.
    max_knn : int
        max number of knn
    within_doc=False : bool
        Boolean flag that indicates whether the evaluation should be constrained to within-document contexts
    context_doc_ids=None : bool
        This would be used in conjunction with within_doc to limit evaluations within the same document.
    """

    joint_graphs = (
        {}
    )  # Store results of the NN search and distance between entities and mentions

    knn_vals = [0] + [2**i for i in range(int(math.log(params["knn"], 2)) + 1)]

    for k in knn_vals:
        joint_graphs[k] = {
            "rows": np.array([]),
            "cols": np.array([]),
            "data": np.array([]),
            "shape": (n_entities + n_mentions, n_entities + n_mentions),
        }

    "3) Constructs graphs based on these relationships."
    """
    nn_ent_dists contain information about distance of the closest entity
    nn_ent_idxs contain information about indice of the closest entity
    nn_men_dists contain information about distance of the k nearest mentions
    nn_men_idxs contain information about indice of the k nearest mentions
    - We can fill in the "rows" part (=start nodes) of the graph in the order of the mentions
    - We can fill in the "cols" part (=end nodes) of the graph with nn_ent_idxs and nn_men_idxs
    - We can fill in the "data" part (=weights) of the graph with nn_ent_dists and nn_men_dists
    """
    logger.info("Eval: Building graphs")
    for men_query_idx, men_embed in enumerate(
        tqdm(
            valid_men_embeds, total=len(valid_men_embeds), desc="Eval: Building graphs"
        )
    ):
        # Get nearest entity candidate
        dict_cand_idx = nn_ent_idxs[men_query_idx][
            0
        ]  # Use of [0] to retrieve a scalar and not an 1D array
        dict_cand_score = nn_ent_dists[men_query_idx][0]

        # Filter candidates to remove -1s, mention query, within doc (if reqd.), and keep only the top k candidates
        filter_mask_neg1 = (
            nn_men_idxs[men_query_idx] != -1
        )  # bool ndarray. Ex : np.array([True, False, True, False])
        men_cand_idxs = nn_men_idxs[men_query_idx][
            filter_mask_neg1
        ]  # Only keep the elements != -1
        men_cand_scores = nn_men_dists[men_query_idx][filter_mask_neg1]

        if within_doc:
            men_cand_idxs, wd_mask = filter_by_context_doc_id(
                mention_idxs=men_cand_idxs,
                doc_id=context_doc_ids[men_query_idx],
                doc_id_list=context_doc_ids,
                return_numpy=True,
            )
            men_cand_scores = men_cand_scores[wd_mask]

        # Filter self-reference + limits the number of candidate to 'max_knn'
        filter_mask = men_cand_idxs != men_query_idx
        men_cand_idxs, men_cand_scores = (
            men_cand_idxs[filter_mask][:max_knn],
            men_cand_scores[filter_mask][:max_knn],
        )

        # Add edges to the graphs
        for k in joint_graphs:
            joint_graph = joint_graphs[k]
            # Add mention-entity edge
            joint_graph["rows"] = (
                np.append(  # Mentions are offset by the total number of entities to differentiate mention nodes from entity nodes
                    joint_graph["rows"], [n_entities + men_query_idx]
                )
            )
            joint_graph["cols"] = np.append(joint_graph["cols"], dict_cand_idx)
            joint_graph["data"] = np.append(joint_graph["data"], dict_cand_score)
            if k > 0:
                # Add mention-mention edges
                joint_graph["rows"] = np.append(
                    joint_graph["rows"],
                    [n_entities + men_query_idx]
                    * len(
                        men_cand_idxs[:k]
                    ),  # Creates an array where the starting node (current mention) is repeated len(men_cand_idxs[:k]) times
                )
                joint_graph["cols"] = np.append(
                    joint_graph["cols"], n_entities + men_cand_idxs[:k]
                )
                joint_graph["data"] = np.append(
                    joint_graph["data"], men_cand_scores[:k]
                )

    "4) Evaluates the model's accuracy by analyzing how effectively the model can link mentions to the correct entities."
    dict_acc = {}
    max_eval_acc = -1.0
    for k in joint_graphs:
        logger.info(f"\nEval: Graph (k={k}):")
        # Partition graph based on cluster-linking constraints (inference procedure)
        partitioned_graph, clusters = eval_cluster_linking.partition_graph(
            joint_graphs[k], n_entities, directed=True, return_clusters=True
        )
        # Infer predictions from clusters
        result = eval_cluster_linking.analyzeClusters(
            clusters, entity_data, query_data, k
        )
        acc = float(result["accuracy"].split(" ")[0])
        dict_acc[f"k{k}"] = acc
        max_eval_acc = max(acc, max_eval_acc)
        logger.info(f"Eval: accuracy for graph@k={k}: {acc}%")
    logger.info(f"Eval: Best accuracy: {max_eval_acc}%")
    return max_eval_acc, dict_acc


def evaluate_test(
    params,
    reranker,
    test_dict_vecs,
    test_processed_data,
    test_men_vecs,
    logger,
    entity_data,
    train_processed_data=None,
    train_men_vecs=None,
    use_types=False,
    embed_batch_size=768,
    force_exact_search=False,
    probe_mult_factor=1,
    within_doc=False,
    context_doc_ids=None,
):
    """
    Description : Specific evaluation for test set (can control the number of recall k + save data)
    -----------
    1) Computes embeddings and indexes for entities and mentions.
    2) Performs k-nearest neighbors (k-NN) search to establish relationships between them.
    3) Constructs graphs based on these relationships.
    4) Evaluates the model's accuracy by analyzing how effectively the model can link mentions to the correct entities.

    Parameters
    ----------
    reranker : BiEncoderRanker
        NN-based ranking model
    test_dict_vec : list of tensors
        Contains the IDs of the entities (title + description) tokens
    test_processed_data : list of dict
        Contains information about mentions (mention_id, mention_name, context, etc…) of test set
    test_men_vecs : list of tensors
        Contains only the IDs of (mention + surrounding context) tokens of test set
    logger : 'Logger' object
        Logging object used to record messages
    entity_data : list or dict
        Entities from the data
    train_processed_data : list of dict
        Contains information about mentions (mention_id, mention_name, context, etc…) of train set
    train_men_vecs : list of tensors
        Contains only the IDs of (mention + surrounding context) tokens of train set
    use_types=False : bool
        A boolean flag that indicates whether or not to use type-specific indexes for entities and mentions
    embed_batch_size=128 : int
        The batch size to use when processing embeddings.
    force_exact_search=False : bool
        force the embedding process to use exact search methods rather than approximate methods.
    probe_mult_factor=1 : int
        A multiplier factor used in index building for probing in case of approximate search (bigger = better but slower)
    within_doc=False : bool
        Boolean flag that indicates whether the evaluation should be constrained to within-document contexts
    context_doc_ids=None : bool
        This would be used in conjunction with within_doc to limit evaluations within the same document.
    """
    print("Number of test mentions:", len(test_men_vecs))

    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    embed_data_path = params["embed_data_path"]
    if embed_data_path is None or not os.path.exists(embed_data_path):
        embed_data_path = output_path

    knn = params["knn"]

    n_entities = len(test_dict_vecs)
    n_mentions = len(test_men_vecs)  ## Change

    if params["transductive"]:
        n_mentions += len(train_men_vecs)
        n_train_mentions = len(train_men_vecs)

    # Values of k to run the evaluation against
    knn_vals = [0] + [2**i for i in range(int(math.log(knn, 2)) + 1)]
    # Store the maximum evaluation k
    max_knn = knn_vals[-1]

    # Maps cui to unique identifier
    dict_cui_to_idx = {}
    for idx, ent in enumerate(entity_data):
        dict_cui_to_idx[ent["cui"]] = idx
    # Maps cui to unique identifier : Used for retrieving cui of top candidates
    dict_idx_to_cui = {v: k for k, v in dict_cui_to_idx.items()}

    # Needed for the output for evaluation
    data = load_bigbio_dataset(params["dataset"], fold_number=params.get("fold_number"))
    exclude = CUIS_TO_EXCLUDE[params["dataset"]]
    remap = CUIS_TO_REMAP[params["dataset"]]
    if params["path_to_abbrev"]:
        data_with_abbrev = add_deabbreviations(
            dataset=data, path_to_abbrev=params["path_to_abbrev"]
        )
        df = dataset_to_df(
            data_with_abbrev, entity_remapping_dict=remap, cuis_to_exclude=exclude
        )
    else:
        df = dataset_to_df(data, entity_remapping_dict=remap, cuis_to_exclude=exclude)

    # Output for evaluation object
    output_eval = []

    time_start = time.time()

    # Check if graphs are already built
    graph_path = os.path.join(output_path, "graphs.pickle")
    if not params["only_recall"] and os.path.isfile(graph_path):
        print("Loading stored joint graphs...")
        with open(graph_path, "rb") as read_handle:
            joint_graphs = pickle.load(read_handle)
    else:
        # Initialize graphs to store mention-mention and mention-entity similarity score edges;
        # Keyed on k, the number of nearest mentions retrieved
        joint_graphs = {}
        for k in knn_vals:
            joint_graphs[k] = {
                "rows": np.array([]),
                "cols": np.array([]),
                "data": np.array([]),
                "shape": (n_entities + n_mentions, n_entities + n_mentions),
            }
        # Check and load stored embedding data
        embed_data = None
        if os.path.isfile(embed_data_path):
            embed_data = torch.load(params["embed_data_path"])

        "1) Computes embeddings and indexes for entities and mentions. "
        if use_types:
            if embed_data is not None:
                logger.info("Dictionary : Loading stored embedding (entities)")
                dict_embeds = embed_data["dict_embeds"]
                if "dict_idxs_by_type" in embed_data:
                    dict_idxs_by_type = embed_data["dict_idxs_by_type"]
                else:
                    dict_idxs_by_type = data_process.get_idxs_by_type(entity_data)
                dict_indexes = data_process.get_index_from_embeds(
                    dict_embeds,
                    dict_idxs_by_type,
                    force_exact_search=params["force_exact_search"],
                    probe_mult_factor=params["probe_mult_factor"],
                )

                logger.info("Queries: Embedding and building index (mentions)")
                vecs = test_men_vecs  # entity_dict_vecs
                men_data = test_processed_data
                if params["transductive"]:
                    vecs = torch.cat((train_men_vecs, vecs), dim=0)
                    men_data = train_processed_data + test_processed_data
                men_embeds, men_indexes, men_idxs_by_type = (
                    data_process.embed_and_index(
                        model=reranker,
                        token_id_vecs=vecs,
                        encoder_type="context",
                        corpus=men_data,
                        force_exact_search=params["force_exact_search"],
                        batch_size=params["embed_batch_size"],
                        probe_mult_factor=params["probe_mult_factor"],
                    )
                )

            else:
                logger.info("Dictionary: Embedding and building index (entities)")
                dict_embeds, dict_indexes, dict_idxs_by_type = (
                    data_process.embed_and_index(
                        model=reranker,
                        token_id_vecs=test_dict_vecs,
                        encoder_type="candidate",
                        corpus=entity_data,
                        force_exact_search=params["force_exact_search"],
                        batch_size=params["embed_batch_size"],
                        probe_mult_factor=params["probe_mult_factor"],
                    )
                )

                logger.info("Queries: Embedding and building index (mentions)")
                vecs = test_men_vecs
                men_data = test_processed_data
                if params["transductive"]:
                    vecs = torch.cat((train_men_vecs, vecs), dim=0)
                    men_data = train_processed_data + test_processed_data
                men_embeds, men_indexes, men_idxs_by_type = (
                    data_process.embed_and_index(
                        model=reranker,
                        token_id_vecs=vecs,
                        encoder_type="context",
                        corpus=men_data,
                        force_exact_search=params["force_exact_search"],
                        batch_size=params["embed_batch_size"],
                        probe_mult_factor=params["probe_mult_factor"],
                    )
                )
        else:
            if embed_data is not None:
                logger.info("Dictionary : Loading stored embedding (entities)")
                dict_embeds = embed_data["dict_embeds"]
                dict_index = data_process.get_index_from_embeds(
                    dict_embeds,
                    force_exact_search=params["force_exact_search"],
                    probe_mult_factor=params["probe_mult_factor"],
                )

                logger.info("Queries: Embedding and building index (mentions)")
                vecs = test_men_vecs
                if params["transductive"]:
                    vecs = torch.cat((train_men_vecs, vecs), dim=0)
                men_embeds, men_index = data_process.embed_and_index(
                    model=reranker,
                    token_id_vecs=vecs,
                    encoder_type="context",
                    force_exact_search=params["force_exact_search"],
                    batch_size=params["embed_batch_size"],
                    probe_mult_factor=params["probe_mult_factor"],
                )
            else:
                logger.info("Dictionary: Embedding and building index (entities)")
                dict_embeds, dict_index = data_process.embed_and_index(
                    model=reranker,
                    token_id_vecs=test_dict_vecs,
                    encoder_type="candidate",
                    force_exact_search=params["force_exact_search"],
                    batch_size=params["embed_batch_size"],
                    probe_mult_factor=params["probe_mult_factor"],
                )

                logger.info("Queries: Embedding and building index (mentions)")
                vecs = test_men_vecs
                if params["transductive"]:
                    vecs = torch.cat((train_men_vecs, vecs), dim=0)
                men_embeds, men_index = data_process.embed_and_index(
                    model=reranker,
                    token_id_vecs=vecs,
                    encoder_type="context",
                    force_exact_search=params["force_exact_search"],
                    batch_size=params["embed_batch_size"],
                    probe_mult_factor=params["probe_mult_factor"],
                )

        recall_accuracy = {
            2**i: 0 for i in range(int(math.log(params["recall_k"], 2)) + 1)
        }
        recall_idxs = [0.0] * params["recall_k"]

        "2) Performs k-nearest neighbors (k-NN) search to establish relationships between mentions and entities."
        logger.info("Starting KNN search...")
        # Fetch recall_k (default 16) knn entities for all mentions
        if not use_types:
            _men_embeds = men_embeds
            if params["transductive"]:
                _men_embeds = _men_embeds[n_train_mentions:]
            nn_ent_dists, nn_ent_idxs = dict_index.search(
                _men_embeds, params["recall_k"]
            )
            n_mens_to_fetch = len(_men_embeds) if within_doc else max_knn + 1
            nn_men_dists, nn_men_idxs = men_index.search(_men_embeds, n_mens_to_fetch)
        else:
            query_len = len(men_embeds) - (
                n_train_mentions if params["transductive"] else 0
            )
            nn_ent_idxs = np.zeros((query_len, params["recall_k"]))
            nn_ent_dists = np.zeros((query_len, params["recall_k"]), dtype="float64")
            nn_men_idxs = -1 * np.ones((query_len, query_len), dtype=int)
            nn_men_dists = -1 * np.ones((query_len, query_len), dtype="float64")
            for entity_type in men_indexes:
                men_embeds_by_type = (
                    men_embeds[
                        men_idxs_by_type[entity_type][
                            men_idxs_by_type[entity_type] >= n_train_mentions
                        ]
                    ]
                    if params["transductive"]
                    else men_embeds[men_idxs_by_type[entity_type]]
                )
                nn_ent_dists_by_type, nn_ent_idxs_by_type = dict_indexes[
                    entity_type
                ].search(men_embeds_by_type, params["recall_k"])
                nn_ent_idxs_by_type = np.array(
                    list(
                        map(
                            lambda x: dict_idxs_by_type[entity_type][x],
                            nn_ent_idxs_by_type,
                        )
                    )
                )
                n_mens_to_fetch = len(men_embeds_by_type) if within_doc else max_knn + 1
                nn_men_dists_by_type, nn_men_idxs_by_type = men_indexes[
                    entity_type
                ].search(
                    men_embeds_by_type, min(n_mens_to_fetch, len(men_embeds_by_type))
                )
                nn_men_idxs_by_type = np.array(
                    list(
                        map(
                            lambda x: men_idxs_by_type[entity_type][x],
                            nn_men_idxs_by_type,
                        )
                    )
                )
                i = -1
                for idx in men_idxs_by_type[entity_type]:
                    if params["transductive"]:
                        idx -= n_train_mentions
                    if idx < 0:
                        continue
                    i += 1
                    nn_ent_idxs[idx] = nn_ent_idxs_by_type[i]
                    nn_ent_dists[idx] = nn_ent_dists_by_type[i]
                    nn_men_idxs[idx][: len(nn_men_idxs_by_type[i])] = (
                        nn_men_idxs_by_type[i]
                    )
                    nn_men_dists[idx][: len(nn_men_dists_by_type[i])] = (
                        nn_men_dists_by_type[i]
                    )
        logger.info("Search finished")

        "3) Constructs graphs based on these relationships."
        logger.info("Building graphs")
        # Find the most similar entity and k-nn mentions for each mention query
        for idx in range(len(nn_ent_idxs)):
            # Get nearest entity candidate
            dict_cand_idx = nn_ent_idxs[idx][0]

            # Indices of nearest entity candidates
            idx_top_candidates = nn_ent_idxs[idx][: params["recall_k"]]
            # Cuis of nearest entity candidates
            cuis_top_candidates = [
                [dict_idx_to_cui.get(index)] for index in idx_top_candidates
            ]

            mention_id = test_processed_data[idx]["mention_id"]
            mention_name = test_processed_data[idx]["mention_name"]

            # Check if the mention_id with .abbr_resolved exists
            if (df["mention_id"] + ".abbr_resolved" == mention_id).any():
                filtered_df = df[
                    (df["mention_id"] + ".abbr_resolved" == mention_id)
                ].copy()
                filtered_df["mention_id"] += ".abbr_resolved"
            else:
                filtered_df = df[df["mention_id"] == mention_id]

            if not filtered_df.empty:
                mention_dict = filtered_df.iloc[0].to_dict()
            else:
                print(
                    "This mention name was not found in the mention dataset:",
                    mention_id,
                    mention_name,
                )

            mention_dict["candidates"] = cuis_top_candidates

            if params["equivalent_cuis"]:
                synsets = ujson.load(
                    open(os.path.join(params["data_path"], "cui_synsets.json"))
                )
                """
                (A) expands the CUI IDs to include all equivalent CUIs in the candidate list.
                However each new equivalent CUI is treated as a separate candidate in the evaluation (bioel/evaluate.py).
                (B) Therefore it's just easier to consider all equivalent CUIs of the correct entity (given by column 'db_ids') and as long as a candidate match is found, it will be counted as a correct prediction.
                After careful consideration, it's better to not add it here because other models (SapBERT, BioGenel etc...) have different logic.
                I will just modify 'db_ids' directly in the evaluation script to consider equivalent CUIs.
                """
                ## (A)
                # mention_dict["candidates"] = [
                #     synsets[y[0]] for y in mention_dict["candidates"]
                # ]

                ## (B)
                # expanded_db_ids = []
                # for cui in mention_dict["db_ids"]:
                #     expanded_db_ids.extend(synsets.get(cui, [cui]))
                # mention_dict["db_ids"] = expanded_db_ids

            output_eval.append(mention_dict)

            dict_cand_score = nn_ent_dists[idx][0]
            # Compute recall metric
            gold_idxs = test_processed_data[idx]["label_idxs"][
                : test_processed_data[idx]["n_labels"]
            ]

            recall_idx = np.argwhere(nn_ent_idxs[idx] == gold_idxs[0])
            if len(recall_idx) != 0:
                recall_idxs[int(recall_idx)] += 1.0
                for recall_k in recall_accuracy:
                    if recall_idx < recall_k:
                        recall_accuracy[recall_k] += 1.0
            if not params["only_recall"]:
                filter_mask_neg1 = nn_men_idxs[idx] != -1
                men_cand_idxs = nn_men_idxs[idx][filter_mask_neg1]
                men_cand_scores = nn_men_dists[idx][filter_mask_neg1]

                if within_doc:
                    men_cand_idxs, wd_mask = filter_by_context_doc_id(
                        men_cand_idxs,
                        context_doc_ids[idx],
                        context_doc_ids,
                        return_numpy=True,
                    )
                    men_cand_scores = men_cand_scores[wd_mask]

                # Filter candidates to remove mention query and keep only the top k candidates
                filter_mask = men_cand_idxs != idx
                men_cand_idxs, men_cand_scores = (
                    men_cand_idxs[filter_mask][:max_knn],
                    men_cand_scores[filter_mask][:max_knn],
                )

                if params["transductive"]:
                    idx += n_train_mentions
                # Add edges to the graphs
                for k in joint_graphs:
                    joint_graph = joint_graphs[k]
                    # Add mention-entity edge
                    joint_graph["rows"] = np.append(
                        joint_graph["rows"], [n_entities + idx]
                    )  # Mentions added at an offset of maximum entities
                    joint_graph["cols"] = np.append(joint_graph["cols"], dict_cand_idx)
                    joint_graph["data"] = np.append(
                        joint_graph["data"], dict_cand_score
                    )
                    if k > 0:
                        # Add mention-mention edges
                        joint_graph["rows"] = np.append(
                            joint_graph["rows"],
                            [n_entities + idx] * len(men_cand_idxs[:k]),
                        )
                        joint_graph["cols"] = np.append(
                            joint_graph["cols"], n_entities + men_cand_idxs[:k]
                        )
                        joint_graph["data"] = np.append(
                            joint_graph["data"], men_cand_scores[:k]
                        )

        if params["transductive"]:
            # Add positive infinity mention-entity edges from training queries to labeled entities
            for idx, train_men in enumerate(train_processed_data):
                dict_cand_idx = train_men["label_idxs"][0]
                for k in joint_graphs:
                    joint_graph = joint_graphs[k]
                    joint_graph["rows"] = np.append(
                        joint_graph["rows"], [n_entities + idx]
                    )  # Mentions added at an offset of maximum entities
                    joint_graph["cols"] = np.append(joint_graph["cols"], dict_cand_idx)
                    joint_graph["data"] = np.append(joint_graph["data"], float("inf"))

        # Compute and print recall metric
        recall_idx_mode = np.argmax(recall_idxs)
        recall_idx_mode_prop = recall_idxs[recall_idx_mode] / np.sum(recall_idxs)
        logger.info(
            f"""
        Recall metrics (for {len(nn_ent_idxs)} queries):
        ---------------"""
        )
        logger.info(
            f"highest recall idx = {recall_idx_mode} ({recall_idxs[recall_idx_mode]}/{np.sum(recall_idxs)} = {recall_idx_mode_prop})"
        )
        for recall_k in recall_accuracy:
            recall_accuracy[recall_k] /= len(nn_ent_idxs)
            logger.info(f"recall@{recall_k} = {recall_accuracy[recall_k]}")

        # # Store results for evaluation object
        # if params["save_topk_result"]:
        #     base_filename = "biencoder_output_eval"
        #     biencoder_output_eval = create_versioned_filename(
        #         output_path, base_filename
        #     )
        #     with open(f"{biencoder_output_eval}", "w") as f:
        #         json.dump(output_eval, f, indent=2)
        #         print(f"\nPredictions overview saved at: {biencoder_output_eval}")

        if params["only_recall"]:
            return

        # Pickle the graphs
        print("Saving joint graphs...")
        with open(graph_path, "wb") as write_handle:
            pickle.dump(joint_graphs, write_handle, protocol=pickle.HIGHEST_PROTOCOL)

        if params["only_embed_and_build"]:
            logger.info(f"Saved embedding data at: {embed_data_path}")
            logger.info(f"Saved graphs at: {graph_path}")
            return

    graph_mode = params.get("graph_mode", None)

    result_overview = {
        "n_entities": n_entities,
        "n_mentions": n_mentions - (n_train_mentions if params["transductive"] else 0),
    }
    results = {}
    if graph_mode is None or graph_mode not in ["directed", "undirected"]:
        results["directed"] = []
        results["undirected"] = []
    else:
        results[graph_mode] = []

    knn_fetch_time = time.time() - time_start
    graph_processing_time = time.time()
    n_graphs_processed = 0.0

    for mode in results:
        print(f"\nEvaluation mode: {mode.upper()}")
        "4) Evaluates the model's accuracy by analyzing how effectively the model can link mentions to the correct entities."
        for k in joint_graphs:
            if k <= knn:
                print(f"\nGraph (k={k}):")
                # Partition graph based on cluster-linking constraints
                partitioned_graph, clusters = eval_cluster_linking.partition_graph(
                    joint_graphs[k],
                    n_entities,
                    mode == "directed",
                    return_clusters=True,
                )
                # Infer predictions from clusters
                result = eval_cluster_linking.analyzeClusters(
                    clusters,
                    entity_data,
                    test_processed_data,
                    k,
                    n_train_mentions if params["transductive"] else 0,
                )
                # log the accuracy
                acc = float(result["accuracy"].split(" ")[0])
                logger.info(f"Eval: accuracy for graph@k={k}: {acc}%")
                # Store result
                results[mode].append(result)
                n_graphs_processed += 1

    avg_graph_processing_time = (
        time.time() - graph_processing_time
    ) / n_graphs_processed
    avg_per_graph_time = (knn_fetch_time + avg_graph_processing_time) / 60

    # Store results
    output_file_name = os.path.join(
        output_path,
        f"eval_results_{__import__('calendar').timegm(__import__('time').gmtime())}",
    )

    try:
        for recall_k in recall_accuracy:
            result_overview[f"recall@{recall_k}"] = recall_accuracy[recall_k]
    except:
        logger.info("Recall data not available since graphs were loaded from disk")

    for mode in results:
        mode_results = results[mode]
        result_overview[mode] = {}
        for r in mode_results:
            k = r["knn_mentions"]
            result_overview[mode][f"accuracy@knn{k}"] = r["accuracy"]
            logger.info(f"{mode} accuracy@knn{k} = {r['accuracy']}")
            output_file = f"{output_file_name}-{mode}-{k}.json"
            with open(output_file, "w") as f:
                json.dump(r, f, indent=2)
                print(f"\nPredictions ({mode}) @knn{k} saved at: {output_file}")
    with open(f"{output_file_name}.json", "w") as f:
        json.dump(result_overview, f, indent=2)
        print(f"\nPredictions overview saved at: {output_file_name}.json")

    # Store results for evaluation object
    base_filename = "biencoder_output_eval"
    biencoder_output_eval = create_versioned_filename(output_path, base_filename)
    with open(f"{biencoder_output_eval}", "w") as f:
        json.dump(output_eval, f, indent=2)
        print(f"\nPredictions overview saved at: {biencoder_output_eval}")

    logger.info(
        "\nThe avg. per graph evaluation time is {} minutes\n".format(
            avg_per_graph_time
        )
    )


def loss_function(
    reranker,
    params,
    forward_output,
    data_module,
    n_entities,
    knn_dict,
    batch_context_inputs,
    accumulate_grad_batches,
):
    """
    Compute the loss function during the training.

    Parameters
    ----------
    - reranker : BiEncoderRanker
        biencoder model
    - params : dict
        Contains most of the relevant keys for training (embed_batch_size, batch_size, n_gpu, force_exact_search etc...)
    - forward_output : dict
        Output of the forward() method in the LitArboel class
    - data_module : Instance of ArboelDataModule class
    - n_entities : int
        Total number of entities
    - knn_dict : int (self.knn_dict = self.hparams["knn"]//2)
        Number of negative entities to fetch. It divides the k-nn evenly between entities and mentions
    - accumulate_grad_batches : int
        Number of steps to accumulate gradients
    """
    device = forward_output["context_inputs"].device
    # Compute the loss
    loss_dual_negs = loss_ent_negs = 0
    # loss of a batch includes both negative mentions and entity inputs (alongside positive examples ofc)
    loss_dual_negs, _ = reranker(
        context_input=forward_output["context_inputs"],
        label_input=forward_output["label_inputs"],
        mst_data={
            "positive_embeds": forward_output["positive_embeds"],
            "negative_dict_inputs": forward_output["negative_dict_inputs"],
            "negative_men_inputs": forward_output["negative_men_inputs"],
        },
        pos_neg_loss=params["pos_neg_loss"],
    )
    skipped_context_inputs = []
    if forward_output["skipped"] > 0 and not params["within_doc_skip_strategy"]:
        # Convert the numpy array to a tensor
        entity_dict_vecs_numpy = [
            data_module.entity_dict_vecs[x].numpy()
            for x in forward_output["skipped_negative_dict_inputs"]
        ]
        single_entity_dict_vecs = np.array(entity_dict_vecs_numpy)
        skipped_negative_dict_inputs = torch.tensor(single_entity_dict_vecs)

        skipped_positive_embeds = []
        for i, pos_idx in enumerate(forward_output["skipped_positive_idxs"]):
            if pos_idx < n_entities:
                data_module.entity_dict_vecs = data_module.entity_dict_vecs.to(device)
                pos_embed = reranker.encode_candidate(
                    cands=data_module.entity_dict_vecs.to(device)[
                        pos_idx : pos_idx + 1
                    ],
                    requires_grad=True,
                )
            else:
                pos_embed = reranker.encode_context(
                    ctxt=data_module.train_men_vecs.to(device)[
                        pos_idx - n_entities : pos_idx - n_entities + 1
                    ],
                    requires_grad=True,
                )
            skipped_positive_embeds.append(pos_embed)
        skipped_positive_embeds = torch.cat(skipped_positive_embeds)
        skipped_context_inputs = batch_context_inputs[
            ~np.array(forward_output["context_inputs_mask"])
        ]
        skipped_label_inputs = torch.tensor(
            [[1] + [0] * (knn_dict)] * len(skipped_context_inputs), dtype=torch.float32
        )
        # Loss of a batch that only includes negative entity inputs.
        loss_ent_negs, _ = reranker(
            context_input=skipped_context_inputs,
            label_input=skipped_label_inputs,
            mst_data={
                "positive_embeds": skipped_positive_embeds,
                "negative_dict_inputs": skipped_negative_dict_inputs,
                "negative_men_inputs": None,
            },
            pos_neg_loss=params["pos_neg_loss"],
        )

    # len(context_input) = Number of mentions in the batch that successfully found negative entities and mentions.
    # len(skipped_context_inputs): Number of mentions in the batch that only found negative entities.
    loss = (
        (
            loss_dual_negs * len(forward_output["context_inputs"])
            + loss_ent_negs * len(skipped_context_inputs)
        )
        / (len(forward_output["context_inputs"]) + len(skipped_context_inputs))
    ) / accumulate_grad_batches

    return loss


"Model Training"


class LitArboel(L.LightningModule):
    def __init__(self, params):
        """
        - params : dict
        Contains most of the relevant keys for training (embed_batch_size, batch_size, force_exact_search etc...)
        """
        super(LitArboel, self).__init__()
        self.save_hyperparameters(params)
        self.reranker = BiEncoderRanker(params)
        self.model = self.reranker.model

        self.best_max_acc = 0
        self.best_embed_and_index_dict = None

    def forward(self, batch_context_inputs, candidate_idxs, n_gold, mention_idxs):
        """
        Description
        -----------
        Processes a batch of input data to generate embeddings, and identifies positive and negative examples for training.
        It handles the construction of mention-entity graphs, computes nearest neighbors, and organizes the data for subsequent loss calculation.

        Parameters
        ----------
        - “batch_context_inputs” : Tensor
            Tensor containing IDs of (mention + surrounding context) tokens. Shape: (batch_size, context_length)
        - “candidate_idxs” : Tensor
            Tensor with indices pointing to the entities in the entity dictionary that are considered correct labels for the mention. Shape: (batch_size, candidate_count)
        - “n_gold” : Tensor
            Number of labels (=entities) associated with the mention. Shape: (batch_size,)
        - “mention_idx” : Tensor
            Tensor containing a sequence of integers from 0 to N-1 (N = number of mentions in the dataset) serbing as a unique identifier for each mention.

        Return
        ------
        - label_inputs : Tensor
            Tensor of binary labels indicating the correct candidates. Shape: (batch_size, 1 + knn_dict + knn_men), where 1 represents the positive example and the rest are negative examples.
        - context_inputs : Tensor
            Processed batch context inputs, filtered to remove mentions with no negative examples. Shape: (filtered_batch_size, context_length).
        - negative_men_inputs : Tensor
            Tensor of negative mention inputs. Shape: (filtered_batch_size * knn_men,)
        - negative_dict_inputs : Tensor
            Tensor of negative dictionary (entity) inputs. Shape: (filtered_batch_size * knn_dict)
        - positive_embeds : Tensor
            Tensor of embeddings for the positive examples. Shape: (filtered_batch_size, embedding_dim)
        - skipped : int
            The number of mentions skipped due to lack of valid negative examples.
        - skipped_positive_idxs : list(int)
            List of indices for positive examples that were skipped.
        - skipped_negative_dict_inputs :
            Tensor of negative dictionary inputs for skipped examples. Shape may vary based on the number of skipped examples and available negative dictionary entries.
        - context_inputs_mask : list(bool)
            Mask indicating which entries in batch_context_inputs were retained after filtering out mentions with no negative examples.
        """

        # mentions within the batch
        mention_embeddings = self.train_men_embeds[mention_idxs.cpu()]
        if len(mention_embeddings.shape) == 1:
            mention_embeddings = np.expand_dims(mention_embeddings, axis=0)
        # Convert Back to Tensor and Move to GPU
        mention_embeddings = torch.from_numpy(mention_embeddings).to(self.device)

        positive_idxs = []
        negative_dict_inputs = []
        negative_men_inputs = []

        skipped_positive_idxs = []
        skipped_negative_dict_inputs = []

        min_neg_mens = float("inf")
        skipped = 0
        context_inputs_mask = [True] * len(batch_context_inputs)

        "IV.4.B) For each mention within the batch"
        # For each mention within the batch
        for m_embed_idx, m_embed in enumerate(mention_embeddings):
            mention_idx = int(mention_idxs[m_embed_idx])
            # ground truth entities of the mention "mention_idx"
            gold_idxs = set(
                self.trainer.datamodule.train_processed_data[mention_idx]["label_idxs"][
                    : n_gold[m_embed_idx]
                ]
            )

            # TEMPORARY: Assuming that there is only 1 gold label, TODO: Incorporate multiple case
            assert n_gold[m_embed_idx] == 1

            if mention_idx in self.gold_links:
                gold_link_idx = self.gold_links[mention_idx]
            else:
                "IV.4.B.a) Create the graph with positive edges"
                # This block creates all the positive edges of the mention in this iteration
                # Run MST on mention clusters of all the gold entities of the current query mention to find its positive edge
                rows, cols, data, shape = (
                    [],
                    [],
                    [],
                    (
                        self.n_entities + self.train_n_mentions,
                        self.n_entities + self.train_n_mentions,
                    ),
                )
                seen = set()

                # Set whether the gold edge should be the nearest or the farthest neighbor
                sim_order = 1 if self.hparams["farthest_neighbor"] else -1

                for cluster_ent in gold_idxs:
                    # IDs of all the mentions inside the gold cluster with entity id = "cluster_ent"
                    cluster_mens = self.trainer.datamodule.train_gold_clusters[
                        cluster_ent
                    ]

                    if self.hparams["within_doc"]:
                        # Filter the gold cluster to within-doc
                        cluster_mens, _ = filter_by_context_doc_id(
                            mention_idxs=cluster_mens,
                            doc_id=self.trainer.datamodule.train_context_doc_ids[
                                mention_idx
                            ],
                            doc_id_list=self.trainer.datamodule.train_context_doc_ids,
                        )

                    # ψ(e, mi) = Enc_E(e)^T Enc_M(mi) for all the mention-entity links inside the cluster of the current mention
                    to_ent_data = (
                        self.train_men_embeds[cluster_mens]
                        @ self.train_dict_embeds[cluster_ent].T
                    )

                    # φ(mi, mj) = Enc_M(mi)^T Enc_M(mj) for all the mention-mention links inside the cluster of the current mention
                    to_men_data = (
                        self.train_men_embeds[cluster_mens]
                        @ self.train_men_embeds[cluster_mens].T
                    )

                    if self.hparams["gold_arbo_knn"] is not None:
                        # Descending order of similarity if nearest-neighbor, else ascending order
                        sorti = np.argsort(sim_order * to_men_data, axis=1)
                        sortv = np.take_along_axis(to_men_data, sorti, axis=1)
                        if self.hparams["rand_gold_arbo"]:
                            randperm = np.random.permutation(sorti.shape[1])
                            sortv, sorti = sortv[:, randperm], sorti[:, randperm]

                    for i in range(len(cluster_mens)):
                        from_node = self.n_entities + cluster_mens[i]
                        to_node = cluster_ent
                        # Add mention-entity link
                        rows.append(from_node)
                        cols.append(to_node)
                        data.append(-1 * to_ent_data[i])  # w_e,mi = - ψ(e, mi)
                        if self.hparams["gold_arbo_knn"] is None:
                            # Add forward and reverse mention-mention links over the entire MST
                            for j in range(i + 1, len(cluster_mens)):
                                to_node = self.n_entities + cluster_mens[j]
                                if (from_node, to_node) not in seen:
                                    score = to_men_data[i, j]
                                    rows.append(from_node)
                                    cols.append(to_node)
                                    # w_i,j = -ψ(mi, mj)
                                    data.append(
                                        -1 * score
                                    )  # Negatives needed for SciPy's Minimum Spanning Tree computation
                                    seen.add((from_node, to_node))
                                    seen.add((to_node, from_node))
                        else:
                            # Approximate the MST using <gold_arbo_knn> nearest mentions from the gold cluster
                            added = 0
                            approx_k = min(
                                self.hparams["gold_arbo_knn"] + 1, len(cluster_mens)
                            )
                            for j in range(approx_k):
                                if added == approx_k - 1:
                                    break
                                to_node = self.n_entities + cluster_mens[sorti[i, j]]
                                if to_node == from_node:
                                    continue
                                added += 1
                                if (from_node, to_node) not in seen:
                                    score = sortv[i, j]
                                    rows.append(from_node)
                                    cols.append(to_node)
                                    data.append(
                                        -1 * score
                                    )  # Negatives needed for SciPy's Minimum Spanning Tree computation
                                    seen.add((from_node, to_node))

                "IV.4.B.b) Fine tuning with inference procedure to get a mst"
                # Creates MST with entity constraint (inference procedure)
                csr = csr_matrix(
                    (-sim_order * np.array(data), (rows, cols)), shape=shape
                )
                # Note: minimum_spanning_tree expects distances as edge weights
                mst = minimum_spanning_tree(csr).tocoo()
                # Note: cluster_linking_partition expects similarities as edge weights # Convert directed to undirected graph
                rows, cols, data = cluster_linking_partition(
                    np.concatenate(
                        (mst.row, mst.col)
                    ),  # cluster_linking_partition is imported from eval_cluster_linking
                    np.concatenate((mst.col, mst.row)),
                    np.concatenate((sim_order * mst.data, sim_order * mst.data)),
                    self.n_entities,
                    directed=True,
                    silent=True,
                )
                assert np.array_equal(rows - self.n_entities, cluster_mens)

                for i in range(len(rows)):
                    men_idx = rows[i] - self.n_entities
                    if men_idx in self.gold_links:
                        continue
                    assert men_idx >= 0
                    add_link = True
                    # Store the computed positive edges for the mentions in the clusters only if they have the same gold entities as the query mention
                    for l in self.trainer.datamodule.train_processed_data[men_idx][
                        "label_idxs"
                    ][
                        : self.trainer.datamodule.train_processed_data[men_idx][
                            "n_labels"
                        ]
                    ]:
                        if l not in gold_idxs:
                            add_link = False
                            break
                    if add_link:
                        self.gold_links[men_idx] = cols[i]
                gold_link_idx = self.gold_links[mention_idx]

            "IV.4.B.c) Retrieve the pre-computed nearest neighbors"
            knn_dict_idxs = self.dict_nns[mention_idx]
            knn_dict_idxs = knn_dict_idxs.astype(np.int64).flatten()
            knn_men_idxs = self.men_nns[mention_idx][self.men_nns[mention_idx] != -1]
            knn_men_idxs = knn_men_idxs.astype(np.int64).flatten()
            if self.hparams["within_doc"]:
                knn_men_idxs, _ = filter_by_context_doc_id(
                    mention_idxs=knn_men_idxs,
                    doc_id=self.trainer.datamodule.train_context_doc_ids[mention_idx],
                    doc_id_list=self.trainer.datamodule.train_context_doc_ids,
                    return_numpy=True,
                )
            "IV.4.B.d) Add negative examples"
            neg_mens = list(
                knn_men_idxs[
                    ~np.isin(
                        knn_men_idxs,
                        np.concatenate(
                            [
                                self.trainer.datamodule.train_gold_clusters[gi]
                                for gi in gold_idxs
                            ]
                        ),
                    )
                ][: self.knn_men]
            )
            # Track queries with no valid mention negatives
            if len(neg_mens) == 0:
                context_inputs_mask[m_embed_idx] = False
                skipped_negative_dict_inputs += list(
                    knn_dict_idxs[~np.isin(knn_dict_idxs, list(gold_idxs))][
                        : self.knn_dict
                    ]
                )
                skipped_positive_idxs.append(gold_link_idx)
                skipped += 1
                continue
            else:
                min_neg_mens = min(min_neg_mens, len(neg_mens))
            negative_men_inputs.append(
                knn_men_idxs[
                    ~np.isin(
                        knn_men_idxs,
                        np.concatenate(
                            [
                                self.trainer.datamodule.train_gold_clusters[gi]
                                for gi in gold_idxs
                            ]
                        ),
                    )
                ][: self.knn_men]
            )
            negative_dict_inputs += list(
                knn_dict_idxs[~np.isin(knn_dict_idxs, list(gold_idxs))][: self.knn_dict]
            )
            # Add the positive example
            positive_idxs.append(gold_link_idx)

        "IV.4.C) Skip this iteration if no suitable negative examples found"
        if len(negative_men_inputs) == 0:
            return None

        # Sets the minimum number of negative mentions found across all processed mentions in the current batch
        self.knn_men = min_neg_mens

        # This step ensures that each mention is compared against a uniform number of negative mentions
        filtered_negative_men_inputs = []
        for row in negative_men_inputs:
            filtered_negative_men_inputs += list(row[: self.knn_men])
        negative_men_inputs = filtered_negative_men_inputs
        # Assertions for Data Integrity
        assert (
            len(negative_dict_inputs)
            == (len(mention_embeddings) - skipped) * self.knn_dict
        )
        assert (
            len(negative_men_inputs)
            == (len(mention_embeddings) - skipped) * self.knn_men
        )

        self.total_skipped += skipped
        self.total_knn_men_negs += self.knn_men

        entity_dict_vecs_numpy = [
            self.trainer.datamodule.entity_dict_vecs[x].numpy()
            for x in negative_dict_inputs
        ]
        # Then, convert the list of numpy arrays into a single numpy array
        single_entity_dict_vecs = np.array(entity_dict_vecs_numpy)
        # Finally, convert the numpy array to a tensor
        negative_dict_inputs = torch.tensor(single_entity_dict_vecs)

        train_men_vecs_numpy = [
            self.trainer.datamodule.train_men_vecs[x].numpy()
            for x in negative_men_inputs
        ]
        # Then, convert the list of numpy arrays into a single numpy array
        single_train_men_vecs = np.array(train_men_vecs_numpy)
        # Finally, convert the numpy array to a tensor
        negative_men_inputs = torch.tensor(single_train_men_vecs)

        # Labels indicating the correct candidates. Used for computing loss.
        positive_embeds = []
        for i, pos_idx in enumerate(positive_idxs):
            if pos_idx < self.n_entities:
                pos_embed = self.reranker.encode_candidate(
                    cands=self.trainer.datamodule.entity_dict_vecs.to(self.device)[
                        pos_idx : pos_idx + 1
                    ],
                    requires_grad=True,
                )
            else:
                pos_embed = self.reranker.encode_context(
                    ctxt=self.trainer.datamodule.train_men_vecs.to(self.device)[
                        pos_idx - self.n_entities : pos_idx - self.n_entities + 1
                    ],
                    requires_grad=True,
                )
            positive_embeds.append(pos_embed)
        positive_embeds = torch.cat(positive_embeds)

        # Remove mentions with no negative examples
        context_inputs = batch_context_inputs[context_inputs_mask]

        # Tensor containing binary values that act as indicator variables in the paper:
        # Contains Indicator variable such that I_{u,m_i} = 1 if(u,mi) ∈ E'_{m_i} and I{u,m_i} = 0 otherwise.
        label_inputs = torch.tensor(
            [[1] + [0] * (self.knn_dict + self.knn_men)] * len(context_inputs),
            dtype=torch.float32,
        ).to(self.device)

        return {
            "label_inputs": label_inputs,
            "context_inputs": context_inputs,
            "negative_men_inputs": negative_men_inputs,
            "negative_dict_inputs": negative_dict_inputs,
            "positive_embeds": positive_embeds,
            "skipped": skipped,
            "skipped_positive_idxs": skipped_positive_idxs,
            "skipped_negative_dict_inputs": skipped_negative_dict_inputs,
            "context_inputs_mask": context_inputs_mask,
        }

    def training_step(self, batch, batch_idx):
        # batch is a subsample from tensor_dataset
        batch_context_inputs, candidate_idxs, n_gold, mention_idxs = batch

        f = self.forward(
            batch_context_inputs=batch_context_inputs,
            candidate_idxs=candidate_idxs,
            n_gold=n_gold,
            mention_idxs=mention_idxs,
        )

        for key, value in f.items():
            if isinstance(value, torch.Tensor):
                f[key] = value.to(self.device)

        # Compute the loss
        train_loss = loss_function(
            reranker=self.reranker,
            params=self.hparams,
            forward_output=f,
            data_module=self.trainer.datamodule,
            n_entities=self.n_entities,
            knn_dict=self.knn_dict,
            batch_context_inputs=batch_context_inputs,
            accumulate_grad_batches=self.trainer.accumulate_grad_batches,
        )
        print("training loss :", train_loss)
        self.log(
            "train loss :", train_loss, on_step=True, on_epoch=True, sync_dist=True
        )

        return train_loss

    def configure_optimizers(self):

        # Define optimizer
        optimizer = get_bert_optimizer(
            models=[self.model],
            type_optimization=self.hparams["type_optimization"],
            learning_rate=self.hparams["learning_rate"],
            fp16=self.hparams.get("fp16"),
        )

        # Define scheduler
        num_train_steps = (
            int(
                len(self.trainer.datamodule.train_tensor_data)
                / self.hparams["train_batch_size"]
                / self.trainer.accumulate_grad_batches
            )
            * self.trainer.max_epochs
        )
        num_warmup_steps = int(num_train_steps * self.hparams["warmup_proportion"])

        scheduler = WarmupLinearSchedule(
            optimizer=optimizer,
            warmup_steps=num_warmup_steps,
            t_total=num_train_steps,
        )
        logger.info(" Num optimization steps = %d" % num_train_steps)
        logger.info(" Num warmup steps = %d", num_warmup_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_train_epoch_start(self):

        logger.info("On train epoch start")
        "IV.1) Compute mention and entity embeddings and indexes at the start of each epoch"
        # Compute mention and entity embeddings and indexes at the start of each epoch
        # Corpus is a collection of entities, which is used to build type-specific search indexes if provided.
        """
        With a Corpus : Multiple type-specific indexes are created, allowing for more targeted and efficient searches within specific categories of entities.
        'dict_embeds' and 'men_embeds': The resulting entity and mention embeddings.
        'dict_indexes' and 'men_indexes': Dictionary that will store search indexes (!= indices)for each unique entity type found in the corpus
        'dict_idxs_by_type' and 'men_idxs_by_type': Dictionary to store indices of the corpus elements, grouped by their entity type.
        !!! idxs = indices / indexes = indexes !!!
        """
        logger.info(
            "TRAINING. Dictionary: Embedding and building index"
        )  # For entities
        if self.hparams["use_types"]:  # type-specific indexes
            (
                self.train_dict_embeds,
                self.train_dict_indexes,
                self.train_dict_idxs_by_type,
            ) = data_process.embed_and_index(
                model=self.reranker,
                token_id_vecs=self.trainer.datamodule.entity_dict_vecs,
                encoder_type="candidate",
                corpus=self.trainer.datamodule.entity_dictionary,
                force_exact_search=self.hparams["force_exact_search"],
                batch_size=self.hparams["embed_batch_size"],
                probe_mult_factor=self.hparams["probe_mult_factor"],
            )
            (
                self.train_men_embeds,
                self.train_men_indexes,
                self.train_men_idxs_by_type,
            ) = data_process.embed_and_index(
                model=self.reranker,
                token_id_vecs=self.trainer.datamodule.train_men_vecs,
                encoder_type="context",
                corpus=self.trainer.datamodule.train_processed_data,
                force_exact_search=self.hparams["force_exact_search"],
                batch_size=self.hparams["embed_batch_size"],
                probe_mult_factor=self.hparams["probe_mult_factor"],
            )

        else:  # general indexes
            self.train_dict_embeds, self.train_dict_index = (
                data_process.embed_and_index(
                    model=self.reranker,
                    token_id_vecs=self.trainer.datamodule.entity_dict_vecs,
                    encoder_type="candidate",
                    force_exact_search=self.hparams["force_exact_search"],
                    batch_size=self.hparams["embed_batch_size"],
                    probe_mult_factor=self.hparams["probe_mult_factor"],
                )
            )
            self.train_men_embeds, self.train_men_index = data_process.embed_and_index(
                model=self.reranker,
                token_id_vecs=self.trainer.datamodule.train_men_vecs,
                encoder_type="context",
                force_exact_search=self.hparams["force_exact_search"],
                batch_size=self.hparams["embed_batch_size"],
                probe_mult_factor=self.hparams["probe_mult_factor"],
            )

        # Number of entities
        self.n_entities = len(self.trainer.datamodule.entity_dictionary)
        # Number of mentions in training set
        self.train_n_mentions = len(self.trainer.datamodule.train_processed_data)

        # Store golden MST links
        self.gold_links = {}
        # Calculate the number of negative entities and mentions to fetch # Divides the k-nn evenly between entities and mentions
        self.knn_dict = self.hparams["knn"] // 2
        self.knn_men = self.hparams["knn"] - self.knn_dict

        "3) knn search : indice and distance of k closest mentions and entities"
        logger.info("TRAINING. Starting KNN search...")
        # INFO: Fetching all sorted mentions to be able to filter to within-doc later=
        n_men_to_fetch = (
            len(self.train_men_embeds)
            if self.hparams["use_types"]
            else self.knn_men + self.trainer.datamodule.max_gold_cluster_len
        )
        n_ent_to_fetch = (
            self.knn_dict + 1
        )  # +1 accounts for the possibility of self-reference
        if not self.hparams["use_types"]:
            _, self.dict_nns = self.train_dict_index.search(
                self.train_men_embeds, n_ent_to_fetch
            )
            _, self.men_nns = self.train_men_index.search(
                self.train_men_embeds, n_men_to_fetch
            )
        else:
            self.dict_nns = -1 * np.ones((len(self.train_men_embeds), n_ent_to_fetch))
            self.men_nns = -1 * np.ones((len(self.train_men_embeds), n_men_to_fetch))
            for entity_type in self.train_men_indexes:
                self.men_embeds_by_type = self.train_men_embeds[
                    self.train_men_idxs_by_type[entity_type]
                ]
                _, self.dict_nns_by_type = self.train_dict_indexes[entity_type].search(
                    self.men_embeds_by_type, n_ent_to_fetch
                )
                _, self.men_nns_by_type = self.train_men_indexes[entity_type].search(
                    self.men_embeds_by_type,
                    min(n_men_to_fetch, len(self.men_embeds_by_type)),
                )
                self.dict_nns_idxs = np.array(
                    list(
                        map(
                            lambda x: self.train_dict_idxs_by_type[entity_type][x],
                            self.dict_nns_by_type,
                        )
                    )
                )
                self.men_nns_idxs = np.array(
                    list(
                        map(
                            lambda x: self.train_men_idxs_by_type[entity_type][x],
                            self.men_nns_by_type,
                        )
                    )
                )
                for i, idx in enumerate(self.train_men_idxs_by_type[entity_type]):
                    self.dict_nns[idx] = self.dict_nns_idxs[i]
                    self.men_nns[idx][: len(self.men_nns_idxs[i])] = self.men_nns_idxs[
                        i
                    ]
        logger.info("TRAINING. Search finished")

        self.total_skipped = self.total_knn_men_negs = 0

    def on_validation_epoch_start(self):

        torch.cuda.empty_cache()  # Empty the CUDA cache to free up GPU memory

        self.valid_n_mentions = len(
            self.trainer.datamodule.valid_processed_data
        )  # total number of mentions in validation set
        self.valid_max_knn = 8  # max number of neighbors

        "Computes embeddings and indexes for entities and mentions. "
        "This block is preparing the data for evaluation by transforming raw vectors into a format that can be efficiently used for retrieval and comparison operations"
        if self.hparams["use_types"]:  # corpus = entity data
            # corpus is a collection of entities, which is used to build type-specific search indexes if provided.
            """
            With a Corpus : Multiple type-specific indexes are created, allowing for more targeted and efficient searches within specific categories of entities.
            'dict_embeds' and 'men_embeds': The resulting entity and mention embeddings.
            'dict_indexes' and 'men_indexes': Dictionary that will store search indexes (!= indices)for each unique entity type found in the corpus
            'dict_idxs_by_type' and 'men_idxs_by_type': Dictionary to store indices of the corpus elements, grouped by their entity type.
            !!! idxs = indices / indexes = indexes !!!
            """
            logger.info(
                "VALIDATION. Dictionary: Embedding and building index"
            )  # For entities
            (
                self.valid_dict_embeds,
                self.valid_dict_indexes,
                self.valid_dict_idxs_by_type,
            ) = data_process.embed_and_index(
                model=self.reranker,
                token_id_vecs=self.trainer.datamodule.entity_dict_vecs,
                encoder_type="candidate",
                corpus=self.trainer.datamodule.entity_dictionary,
                force_exact_search=self.hparams["force_exact_search"],
                batch_size=self.hparams["embed_batch_size"],
                probe_mult_factor=self.hparams["probe_mult_factor"],
            )
            logger.info(
                "VALIDATION. Queries: Embedding and building index"
            )  # For mentions
            (
                self.valid_men_embeds,
                self.valid_men_indexes,
                self.valid_men_idxs_by_type,
            ) = data_process.embed_and_index(
                model=self.reranker,
                token_id_vecs=self.trainer.datamodule.valid_men_vecs,
                encoder_type="context",
                corpus=self.trainer.datamodule.valid_processed_data,
                force_exact_search=self.hparams["force_exact_search"],
                batch_size=self.hparams["embed_batch_size"],
                probe_mult_factor=self.hparams["probe_mult_factor"],
            )
        else:  # corpus = None
            """
            Without a Corpus: A single, general index is created for all embeddings, suitable for broad searches across the entire dataset.
            'dict_embeds' and 'men_embeds': The resulting entity and mention embeddings.
            'dict_index' and 'men_index': Dictionary that will store search index
            """
            logger.info("VALIDATION. Dictionary: Embedding and building index")
            self.valid_dict_embeds, self.valid_dict_index = (
                data_process.embed_and_index(
                    model=self.reranker,
                    token_id_vecs=self.trainer.datamodule.entity_dict_vecs,
                    encoder_type="candidate",
                    force_exact_search=self.hparams["force_exact_search"],
                    batch_size=self.hparams["embed_batch_size"],
                    probe_mult_factor=self.hparams["probe_mult_factor"],
                )
            )
            logger.info("VALIDATION. Queries: Embedding and building index")
            self.valid_men_embeds, self.valid_men_index = data_process.embed_and_index(
                model=self.reranker,
                token_id_vecs=self.trainer.datamodule.valid_men_vecs,
                encoder_type="context",
                force_exact_search=self.hparams["force_exact_search"],
                batch_size=self.hparams["embed_batch_size"],
                probe_mult_factor=self.hparams["probe_mult_factor"],
            )

        "Performs k-nearest neighbors (k-NN) search to establish relationships between mentions and entities."
        logger.info("VALIDATION. Eval: Starting KNN search...")
        # Fetch (k+1) NN mention candidates; fetching all mentions for within_doc to filter down later
        self.valid_n_men_to_fetch = (
            len(self.valid_men_embeds)
            if self.hparams["within_doc"]
            else self.valid_max_knn + 1
        )  # Number of mentions to fetch
        if not self.hparams["use_types"]:  # Only one index so only need one search
            self.valid_nn_ent_dists, self.valid_nn_ent_idxs = (
                self.valid_dict_index.search(self.valid_men_embeds, 1)
            )  # Returns the distance and the indice of the closest entity for all mentions in "men_embeds"
            self.valid_nn_men_dists, self.valid_nn_men_idxs = (
                self.valid_men_index.search(
                    self.valid_men_embeds, self.valid_n_men_to_fetch
                )
            )  # Returns the distances and the indices of the k closest mentions for all mentions in "men_embeds"
        else:  # Several indexes corresponding to the different entities in entity_data so we can use the specific search index
            self.valid_nn_ent_idxs = -1 * np.ones(
                (len(self.valid_men_embeds), 1), dtype=int
            )  # Indice of the closest entity for all mentions in "men_embeds"
            self.valid_nn_ent_dists = -1 * np.ones(
                (len(self.valid_men_embeds), 1), dtype="float64"
            )  # Distance of the closest entity for all mentions in "men_embeds"
            self.valid_nn_men_idxs = -1 * np.ones(
                (len(self.valid_men_embeds), self.valid_n_men_to_fetch), dtype=int
            )  # Indice of k closest mentions for all mentions in "men_embeds"
            self.valid_nn_men_dists = -1 * np.ones(
                (len(self.valid_men_embeds), self.valid_n_men_to_fetch), dtype="float64"
            )  # Distance of the k closest mentions for all mentions in "men_embeds"
            for entity_type in self.valid_men_indexes:
                # Creates a new list only containing the mentions for which type = entity_types
                self.valid_men_embeds_by_type = self.valid_men_embeds[
                    self.valid_men_idxs_by_type[entity_type]
                ]  # Only want to search the mentions that belongs to a specific type of entity.
                # Returns the distance and the indice of the closest entity for all mentions in "men_embeds" by entity type
                self.valid_nn_ent_dists_by_type, nn_ent_idxs_by_type = (
                    self.valid_dict_indexes[entity_type].search(
                        self.valid_men_embeds_by_type, 1
                    )
                )
                self.valid_nn_ent_idxs_by_type = np.array(
                    list(
                        map(
                            lambda x: self.valid_dict_idxs_by_type[entity_type][x],
                            nn_ent_idxs_by_type,
                        )
                    )
                )
                # Returns the distance and the indice of the k closest mentions for all mention in "men_embeds" by entity type
                # Note that here we may not necessarily have k mentions in each entity type which is why we use min(k,len(self.valid_men_embeds_by_type))
                (
                    self.valid_nn_men_dists_by_type,
                    nn_men_idxs_by_type,
                ) = self.valid_men_indexes[entity_type].search(
                    self.valid_men_embeds_by_type,
                    min(self.valid_n_men_to_fetch, len(self.valid_men_embeds_by_type)),
                )
                self.valid_nn_men_idxs_by_type = np.array(
                    list(
                        map(
                            lambda x: self.valid_men_idxs_by_type[entity_type][x],
                            nn_men_idxs_by_type,
                        )
                    )
                )
                for i, idx in enumerate(self.valid_men_idxs_by_type[entity_type]):
                    self.valid_nn_ent_idxs[idx] = self.valid_nn_ent_idxs_by_type[i]
                    self.valid_nn_ent_dists[idx] = self.valid_nn_ent_dists_by_type[i]
                    self.valid_nn_men_idxs[idx][
                        : len(self.valid_nn_men_idxs_by_type[i])
                    ] = self.valid_nn_men_idxs_by_type[i]
                    self.valid_nn_men_dists[idx][
                        : len(self.valid_nn_men_dists_by_type[i])
                    ] = self.valid_nn_men_dists_by_type[i]
        logger.info("VALIDATION. Eval: Search finished")

        self.embed_and_index_dict = (
            {
                "dict_embeds": self.valid_dict_embeds,
                "dict_indexes": self.valid_dict_indexes,
                "dict_idxs_by_type": self.valid_dict_idxs_by_type,
            }
            if self.hparams["use_types"]
            else {
                "dict_embeds": self.valid_dict_embeds,
                "dict_index": self.valid_dict_index,
            }
        )

        pass

    def validation_step(self, batch_, batch_idx):

        self.n_entities = len(self.trainer.datamodule.entity_dictionary)
        self.max_acc, self.dict_acc = evaluate(
            params=self.hparams,
            logger=logger,
            entity_data=self.trainer.datamodule.entity_dictionary,
            query_data=self.trainer.datamodule.valid_processed_data,
            valid_men_embeds=self.valid_men_embeds,
            nn_ent_dists=self.valid_nn_ent_dists,
            nn_ent_idxs=self.valid_nn_ent_idxs,
            nn_men_dists=self.valid_nn_men_dists,
            nn_men_idxs=self.valid_nn_men_idxs,
            n_entities=self.n_entities,
            n_mentions=self.valid_n_mentions,
            max_knn=self.valid_max_knn,
            use_types=self.hparams["use_types"],
            within_doc=self.hparams["within_doc"],
            context_doc_ids=self.trainer.datamodule.valid_context_doc_ids,
        )

    def on_validation_epoch_end(self):
        self.log("max_acc", self.max_acc, on_epoch=True, sync_dist=True)

        for key, value in self.dict_acc.items():
            self.log(f"dict_acc_{key}", value, on_epoch=True, sync_dist=True)

        if self.max_acc > self.best_max_acc:
            self.best_max_acc = self.max_acc
            self.best_embed_and_index_dict = self.embed_and_index_dict
            save_path = os.path.join(
                self.hparams.get("output_path"),
                f"embed_and_index_dict_:epoch_{self.current_epoch}.pth",
            )
            torch.save(self.best_embed_and_index_dict, save_path, pickle_protocol=4)

    def test_step(self, batch_, batch_idx):

        evaluate_test(
            params=self.hparams,
            reranker=self.reranker,
            test_dict_vecs=self.trainer.datamodule.entity_dict_vecs,
            test_processed_data=self.trainer.datamodule.test_processed_data,
            test_men_vecs=self.trainer.datamodule.test_men_vecs,
            logger=logger,
            entity_data=self.trainer.datamodule.entity_dictionary,
            train_processed_data=(
                self.trainer.datamodule.train_processed_data
                if self.hparams.get("transductive")
                else None
            ),
            train_men_vecs=(
                self.trainer.datamodule.train_men_vecs
                if self.hparams.get("transductive")
                else None
            ),
            use_types=self.hparams["use_types"],
            embed_batch_size=self.hparams["embed_batch_size"],
            force_exact_search=self.hparams["force_exact_search"],
            probe_mult_factor=self.hparams["probe_mult_factor"],
            within_doc=self.hparams["within_doc"],
            context_doc_ids=(
                self.trainer.datamodule.test_context_doc_ids
                if self.hparams["within_doc"]
                else None
            ),
        )

    def on_train_start(self):
        self.start_time = time.time()

    def on_train_end(self):
        if self.start_time is not None:
            execution_time = (time.time() - self.start_time) / 60
            logging.info(f"The training took {execution_time:.2f} minutes")
        else:
            logging.warning("Start time not set. Unable to calculate execution time.")

    def on_test_start(self):
        self.start_time = time.time()

    def on_test_end(self):
        if self.start_time is not None:
            execution_time = (time.time() - self.start_time) / 60
            logging.info(f"The testing took {execution_time:.2f} minutes")
        else:
            logging.warning("Start time not set. Unable to calculate execution time.")
