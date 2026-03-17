# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import io
import copy
import torch
import logging
import numpy as np
import pickle
import json
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import bioel.models.arboel.biencoder.data.data_process as data_process
from IPython import embed

from bioel.logger import setup_logger

import lightning as L
from bioel.models.arboel.biencoder.model.common.params import BlinkParser
from bioel.models.arboel.crossencoder.data.top_cands import save_topk_biencoder_cands
from bioel.models.arboel.biencoder.data.BiEncoderLightningDataModule import (
    read_dataset,
)
from bioel.models.arboel.biencoder.model.BiEncoderLightningModule import LitArboel
from datetime import datetime, timedelta
import torch.distributed as dist

# logger = setup_logger()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME

OPTIM_SCHED_FNAME = "optim_sched.pth"


def modify(context_input, candidate_input, max_seq_length):
    new_input = []
    context_input = context_input.tolist()
    candidate_input = candidate_input.tolist()

    for i in tqdm(range(len(context_input)), desc="Concatenating"):
        cur_input = context_input[i]
        cur_candidate = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidate)):
            # remove [CLS] token from candidate
            sample = cur_input + cur_candidate[j][1:]
            sample = sample[:max_seq_length]
            mod_input.append(sample)

        new_input.append(mod_input)

    return torch.LongTensor(new_input)


def prepare_data(params):
    bi_knn = params.get("bi_knn", 64)
    candidate_roots = [params.get("biencoder_indices_path"), params.get("data_path")]
    candidate_roots = [p for p in candidate_roots if p is not None]

    def _candidate_exists(data_split):
        return any(
            os.path.exists(
                os.path.join(root, f"candidates_{data_split}_top{bi_knn}.t7")
            )
            for root in candidate_roots
        )

    missing_files = any(
        not _candidate_exists(data_split)
        for data_split in ["train", "valid", "test"]
    )

    # Load the model and process data only if any file is missing
    if missing_files:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        MyModel = LitArboel.load_from_checkpoint(
            checkpoint_path=params["biencoder_checkpoint"]
        )
        MyModel.to(device)  # Move to the current device
        reranker = MyModel.reranker

    for data_split in ["train", "valid", "test"]:
        if not _candidate_exists(data_split):
            save_topk_biencoder_cands(
                bi_reranker=reranker,
                params=params,
                logger=logger,
                bi_tokenizer=AutoTokenizer.from_pretrained(
                    params["model_name_or_path"]
                ),
                topk=bi_knn,
            )


def load_data(
    data_split,
    bi_tokenizer,
    max_context_length,
    max_cand_length,
    knn,
    pickle_src_path,
    params,
    logger,
    return_dict_only=False,
):
    entity_dictionary_loaded = False
    entity_dictionary_pkl_path = os.path.join(
        pickle_src_path, "entity_dictionary.pickle"
    )
    if os.path.isfile(entity_dictionary_pkl_path):
        print("Loading stored processed entity dictionary...")
        with open(entity_dictionary_pkl_path, "rb") as read_handle:
            entity_dictionary = pickle.load(read_handle)
        entity_dictionary_loaded = True

    if return_dict_only and entity_dictionary_loaded:
        return entity_dictionary

    # Load data
    tensor_data_pkl_path = os.path.join(
        pickle_src_path, f"{data_split}_tensor_data.pickle"
    )
    processed_data_pkl_path = os.path.join(
        pickle_src_path, f"{data_split}_processed_data.pickle"
    )
    if os.path.isfile(tensor_data_pkl_path) and os.path.isfile(processed_data_pkl_path):
        print("Loading stored processed data...")
        with open(tensor_data_pkl_path, "rb") as read_handle:
            tensor_data = pickle.load(read_handle)
        with open(processed_data_pkl_path, "rb") as read_handle:
            processed_data = pickle.load(read_handle)
    else:
        data_samples = read_dataset(data_split, params["data_path"])
        if not entity_dictionary_loaded:
            with open(
                os.path.join(params["data_path"], "dictionary.pickle"), "rb"
            ) as read_handle:
                entity_dictionary = pickle.load(read_handle)

        # Check if dataset has multiple ground-truth labels
        mult_labels = "labels" in data_samples[0].keys()
        # Filter samples without gold entities
        data_samples = list(
            filter(
                lambda sample: (
                    (len(sample["labels"]) > 0)
                    if mult_labels
                    else (sample["label"] is not None)
                ),
                data_samples,
            )
        )
        logger.info("Read %d data samples." % len(data_samples))

        processed_data, entity_dictionary, tensor_data = (
            data_process.process_mention_data(
                data_samples,
                entity_dictionary,
                bi_tokenizer,
                max_context_length,
                max_cand_length,
                context_key=params["context_key"],
                multi_label_key="labels" if mult_labels else None,
                silent=params["silent"],
                logger=logger,
                debug=params["debug"],
                knn=knn,
                dictionary_processed=entity_dictionary_loaded,
            )
        )
        print("Saving processed data...")
        if not entity_dictionary_loaded:
            with open(entity_dictionary_pkl_path, "wb") as write_handle:
                pickle.dump(
                    entity_dictionary, write_handle, protocol=pickle.HIGHEST_PROTOCOL
                )
        with open(tensor_data_pkl_path, "wb") as write_handle:
            pickle.dump(tensor_data, write_handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(processed_data_pkl_path, "wb") as write_handle:
            pickle.dump(processed_data, write_handle, protocol=pickle.HIGHEST_PROTOCOL)

    if return_dict_only:
        return entity_dictionary
    return processed_data, entity_dictionary, tensor_data


def get_data_loader(
    params,
    data_split,
    tokenizer,
    context_length,
    candidate_length,
    max_seq_length,
    pickle_src_path,
    logger,
    inject_ground_truth=False,
    max_n=None,
    shuffle=True,
    return_data=False,
    custom_cand_set=None,
):
    # Load the top-k indices for each mention query and the ground truth label if it exists in the candidate set
    bi_knn = params.get("bi_knn", 64)
    logger.info(f"Loading {data_split} data...")
    cand_name = data_split
    if custom_cand_set is not None:
        logger.info(f"Loading custom candidate set: {custom_cand_set}...")
        cand_name = custom_cand_set
    primary_fname = os.path.join(
        params["biencoder_indices_path"], f"candidates_{cand_name}_top{bi_knn}.t7"
    )
    fallback_fname = os.path.join(
        params["data_path"], f"candidates_{cand_name}_top{bi_knn}.t7"
    )
    fname = primary_fname if os.path.exists(primary_fname) else fallback_fname
    if not os.path.exists(fname):
        logger.info(
            "Missing %s. Building biencoder top-k candidate files now...", fname
        )
        prepare_data(params)
        fname = primary_fname if os.path.exists(primary_fname) else fallback_fname
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Missing candidate file after preparation: {primary_fname} "
            f"(fallback: {fallback_fname})"
        )
    stored_data = torch.load(fname, weights_only=False)
    processed_data, entity_dictionary, tensor_data = load_data(
        data_split,
        tokenizer,
        context_length,
        candidate_length,
        1,
        pickle_src_path,
        params,
        logger,
    )
    logger.info("Loaded")
    dict_vecs = list(map(lambda x: x["ids"], entity_dictionary))

    mention_idxs = torch.tensor([i for i in range(len(stored_data["labels"]))])
    candidate_input = []
    keep_mask = [True] * len(stored_data["labels"])
    for i in tqdm(range(len(stored_data["labels"])), desc="Processing"):
        if stored_data["labels"][i] == -1:
            # If ground truth not in candidates, replace the last candidate with the ground truth
            if inject_ground_truth:
                gold_idx = processed_data[i]["label_idxs"][0]
                stored_data["labels"][i] = len(stored_data["candidates"][i]) - 1
                stored_data["candidates"][i][-1] = gold_idx
            else:  # skip the mention
                keep_mask[i] = False
                continue
        cands = list(map(lambda x: dict_vecs[x], stored_data["candidates"][i]))
        candidate_input.append(cands)
    candidate_input = np.array(candidate_input)
    context_input = tensor_data[:][0][keep_mask]
    label_input = torch.tensor(stored_data["labels"])[keep_mask]
    mention_idxs = mention_idxs[keep_mask]

    n_no_label = len(stored_data["labels"]) - np.sum(keep_mask)

    if max_n is not None:
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]

    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]

    context_input = modify(context_input, candidate_input, max_seq_length)
    tensor_data = TensorDataset(context_input, label_input, mention_idxs)
    dataloader = DataLoader(
        tensor_data,
        shuffle=shuffle,
        num_workers=47,
        batch_size=params[
            "train_batch_size" if data_split == "train" else "eval_batch_size"
        ],
        drop_last=True if data_split == "train" else False,
    )
    if return_data:
        return (
            dataloader,
            n_no_label,
            {
                "entity_dictionary": entity_dictionary,
                "mention_data": processed_data,
                "stored_candidates": stored_data,
            },
        )
    return dataloader, n_no_label


class CrossEncoderDataModule(L.LightningDataModule):
    """ """

    def __init__(self, params):
        """
        Parameters
        ----------
        - params : dict(str)
        Contains configuration options
        """
        super(CrossEncoderDataModule, self).__init__()
        self.save_hyperparameters(params)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams["model_name_or_path"]
        )

    def prepare_data(self):
        prepare_data(self.hparams)

    def setup(self, stage=None):
        """
        For processing and splitting. Called at the beginning of fit (train + validate), test, or predict.
        """

        # Fit
        if stage == "fit" or stage is None:

            # Training set
            self.train_loader, _, self.train_data = get_data_loader(
                params=self.hparams,
                data_split="train",
                tokenizer=self.tokenizer,
                context_length=self.hparams["max_context_length"],
                candidate_length=self.hparams["max_context_length"],
                max_seq_length=self.hparams["max_seq_length"],
                pickle_src_path=self.hparams["data_path"],
                logger=logger,
                inject_ground_truth=self.hparams["inject_train_ground_truth"],
                return_data=True,
            )

            # Validation set
            self.valid_loader, _, self.valid_data = get_data_loader(
                params=self.hparams,
                data_split="valid",
                tokenizer=self.tokenizer,
                context_length=self.hparams["max_context_length"],
                candidate_length=self.hparams["max_context_length"],
                max_seq_length=self.hparams["max_seq_length"],
                pickle_src_path=self.hparams["data_path"],
                shuffle=False,
                logger=logger,
                inject_ground_truth=self.hparams["inject_eval_ground_truth"],
                return_data=True,
            )

        # Test
        if stage == "test":
            # Test set
            self.test_loader, self.nb_no_label_test, self.test_data = get_data_loader(
                params=self.hparams,
                data_split="test",
                tokenizer=self.tokenizer,
                context_length=self.hparams["max_context_length"],
                candidate_length=self.hparams["max_context_length"],
                max_seq_length=self.hparams["max_seq_length"],
                pickle_src_path=self.hparams["data_path"],
                shuffle=False,
                logger=logger,
                inject_ground_truth=self.hparams["inject_eval_ground_truth"],
                return_data=True,
            )

    def train_dataloader(self):
        # Return the training DataLoader
        return self.train_loader

    def val_dataloader(self):
        # Return the training DataLoader
        return self.valid_loader

    def test_dataloader(self):
        # Return the training DataLoader
        return self.test_loader
