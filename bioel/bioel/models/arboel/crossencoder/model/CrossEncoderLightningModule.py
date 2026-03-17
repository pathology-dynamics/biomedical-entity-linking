# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import collections
import os
import io
import copy
import torch
import logging
import random
import time
import numpy as np
import pickle
import json
import ujson
from tqdm import tqdm, trange
from bioel.models.arboel.biencoder.model.common.optimizer import get_bert_optimizer
from bioel.models.arboel.biencoder.model.BiEncoderLightningModule import (
    create_versioned_filename,
)
from pytorch_transformers.optimization import WarmupLinearSchedule
from bioel.models.arboel.crossencoder.model.crossencoder import CrossEncoderRanker
import bioel.models.arboel.biencoder.data.data_process as data_process
from IPython import embed

from bioel.logger import setup_logger
from bioel.utils.bigbio_utils import (
    load_bigbio_dataset,
    dataset_to_df,
    add_deabbreviations,
    CUIS_TO_REMAP,
    CUIS_TO_EXCLUDE,
)

import lightning as L
from bioel.models.arboel.biencoder.model.common.params import BlinkParser

# logger = setup_logger()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME

OPTIM_SCHED_FNAME = "optim_sched.pth"


def convert_defaultdict(d):
    if isinstance(d, collections.defaultdict):
        d = {k: convert_defaultdict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        return {k: convert_defaultdict(v) for k, v in d.items()}
    elif isinstance(d, np.int64):
        return int(d)
    elif isinstance(d, np.float64):
        return float(d)
    return d


def merge_dicts(dict1, dict2, non_summing_keys=None):
    """Merge two dictionaries with support for nested structures and summing numeric values"""
    if non_summing_keys is None:
        non_summing_keys = set()

    merged_dict = copy.deepcopy(dict1)
    for key, value in dict2.items():
        if key in merged_dict:
            if key in non_summing_keys:
                # Replace value for keys that should not be summed
                merged_dict[key] = copy.deepcopy(value)
            elif isinstance(value, collections.defaultdict) and isinstance(
                merged_dict[key], collections.defaultdict
            ):
                for sub_key, sub_value in value.items():
                    merged_dict[key][sub_key] = (
                        merged_dict[key].get(sub_key, 0) + sub_value
                    )
            elif isinstance(value, dict) and isinstance(merged_dict[key], dict):
                merged_dict[key] = merge_dicts(
                    merged_dict[key], value, non_summing_keys
                )
            elif isinstance(value, (int, float)) and isinstance(
                merged_dict[key], (int, float)
            ):
                merged_dict[key] += value
            else:
                if isinstance(value, list) and isinstance(merged_dict[key], list):
                    merged_dict[key].extend(value)
                else:
                    merged_dict[key] = copy.deepcopy(value)
        else:
            merged_dict[key] = copy.deepcopy(value)
    return merged_dict


def accuracy(out, labels, return_bool_arr=False):
    outputs = np.argmax(out, axis=1)
    if return_bool_arr:
        return outputs == labels, outputs
    return np.sum(outputs == labels)


def recall(out, labels, k, return_bool_arr=False):
    top_k_indices = np.argsort(-out, axis=1)[:, :k]
    matches = np.any(top_k_indices == labels[:, None], axis=1)
    if return_bool_arr:
        return matches, top_k_indices
    return matches


def evaluate_single_batch(
    reranker,
    batch,
    logger,
    context_length,
    params,
    output_eval=None,
    recall_k=None,
    mention_data=None,
    compute_macro_avg=False,
    store_failure_success=False,
    fold_number=None,
):
    """
    Evaluate the performance of the crossencoder on a single batch of data.
    Parameters:
    - reranker: CrossEncoderRanker object
    - batch: tuple of torch tensors
    - logger: logger object
    - context_length: int
        Maximum context length for the crossencoder
    - params : dict
        Contains most of the relevant keys for training (embed_batch_size, batch_size, n_gpu, etc...)
    - output_eval: list (used for testing, not validation)
        List to store the output of the evaluation
    - recall_k: int
        Number of candidates to consider for recall
    - mention_data: (Optional) torch tensor
        Contains the test mention data (used for testing)
    - compute_macro_avg: bool
        Whether to compute the macro average accuracy (=Accuracy over each mention type).
    - store_failure_success: bool
        Whether to store the failure and success cases for analysis
    """
    context_input, label_input, mention_idxs = batch
    results = {}
    eval_accuracy = 0
    nb_eval_examples = 0

    if mention_data is not None:
        processed_mention_data = mention_data["mention_data"]
        dictionary = mention_data["entity_dictionary"]
        stored_candidates = mention_data["stored_candidates"]
        if store_failure_success:
            failsucc = {"failure": [], "success": []}

    n_evaluated_per_type = collections.defaultdict(int)
    n_hits_per_type = collections.defaultdict(int)

    with torch.no_grad():
        eval_loss, logits = reranker(context_input, label_input, context_length)

    logits = logits.detach().cpu().numpy()
    label_ids = label_input.cpu().numpy()

    # Accuracy
    tmp_eval_hits, predicted = accuracy(
        out=logits, labels=label_ids, return_bool_arr=True
    )
    eval_accuracy = np.sum(tmp_eval_hits)

    nb_eval_examples = context_input.size(0)

    if compute_macro_avg:
        for i, m_idx in enumerate(mention_idxs):
            mention_type = processed_mention_data[m_idx]["type"]
            n_evaluated_per_type[mention_type] += 1
            is_hit = tmp_eval_hits[i]
            n_hits_per_type[mention_type] += is_hit

    if store_failure_success:

        # Needed for the output for evaluation
        data = load_bigbio_dataset(params["dataset"], fold_number=fold_number)
        if params["path_to_abbrev"]:
            data_with_abbrev = add_deabbreviations(
                dataset=data, path_to_abbrev=params["path_to_abbrev"]
            )
        exclude = CUIS_TO_EXCLUDE[params["dataset"]]
        remap = CUIS_TO_REMAP[params["dataset"]]
        df = dataset_to_df(
            data_with_abbrev, entity_remapping_dict=remap, cuis_to_exclude=exclude
        )

        recall_tmp_eval_hits, recall_predicted = recall(
            out=logits, labels=label_ids, k=recall_k, return_bool_arr=True
        )
        eval_recall = np.sum(recall_tmp_eval_hits)
        print(f"Recall@{recall_k}: {eval_recall/nb_eval_examples*100}%")
        for i, m_idx in enumerate(mention_idxs):
            m_idx = m_idx.item()
            men_query = processed_mention_data[m_idx]
            print("mention_id : ", men_query["mention_id"])
            best_idx = recall_predicted[i][0]
            dict_pred = dictionary[stored_candidates["candidates"][m_idx][best_idx]]
            report_obj = {
                "mention_id": men_query["mention_id"],
                "mention_name": men_query["mention_name"],
                "mention_gold_cui": "|".join(men_query["label_cuis"]),
                "mention_gold_cui_name": "|".join(
                    [
                        dictionary[i]["title"]
                        for i in men_query["label_idxs"][: men_query["n_labels"]]
                    ]
                ),
                "predicted_name": dict_pred["title"],
                "predicted_cui": dict_pred["cui"],
            }
            failsucc["success" if tmp_eval_hits[i] else "failure"].append(report_obj)

            cuis_top_candidates = []
            mention_dict = {}
            for j in range(recall_k):
                idx = recall_predicted[i][j]
                dict_preds = dictionary[stored_candidates["candidates"][m_idx][idx]]
                cuis_top_candidates.append([dict_preds["cui"]])

            if ((df["mention_id"] + ".abbr_resolved") == men_query["mention_id"]).any():
                filtered_df = df[
                    df["mention_id"] + ".abbr_resolved" == men_query["mention_id"]
                ].copy()
                filtered_df["mention_id"] += ".abbr_resolved"

            else:
                filtered_df = df[df["mention_id"] == men_query["mention_id"]]

            if not filtered_df.empty:
                mention_dict = filtered_df.iloc[0].to_dict()
            else:
                print(
                    "This mention name was not found in the mention dataset :",
                    men_query["mention_id"],
                    men_query["mention_name"],
                )

            mention_dict["candidates"] = cuis_top_candidates

            if params["equivalent_cuis"]:
                synsets = ujson.load(
                    open(os.path.join(params["data_path"], "cui_synsets.json"))
                )
                mention_dict["candidates"] = [
                    synsets[y[0]] for y in mention_dict["candidates"]
                ]

            output_eval.append(mention_dict)

    results["nb_samples_evaluated"] = nb_eval_examples
    results["correct_pred"] = int(eval_accuracy)

    print(f"Eval: Accuracy: {eval_accuracy/nb_eval_examples*100}%")

    if compute_macro_avg:
        results["n_hits_per_type"] = n_hits_per_type
        results["n_evaluated_per_type"] = n_evaluated_per_type

    if store_failure_success:
        results["failure"] = failsucc["failure"]
        results["success"] = failsucc["success"]

    if not store_failure_success:
        logger.info(json.dumps(results))

    return results


class LitCrossEncoder(L.LightningModule):
    """ """

    def __init__(self, params):
        super(LitCrossEncoder, self).__init__()
        self.save_hyperparameters(params)
        self.reranker = CrossEncoderRanker(params)
        self.model = self.reranker.model

    def training_step(self, batch, batch_idx):
        context_input, label_input, _ = batch
        train_loss, _ = self.reranker(
            context_input, label_input, self.hparams["max_context_length"]
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
                len(self.trainer.datamodule.train_data["mention_data"])
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

    def validation_step(self, batch, batch_idx):

        results = evaluate_single_batch(
            reranker=self.reranker,
            batch=batch,
            logger=logger,
            context_length=self.hparams["max_context_length"],
            params=self.hparams,
            fold_number=self.hparams.get("fold_number"),
        )

        self.log(
            "Accuracy",
            results["correct_pred"] / results["nb_samples_evaluated"],
            on_epoch=True,
            sync_dist=True,
        )

    def on_test_epoch_start(self):

        self.n_evaluated_per_type = {}
        self.n_mentions_per_type = {}
        self.output_eval = []

        self.test_results = {
            # Accuracy computed when we only considered mentions with the correct cui in top candidates
            "normalized_accuracy": 0,
            # Average accuracy across all types (Only mentions with the correct cui in top candidates).
            "normalized_macro_avg_acc": 0,
            # Number of samples with the correct cui in top candidates in the test set
            "filtered_length": int(
                len(self.trainer.datamodule.test_data["mention_data"])
                - self.trainer.datamodule.nb_no_label_test
            ),
            # Number of samples with and without the correct cui in top candidates in the test set
            "unfiltered_length": len(self.trainer.datamodule.test_data["mention_data"]),
            "n_hits_per_type": collections.defaultdict(int),
            "n_evaluated_per_type": collections.defaultdict(int),
        }

        if self.trainer.limit_test_batches == 1.0:
            # Accuracy computed when we included mentions without the correct cui in top candidates
            self.test_results["unnormalized_accuracy"] = 0
            # Average accuracy across all types (Including mentions without the correct cui in top candidates)
            self.test_results["unnormalized_macro_avg_acc"] = 0
            self.test_results["n_mentions_per_type"] = collections.defaultdict(int)

    def test_step(self, batch, batch_idx):

        results = evaluate_single_batch(
            reranker=self.reranker,
            batch=batch,
            logger=logger,
            context_length=self.hparams["max_context_length"],
            recall_k=self.hparams["recall_k"],
            params=self.hparams,
            output_eval=self.output_eval,
            mention_data=self.trainer.datamodule.test_data,
            compute_macro_avg=True,
            store_failure_success=True,
            fold_number=self.hparams.get("fold_number"),
        )
        self.test_results = merge_dicts(self.test_results, results)

    def on_test_epoch_end(self):
        self.test_results["normalized_accuracy"] = float(
            self.test_results["correct_pred"]
            / self.test_results["nb_samples_evaluated"]
            * 100
        )

        for mention_type in self.test_results["n_evaluated_per_type"]:
            self.test_results["normalized_macro_avg_acc"] += float(
                self.test_results["n_hits_per_type"][mention_type]
                / self.test_results["n_evaluated_per_type"][mention_type]
                if self.test_results["n_evaluated_per_type"][mention_type] > 0
                else 0
            )

        self.test_results["normalized_macro_avg_acc"] = (
            self.test_results["normalized_macro_avg_acc"]
            / len(self.test_results["n_evaluated_per_type"])
            * 100
        )

        self.log(
            "Normalized Accuracy : Only considered mentions with the correct cui in top candidates",
            self.test_results["normalized_accuracy"],
            sync_dist=True,
        )

        if self.trainer.limit_test_batches == 1.0:

            self.test_results["unnormalized_accuracy"] = float(
                self.test_results["normalized_accuracy"]
                * self.test_results["filtered_length"]
                / self.test_results["unfiltered_length"]
            )

            self.log(
                "Unnormalized Accuracy : Include mentions without the correct cui in top candidates",
                self.test_results["unnormalized_accuracy"],
                sync_dist=True,
            )

            for men in self.trainer.datamodule.test_data["mention_data"]:
                self.test_results["n_mentions_per_type"][men["type"]] += 1

        self.test_results = convert_defaultdict(self.test_results)

        # Gather results from all GPUs
        gathered_test_results = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_test_results, self.test_results)
        gathered_output_eval = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_output_eval, self.output_eval)

        # Only the main process should save the combined results
        if self.trainer.is_global_zero:
            non_summing_keys = {
                "n_mentions_per_type",
                "filtered_length",
                "unfiltered_length",
            }
            combined_test_results = gathered_test_results[0]

            for result in gathered_test_results[1:]:
                combined_test_results = merge_dicts(
                    combined_test_results, result, non_summing_keys
                )

            # Average the results across all GPUs for the specific metrics
            world_size = torch.distributed.get_world_size()
            combined_test_results["normalized_accuracy"] /= world_size
            combined_test_results["normalized_macro_avg_acc"] /= world_size

            if self.trainer.limit_test_batches == 1.0 and isinstance(
                self.trainer.limit_test_batches, float
            ):
                combined_test_results["unnormalized_accuracy"] /= world_size

                # For unnormalized_macro_avg_acc
                for mention_type in combined_test_results["n_mentions_per_type"]:
                    combined_test_results["unnormalized_macro_avg_acc"] += float(
                        combined_test_results["n_hits_per_type"][mention_type]
                        / combined_test_results["n_mentions_per_type"][mention_type]
                        if combined_test_results["n_mentions_per_type"][mention_type]
                        > 0
                        else 0
                    )

                combined_test_results["unnormalized_macro_avg_acc"] = (
                    combined_test_results["unnormalized_macro_avg_acc"]
                    / len(combined_test_results["n_mentions_per_type"])
                    * 100
                )

            all_output_eval = [
                item for sublist in gathered_output_eval for item in sublist
            ]

            eval_filename = "crossencoder_eval_results"
            crossencoder_eval_results = create_versioned_filename(
                self.hparams["output_path"], eval_filename
            )
            with open(crossencoder_eval_results, "w") as f:
                json.dump(combined_test_results, f, indent=2)
                print(
                    f"\ncrossencoder_eval_results overview saved at: {crossencoder_eval_results}"
                )

            eval_filename2 = "crossencoder_output_eval"
            crossencoder_output_eval = create_versioned_filename(
                self.hparams["output_path"], eval_filename2
            )
            with open(crossencoder_output_eval, "w") as f:
                json.dump(all_output_eval, f, indent=2)
                print(
                    f"\ncrossencoder_output_eval overview saved at: {crossencoder_output_eval}"
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
