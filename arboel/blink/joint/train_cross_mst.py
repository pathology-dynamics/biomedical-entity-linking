# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2021 Dhruv Agarwal and authors of arboEL.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
from datetime import datetime
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np
import math
 
from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer

import blink.candidate_retrieval.utils
from blink.joint.crossencoder import CrossEncoderRanker
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser
from blink.utils import copy_directory

from IPython import embed


logger = None


def modify(context_input, candidate_input, max_seq_length):
    new_input = []
    context_input = context_input.tolist()
    candidate_input = candidate_input.tolist()

    for i in range(len(context_input)):
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


def evaluate(
    reranker,
    eval_dataloader,
    device,
    logger,
    context_length,
    suffix=None,
    silent=True
):

    assert suffix is not None

    reranker.model.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    all_logits = []

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, label_input = batch
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, label_input, context_length)

        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()

        tmp_eval_accuracy = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        all_logits.extend(logits)

        nb_eval_examples += context_input.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy (%s): %.5f" % (suffix, normalized_eval_accuracy))
    results["normalized_accuracy"] = normalized_eval_accuracy
    results["logits"] = all_logits
    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def build_gold_coref_clusters(data):
    context_uids = data["context_uids"].tolist()
    uid2idx = {u : i for i, u in enumerate(context_uids)}
    # build ground truth coref clusters in terms of idxs, NOT uids
    gold_coref_clusters = [
        tuple(sorted([ctxt_uid] + coref_ctxts.tolist()))
            for ctxt_uid, coref_ctxts in zip(
                context_uids,
                data["pos_coref_ctxt_uids"]
            )
    ]
    gold_coref_clusters = [list(x) for x in set(gold_coref_clusters)]
    gold_coref_clusters = [sorted([uid2idx[x] for x in l])
                                for l in gold_coref_clusters]
    return gold_coref_clusters


def create_mst_dataloader(
    params,
    gold_coref_clusters,
    contexts,
    pos_cands,
    pos_cand_uids,
    knn_cands,
    knn_cand_uids,
    evaluate=False
):

    max_n = None
    if evaluate:
        max_n = 2048
    if params["debug"]:
        max_n = 200
    example_bundle_size = params["example_bundle_size"]
    batch_size = params["eval_batch_size"] if evaluate \
                    else params["train_batch_size"]

    context_input = None
    context_input_clusters = []

    if max_n:
        gold_coref_clusters = gold_coref_clusters[:max_n]

    for c in tqdm(gold_coref_clusters): # for every gold cluster
        if len(c) < 2:
            continue
        cluster_input_chunks = []
        for i in c:   # for all idxs in the cluster
            if len(pos_cands[i]) == 0 or knn_cand_uids[i].shape[0] == 0:
                continue
            ex_pos_cands = pos_cands[i]
            assert sorted(pos_cand_uids[i].tolist()) == pos_cand_uids[i].tolist()
            if len(ex_pos_cands.shape) == 1:
                ex_pos_cands = ex_pos_cands.unsqueeze(0)
            for j in range(ex_pos_cands.shape[0]): # for each coreferent mention
                candidate_bundle = ex_pos_cands[j].unsqueeze(0)
                k = 0
                while candidate_bundle.shape[0] < example_bundle_size:
                    k %= knn_cand_uids[i].shape[0]
                    if knn_cand_uids[i][k].item() in pos_cand_uids[i]:
                        k += 1
                        continue
                    candidate_bundle = torch.cat(
                        (candidate_bundle, knn_cands[i][k].unsqueeze(0)) # Add negatives to the bundle
                    )
                    k += 1

                cluster_input_chunks.append(
                    modify( # Modify with concatenated mention-entity pairs for every candidate in the bundle
                        contexts[i].unsqueeze(0),
                        candidate_bundle.unsqueeze(0),
                        params["max_seq_length"]
                    )
                )

        context_input_clusters.append(torch.cat(cluster_input_chunks))

    if max_n:
        context_input_clusters = context_input_clusters[:max_n]

    sampler = RandomSampler(context_input_clusters)
    mst_dataloader = DataLoader(
        context_input_clusters, 
        sampler=sampler, 
        batch_size=1
    )
    return mst_dataloader


def create_dataloader(
    params,
    contexts,
    pos_cands,
    pos_cand_uids,
    knn_cands,
    knn_cand_uids,
    evaluate=False
):

    max_n = None
    if evaluate:
        max_n = 2048
    if params["debug"]:
        max_n = 200
    example_bundle_size = params["example_bundle_size"]
    batch_size = params["eval_batch_size"] if evaluate \
                    else params["train_batch_size"]

    context_input = None
    context_input_chunks = []

    for i in trange(contexts.shape[0]):
        if len(pos_cands[i]) == 0 or knn_cand_uids[i].shape[0] == 0:
            continue
        ex_pos_cands = pos_cands[i]
        if len(ex_pos_cands.shape) == 1:
            ex_pos_cands = ex_pos_cands.unsqueeze(0)
        for j in range(ex_pos_cands.shape[0]):
            candidate_bundle = ex_pos_cands[j].unsqueeze(0)
            k = 0
            while candidate_bundle.shape[0] < example_bundle_size:
                k %= knn_cand_uids[i].shape[0]
                if knn_cand_uids[i][k].item() in pos_cand_uids[i]:
                    k += 1
                    continue
                candidate_bundle = torch.cat(
                    (candidate_bundle, knn_cands[i][k].unsqueeze(0))
                )
                k += 1

            context_input_chunks.append(
                modify(
                    contexts[i].unsqueeze(0),
                    candidate_bundle.unsqueeze(0),
                    params["max_seq_length"]
                )
            )

    # concatenate all of the chunks together
    context_input = torch.cat(context_input_chunks) 
    if max_n:
        context_input = context_input[:max_n]

    # labels for each softmax bundle (positive always first)
    label_input = torch.zeros((context_input.shape[0],), dtype=torch.long)

    tensor_data = TensorDataset(context_input, label_input)
    sampler = RandomSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, 
        sampler=sampler, 
        batch_size=batch_size
    )
    return dataloader


def train_one_epoch_mst(
    train_dataloader,
    reranker,
    optimizer,
    scheduler,
    logger,
    params,
    epoch_idx,
    device=None,
    suffix=None,
):

    assert suffix is not None
    context_length = params["max_context_length"]
    grad_acc_steps = params["gradient_accumulation_steps"]
    model = reranker.model

    model.train()

    tr_loss = 0
    results = None

    if params["silent"]:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader, desc="Batch ({})".format(suffix))

    for step, batch in enumerate(iter_):
        cluster_context_input = batch[0].squeeze(0)
        # quadratic formula to get cluster size from num edges
        cluster_size = int((1 + math.sqrt(1 + 4*cluster_context_input.shape[0]))/2)

        context_input = None
        with torch.no_grad():
            # get scores
            scores = []
            sampler = SequentialSampler(cluster_context_input)
            scores_dataloader = DataLoader(
                cluster_context_input, 
                sampler=sampler, 
                batch_size=params["eval_batch_size"]
            )
            for sub_context_input in scores_dataloader:
                sub_context_input = sub_context_input.to(device)
                scores.append(
                    reranker.score_candidate(
                        sub_context_input,
                        context_length
                    )
                )

            scores = torch.cat(scores)

            scores = scores.cpu()
            cluster_context_input = cluster_context_input.cpu()

            # compute mst
            pos_scores = scores[:, 0].tolist()
            k = 0
            affinity_matrix = np.zeros((cluster_size, cluster_size))
            reverse_idx_map = {}
            for i in range(cluster_size):
                for j in range(cluster_size):
                    if i != j:
                        reverse_idx_map[(i, j)] = k
                        # negative because mst
                        affinity_matrix[i, j] = -pos_scores[k]
                        k += 1
            mst = minimum_spanning_tree(csr_matrix(affinity_matrix)).tocoo()

            # compute training edges
            train_idxs = [reverse_idx_map[t] for t in zip(mst.row, mst.col)]
            hard_neg_idxs = torch.argmax(scores[:,1:][train_idxs], 1) + 1
            hard_neg_idxs = hard_neg_idxs.tolist()
            context_bundles = []
            for i, neg in zip(train_idxs, hard_neg_idxs):
                bundle = [cluster_context_input[i,0,:].unsqueeze(0)]
                bundle.append(cluster_context_input[i,neg,:].unsqueeze(0))
                bundle = torch.cat(bundle)
                context_bundles.append(bundle.unsqueeze(0))
            context_input = torch.cat(context_bundles)

        label_input = torch.zeros((context_input.shape[0],), dtype=torch.long) # DA: Where is the positive edge being set?
        tensor_data = TensorDataset(context_input, label_input)
        sampler = RandomSampler(tensor_data)
        sub_dataloader = DataLoader(
            tensor_data, 
            sampler=sampler, 
            batch_size=params["train_batch_size"] * params["example_bundle_size"] // 2
        )

        iter_ = sub_dataloader
        for _, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input, label_input = batch
            loss, _ = reranker(context_input, label_input, context_length)
            loss.backward()
            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), params["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def train_one_epoch(
    train_dataloader,
    reranker,
    optimizer,
    scheduler,
    logger,
    params,
    epoch_idx,
    device=None,
    suffix=None,
):

    assert suffix is not None
    context_length = params["max_context_length"]
    grad_acc_steps = params["gradient_accumulation_steps"]
    model = reranker.model

    model.train()

    tr_loss = 0
    results = None

    if params["silent"]:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader, desc="Batch ({})".format(suffix))

    part = 0
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, label_input = batch
        loss, _ = reranker(context_input, label_input, context_length)

        # if n_gpu > 1:
        #     loss = loss.mean() # mean() to average on multi-gpu.

        if grad_acc_steps > 1:
            loss = loss / grad_acc_steps

        tr_loss += loss.item()

        if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
            logger.info(
                "({}) Step {} - epoch {} average loss: {}\n".format(
                    suffix,
                    step,
                    epoch_idx,
                    tr_loss / (params["print_interval"] * grad_acc_steps),
                )
            )
            tr_loss = 0

        loss.backward()

        if (step + 1) % grad_acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), params["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def main(params):

    # create output dir
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    model_output_path = os.path.join(params["output_path"], datetime_str)
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    # get logger
    logger = utils.get_logger(model_output_path)

    # copy blink source and create rerun script
    blink_copy_path = os.path.join(model_output_path, "blink")
    copy_directory("blink", blink_copy_path)
    cmd = sys.argv
    with open(os.path.join(model_output_path, "rerun.sh"), "w") as f:
        cmd.insert(0, "python")
        f.write(" ".join(cmd))

    # Init model
    ctxt_reranker = CrossEncoderRanker(params)
    ctxt_model = ctxt_reranker.model
    tokenizer = ctxt_reranker.tokenizer

    params["pool_highlighted"] = False # only `True` for ctxt
    cand_reranker = CrossEncoderRanker(params)
    cand_model = cand_reranker.model

    device = ctxt_reranker.device
    n_gpu = ctxt_reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if ctxt_reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    context_length = params["max_context_length"]

    # create train dataloaders
    fname = os.path.join(params["data_path"], "joint_train.t7")
    train_data = torch.load(fname)
    gold_coref_clusters = build_gold_coref_clusters(train_data)
    ctxt_train_dataloader = create_mst_dataloader(
        params,
        gold_coref_clusters,
        train_data["contexts"],
        train_data["pos_coref_ctxts"],
        train_data["pos_coref_ctxt_uids"],
        train_data["knn_ctxts"],
        train_data["knn_ctxt_uids"]
    )

    cand_train_dataloader = create_dataloader(
        params,
        train_data["contexts"],
        train_data["pos_cands"],
        train_data["pos_cand_uids"],
        train_data["knn_cands"],
        train_data["knn_cand_uids"]
    )
    
    fname = os.path.join(params["data_path"], "joint_valid.t7")
    valid_data = torch.load(fname)
    ctxt_valid_dataloader = create_dataloader(
        params,
        valid_data["contexts"],
        valid_data["pos_coref_ctxts"],
        valid_data["pos_coref_ctxt_uids"],
        valid_data["knn_ctxts"],
        valid_data["knn_ctxt_uids"],
        evaluate=True
    )
    cand_valid_dataloader = create_dataloader(
        params,
        valid_data["contexts"],
        valid_data["pos_cands"],
        valid_data["pos_cand_uids"],
        valid_data["knn_cands"],
        valid_data["knn_cand_uids"],
        evaluate=True
    )

    # evaluate before training
    ctxt_results = evaluate(
        ctxt_reranker,
        ctxt_valid_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
        suffix="ctxt",
        silent=params["silent"],
    )
    cand_results = evaluate(
        cand_reranker,
        cand_valid_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
        suffix="cand",
        silent=params["silent"],
    )

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    ctxt_optimizer = get_optimizer(ctxt_model, params)
    ctxt_scheduler = get_scheduler(
        params,
        ctxt_optimizer,
        len(ctxt_train_dataloader) * train_batch_size,
        logger
    )

    cand_optimizer = get_optimizer(cand_model, params)
    cand_scheduler = get_scheduler(
        params,
        cand_optimizer,
        len(cand_train_dataloader) * train_batch_size,
        logger
    )

    ctxt_best_epoch_idx = -1
    ctxt_best_score = -1
    cand_best_epoch_idx = -1
    cand_best_score = -1

    num_train_epochs = params["num_train_epochs"]

    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        # train ctxt model
        train_one_epoch_mst(
            ctxt_train_dataloader,
            ctxt_reranker,
            ctxt_optimizer,
            ctxt_scheduler,
            logger,
            params,
            epoch_idx,
            device=device,
            suffix='ctxt'
        )

        # train cand model
        train_one_epoch(
            cand_train_dataloader,
            cand_reranker,
            cand_optimizer,
            cand_scheduler,
            logger,
            params,
            epoch_idx,
            device=device,
            suffix='cand'
        )

        logger.info("***** Saving fine - tuned models *****")
        ctxt_epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx), "ctxt"
        )
        utils.save_model(ctxt_model, tokenizer, ctxt_epoch_output_folder_path)
        cand_epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx), "cand"
        )
        utils.save_model(cand_model, tokenizer, cand_epoch_output_folder_path)

        ctxt_results = evaluate(
            ctxt_reranker,
            ctxt_valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            suffix="ctxt",
            silent=params["silent"],
        )
        cand_results = evaluate(
            cand_reranker,
            cand_valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            suffix="cand",
            silent=params["silent"],
        )

        ctxt_ls = [ctxt_best_score, ctxt_results["normalized_accuracy"]]
        ctxt_li = [ctxt_best_epoch_idx, epoch_idx]
        ctxt_best_score = ctxt_ls[np.argmax(ctxt_ls)]
        ctxt_best_epoch_idx = ctxt_li[np.argmax(ctxt_ls)]

        cand_ls = [cand_best_score, cand_results["normalized_accuracy"]]
        cand_li = [cand_best_epoch_idx, epoch_idx]
        cand_best_score = cand_ls[np.argmax(cand_ls)]
        cand_best_epoch_idx = cand_li[np.argmax(cand_ls)]

        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best models
    logger.info(
        "Best ctxt performance in epoch: {}".format(ctxt_best_epoch_idx)
    )
    best_ctxt_model_path = os.path.join(
        model_output_path, "epoch_{}".format(ctxt_best_epoch_idx), "ctxt"
    )
    logger.info(
        "Best cand performance in epoch: {}".format(cand_best_epoch_idx)
    )
    best_cand_model_path = os.path.join(
        model_output_path, "epoch_{}".format(cand_best_epoch_idx), "cand"
    )

    copy_directory(
        best_ctxt_model_path,
        os.path.join(model_output_path, "best_epoch", "ctxt")
    )
    copy_directory(
        best_cand_model_path,
        os.path.join(model_output_path, "best_epoch", "cand")
    )


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_joint_train_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
