#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import csv
import logging
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from config import Configurable
from prepare_data.data_utils import generate_mini_batch_input
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from prepare_data.mrc_processor import MRCProcessor
from models.bert_mrc import BertTagger
from prepare_data.mrc_utils import convert_examples_to_features, convert_relation_examples_to_features
from utils.evaluate_funcs import compute_performance, generate_relation_examples, compute_performance_eachq
from log.get_logger import logger


def load_data(config):
    data_processor = MRCProcessor()

    # load data exampels
    logger.info("loading {} ...".format(config.train_file))
    train_examples = data_processor.get_train_examples(config.train_file)
    logger.info("{} train examples load sucessful.".format(len(train_examples)))

    logger.info("loading {} ...".format(config.dev_file))
    dev_examples = data_processor.get_test_examples(config.dev_file)
    logger.info("{} dev examples load sucessful.".format(len(dev_examples)))

    logger.info("loading {} ...".format(config.test_file))
    test_examples = data_processor.get_test_examples(config.test_file)
    logger.info("{} test examples load sucessful.".format(len(test_examples)))

    label_list = data_processor.get_labels([train_examples, dev_examples, test_examples])
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    # convert data example into featrues
    train_features = convert_examples_to_features(train_examples, tokenizer, label_list, config.max_seq_length, config.max_query_length, config.doc_stride)
    dev_features = convert_examples_to_features(dev_examples, tokenizer, label_list, config.max_seq_length, config.max_query_length, config.doc_stride)
    test_features = convert_examples_to_features(test_examples, tokenizer, label_list, config.max_seq_length, config.max_query_length, config.doc_stride)

    num_train_steps = int(len(train_examples) / config.train_batch_size * config.epochs)
    return tokenizer, train_features, dev_features, test_features, num_train_steps, label_list


def load_model(config, num_train_steps, label_list):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    model = BertTagger(config, num_labels=len(label_list))
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())

    # if config.bert_frozen == "true":
    # logger.info(param_optimizer)
    # param_optimizer = [tmp for tmp in param_optimizer if tmp.requires_grad]

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup_proportion,
                         t_total=num_train_steps, max_grad_norm=config.clip_grad)
    return model, optimizer, device, n_gpu


def adjust_learning_rate(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
    logger.info("current learning rate" + str(param_group['lr']))


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def train(tokenizer, model, optimizer, train_features, dev_features, test_features, config, \
          device, n_gpu, label_list, num_train_steps):
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    dev_best_ent_acc = 0
    dev_best_rel_acc = 0
    dev_best_ent_precision = 0
    dev_best_ent_recall = 0
    dev_best_ent_f1 = 0
    dev_best_rel_precision = 0
    dev_best_rel_recall = 0
    dev_best_rel_f1 = 0
    dev_best_loss = 1000000000000000

    test_best_ent_acc = 0
    test_best_rel_acc = 0
    test_best_ent_precision = 0
    test_best_ent_recall = 0
    test_best_ent_f1 = 0
    test_best_rel_precision = 0
    test_best_rel_recall = 0
    test_best_rel_f1 = 0
    test_best_loss = 1000000000000000

    model.train()

    step = 0
    for idx in range(int(config.epochs)):

        if idx == 4:
            logger.info(idx)

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        logger.info("#######" * 10)
        logger.info("EPOCH: " + str(idx))
        adjust_learning_rate(optimizer)
        num_example = len(train_features)
        num_batches = int(num_example / config.train_batch_size)
        train_indecies = np.random.permutation(num_example)
        for batch_i in tqdm(range(num_batches)):
            step += 1
            start_idx = batch_i * config.train_batch_size
            end_idx = min((batch_i + 1) * config.train_batch_size, num_example)
            mini_batch_idx = train_indecies[start_idx:end_idx]
            input_ids, input_mask, segment_ids, label_ids, valid_ids, label_mask, input_types, entity_types, relations, doc_tokens = \
                generate_mini_batch_input(train_features, mini_batch_idx, config)

            if config.use_cuda:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                valid_ids = valid_ids.to(device)
                label_mask = label_mask.to(device)

            loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, label_mask)
            if n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (batch_i + 1) % config.gradient_accumulation_steps == 0:
                lr_this_step = config.learning_rate * warmup_linear(global_step / num_train_steps, config.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        logger.info("")
        logger.info("current training loss is : " + str(loss.item()))

        tmp_dev_loss, tmp_dev_entity, tmp_dev_relation, ent_weight, rel_weight = eval_checkpoint(model, dev_features, config, device, n_gpu,
                                                                         label_list, eval_sign="dev")
        logger.info("ent_weight: {}".format(ent_weight))
        logger.info("rel_weight: {}".format(rel_weight))
        
        _, tmp_test_entity, tmp_test_relation = eval_checkpoint(model, test_features, config, device, n_gpu,
                                                                label_list, "test", tokenizer, ent_weight, rel_weight)

        logger.info("......" * 10)
        logger.info("TEST:")
        test_ent_acc, test_ent_pcs, test_ent_recall, test_ent_f1 = tmp_test_entity["accuracy"], tmp_test_entity["precision"], \
                                               tmp_test_entity["recall"], tmp_test_entity["f1"]
        test_rel_acc, test_rel_pcs, test_rel_recall, test_rel_f1 = tmp_test_relation["accuracy"], tmp_test_relation["precision"], \
                                               tmp_test_relation["recall"], tmp_test_relation["f1"]

        logger.info("question:")
        logger.info("entity  : acc={}, precision={}, recall={}, f1={}".format(test_ent_acc, test_ent_pcs, test_ent_recall, test_ent_f1))
        logger.info("relation: acc={}, precision={}, recall={}, f1={}".format(test_rel_acc, test_rel_pcs, test_rel_recall, test_rel_f1))
        logger.info("")

    # export a trained mdoel
    model_to_save = model
    output_model_file = os.path.join(config.output_dir, "bert_model.bin")
    if config.export_model == "True":
        torch.save(model_to_save.state_dict(), output_model_file)

    logger.info("TEST: loss={}".format(test_best_loss))
    logger.info("current best precision, recall, f1, acc :")
    logger.info("entity  : {}, {}, {}, {}".format(test_best_ent_precision, test_best_ent_recall, test_best_ent_f1, test_best_ent_acc))
    logger.info("relation: {}, {}, {}, {}".format(test_best_rel_precision, test_best_rel_recall, test_best_rel_f1, test_best_rel_acc))
    logger.info("=&=" * 15)


def eval_checkpoint(model_object, eval_features, config, device, n_gpu, label_list, eval_sign="dev", tokenizer=None,
                    ent_weight=[1,1,1], rel_weight=[1,1,1]):

    if eval_sign == "dev":
        loss, input_lst, doc_token_lst, input_mask_lst, pred_lst, gold_lst, label_mask_lst, type_lst, etype_lst, gold_relation = evaluate(model_object,
                                                                                               eval_features, config,
                                                                                               device, eval_sign="dev")
        eval_performance, _ = compute_performance_eachq(input_lst, doc_token_lst, input_mask_lst, pred_lst, gold_lst, label_mask_lst, type_lst, label_list)
        ent_p_list = np.array([ent_p["f1"] for ent_p in eval_performance["entity"]])
        rel_p_list = np.array( [rel_p["f1"] for rel_p in eval_performance["relation"]])

        ent_weight = (np.exp(ent_p_list) / sum(np.exp(ent_p_list))) * len(ent_p_list)
        rel_weight = (np.exp(rel_p_list) / sum(np.exp(rel_p_list))) * len(rel_p_list)
        ent_weight, rel_weight = ent_weight.tolist(), rel_weight.tolist()

        return loss, eval_performance["entity"], eval_performance["relation"], ent_weight, rel_weight

    elif eval_sign == "test" and tokenizer is not None:
        # evaluate head entity extraction
        _, ent_input_lst, ent_doc_lst, ent_input_mask_lst, ent_pred_lst, ent_gold_lst, ent_label_mask_lst, ent_type_lst, ent_etype_lst, ent_gold_relation = evaluate(model_object,
                                                                                                     eval_features,
                                                                                                     config,
                                                                                                     device,
                                                                                                     eval_sign="test")

        entity_performance, entity_logs = compute_performance(ent_input_lst, ent_doc_lst, ent_input_mask_lst, ent_pred_lst, ent_gold_lst, ent_label_mask_lst, ent_type_lst, label_list, tokenizer, ent_weight, rel_weight)

        best_rel_f1 = -1
        if len(entity_logs) > 0:
            entity_result_file = os.path.join(config.result_dir, "entity_vote_best_q.output")
            with open(entity_result_file, "w") as fw:
                for log in entity_logs:
                    for question in log["questions"]:
                        fw.write(question + "\n")
                    for token, true_label, pred_label in zip(log["doc_tokens"], log["true_label"], log["pred_label"]):
                        fw.write("\t".join(["{:<20}".format(token), true_label, pred_label]) + '\n')
                    fw.write("\n")

        # generate relation question based on head entity
        relation_examples = generate_relation_examples(ent_input_lst, ent_doc_lst, ent_input_mask_lst, ent_pred_lst, ent_gold_lst, ent_label_mask_lst, ent_etype_lst, ent_gold_relation, label_list, config, tokenizer, ent_weight) # batch x 3 x max_seq_len
        relation_features = convert_examples_to_features(relation_examples, tokenizer, label_list,
                                                                  config.max_seq_length, config.max_query_length,
                                                                  config.doc_stride)
        
        # evaluate tail entity extraction
        if len(relation_features) > 0:
            _, rel_input_lst, rel_doc_lst, rel_input_mask_lst, rel_pred_lst, rel_gold_lst, rel_label_mask_lst, rel_type_lst, rel_etype_lst, rel_gold_relation = evaluate(model_object, relation_features, config, device, eval_sign="test")
            relation_performance, relation_logs = compute_performance(rel_input_lst, rel_doc_lst, rel_input_mask_lst, rel_pred_lst, rel_gold_lst, rel_label_mask_lst, rel_type_lst, label_list, tokenizer)
            cur_rel_f1 = relation_performance["relation"]["f1"]
            if len(relation_logs) > 0 and cur_rel_f1 > best_rel_f1:
                relation_result_file = os.path.join(config.result_dir, "relation_vote_best_q.output")
                with open(relation_result_file, "w") as fw:
                    for log in relation_logs:
                        for question in log["questions"]:
                            fw.write(question + "\n")
                        for token, true_label, pred_label in zip(log["doc_tokens"], log["true_label"], log["pred_label"]):
                            fw.write("\t".join(["{:<20}".format(token), true_label, pred_label]) + '\n')
                        fw.write("\n")

            return 0, entity_performance["entity"], relation_performance["relation"]
        else:
            return 0, entity_performance["entity"], entity_performance["relation"]


def evaluate(model_object, eval_features, config, device, eval_sign="dev"):
    model_object.eval()

    eval_loss = 0
    input_lst = []
    input_mask_lst = []
    pred_lst = []
    label_mask_lst = []
    gold_lst = []
    type_lst = []
    etype_lst = []
    valid_lst = []
    gold_relation = []
    doc_token_lst = []
    eval_steps = 0

    num_example = len(eval_features)
    batch_size = config.dev_batch_size if eval_sign == "dev" else config.test_batch_size
    num_batches = int(num_example / batch_size)
    eval_indecies = range(num_example)
    for batch_i in range(num_batches):
        start_idx = batch_i * batch_size
        end_idx = min((batch_i + 1) * batch_size, num_example)
        mini_batch_idx = eval_indecies[start_idx:end_idx]
        input_ids, input_mask, segment_ids, label_ids, valid_ids, label_mask, input_types, entity_types, relations, doc_tokens = \
            generate_mini_batch_input(eval_features, mini_batch_idx, config)

        if config.use_cuda:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device) # [cls]+doc_labels+[sep]
            valid_ids = valid_ids.to(device)
            label_mask.to(device)

        with torch.no_grad():
            tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, label_ids, valid_ids, label_mask)
            logits = model_object(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, valid_ids=valid_ids, attention_mask_label=label_mask) # batch, max_seq_len, n_class

        input_ids = input_ids.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        input_mask = input_mask.to("cpu").numpy()
        label_ids = label_ids.to("cpu").numpy()
        valid_ids = valid_ids.to("cpu").numpy()
        label_mask = label_mask.to("cpu").numpy()
        logits = np.argmax(logits, axis=-1)  # batch*3, max_seq_len

        n_ques = int(input_ids.shape[0] / batch_size)
        input_ids = np.reshape(input_ids, (-1, n_ques, config.max_seq_length)).tolist()
        logits = np.reshape(logits, (-1, n_ques, config.max_seq_length)).tolist()  # batch, 3, max_seq_len
        input_mask = np.reshape(input_mask, (-1, n_ques, config.max_seq_length)).tolist()
        label_mask = np.reshape(label_mask, (-1, n_ques, config.max_seq_length)).tolist()
        valid_ids = np.reshape(valid_ids, (-1, n_ques, config.max_seq_length)).tolist()
        label_ids = np.reshape(label_ids, (-1, n_ques, config.max_seq_length)).tolist()  # batch, 3, max_seq_len

        eval_loss += tmp_eval_loss.mean().item()

        input_lst += input_ids
        input_mask_lst += input_mask
        pred_lst += logits
        gold_lst += [batch_input_type[0] for batch_input_type in label_ids]  # batch, 1, max_seq_len
        label_mask_lst += [batch_input_type[0] for batch_input_type in label_mask]
        valid_lst += [batch_valid_ids[0] for batch_valid_ids in valid_ids]
        type_lst += input_types  # type_lst: all_example
        etype_lst += entity_types  # etype_lst: all_example
        gold_relation += relations
        doc_token_lst += doc_tokens
        eval_steps += 1

    loss = round(eval_loss / eval_steps, 4)

    return loss, input_lst, doc_token_lst, input_mask_lst, pred_lst, gold_lst, label_mask_lst, type_lst, etype_lst, gold_relation


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/default.cfg')
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args, logger)
    tokenizer, train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config)
    model, optimizer, device, n_gpu = load_model(config, num_train_steps, label_list)
    train(tokenizer, model, optimizer, train_loader, dev_loader, test_loader, config, device, n_gpu, label_list, num_train_steps)


if __name__ == "__main__":
    main()
