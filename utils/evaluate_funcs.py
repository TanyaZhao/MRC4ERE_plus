#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from .relation_template import *
from prepare_data.mrc_example import MRCExample
from prepare_data.mrc_utils import iob2_to_iobes
from collections import Counter

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)


def cal_f1_score(pcs, rec):
    tmp = 2 * pcs * rec / (pcs + rec)
    return round(tmp, 4)


def extract_entities(label_lst, label_list):
    label_dict = {label: i for i, label in enumerate(label_list)}

    entities = dict()
    temp_entity = []

    for idx, label in enumerate(label_lst):
        if label == label_dict["S"] or label == label_dict["B"]:
            if len(temp_entity) > 0:
                entities["{0}_{1}".format(temp_entity[0], temp_entity[-1])] = temp_entity
                temp_entity = []
            temp_entity.append(idx)
        elif label == label_dict["I"] or label == label_dict["E"]:
            temp_entity.append(idx)
        elif label == label_dict["O"] and len(temp_entity) > 0:
            entities["{0}_{1}".format(temp_entity[0], temp_entity[-1])] = temp_entity
            temp_entity = []

    if len(temp_entity) > 0:
        entities["{0}_{1}".format(temp_entity[0], temp_entity[-1])] = temp_entity

    return entities


def split_index(label_list):
    label_dict = {label: i for i, label in enumerate(label_list)}
    label_idx = [tmp_value for tmp_key, tmp_value in label_dict.items() if
                 "S" in tmp_key.split("-")[0] or "B" in tmp_key]
    str_label_idx = [str(tmp) for tmp in label_idx]
    label_idx = "_".join(str_label_idx)
    return label_idx

def compute_performance_eachq(input_ids, doc_tokens, input_masks, pred_labels, gold_labels, label_masks, input_types, label_list, tokenizer=None):
    '''
    :param input_ids: num_all_example, 3, max_seq
    :param doc_tokens: num_all_example, max_seq
    :param input_masks: num_all_example, 3, max_seq
    :param pred_labels: num_all_example, 3, max_seq
    :param gold_labels: num_all_example, max_seq
    :param label_masks: num_all_example, max_seq
    :param input_types:
    :param label_list:
    :param tokenizer:
    :return:
    '''
    label_map = {i: label for i, label in enumerate(label_list)}

    num_ques = len(input_ids[0])
    ent_accuracy, ent_positive, ent_extracted, ent_true = [0]*num_ques, [0]*num_ques, [0]*num_ques, [0]*num_ques
    rel_accuracy, rel_positive, rel_extracted, rel_true = [0]*num_ques, [0]*num_ques, [0]*num_ques, [0]*num_ques
    num_rel, num_ent = 0, 0
    logs = []
    for every_input_ids, every_doc_tokens, every_input_masks, every_pred_label, every_golden_label, every_label_masks, every_input_type in \
            zip(input_ids, doc_tokens, input_masks, pred_labels, gold_labels, label_masks, input_types):
        # every_doc_tokens: max_seq
        # batch_pred_label: 3, max_seq
        # extract gold_label
        mask_index = [tmp_idx for tmp_idx, tmp in enumerate(every_label_masks) if tmp != 0]
        gold_label_ids = [tmp for tmp_idx, tmp in enumerate(every_golden_label) if tmp_idx in mask_index]
        gold_label_ids = gold_label_ids[1:-1]
        truth_entities = extract_entities(gold_label_ids, label_list)

        # extract gold_label from multiple question answers
        final_pred_entities = {}
        final_pred_list = []
        pred_label_list = []
        for pred_label_i in every_pred_label: # pred_label_i: max_seq
            pred_label_ids = [tmp for tmp_idx, tmp in enumerate(pred_label_i) if tmp_idx in mask_index]
            pred_label_ids = pred_label_ids[1:-1]
            pred_entities = extract_entities(pred_label_ids, label_list) # {"start_end":[ids]}
            final_pred_entities.update(pred_entities)
            final_pred_list.append(pred_entities)

            pred_label = ["O"] * len(gold_label_ids)
            for key, ids in final_pred_entities.items():
                try:
                    pred_label[ids[0]] = "B"
                    for id in ids[1:]:
                        pred_label[id] = "I"
                except:
                    print(len(pred_label))
                    print(ids)
            pred_label = iob2_to_iobes(pred_label)
            gold_label = [label_map[l] for l in gold_label_ids]
            assert len(gold_label) == len(pred_label)

            pred_label_list.append(pred_label)

        log = {}
        if tokenizer is not None: # log is a dict
            group_questions = []
            sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
            for input_masks_i, input_ids_i in zip(every_input_masks, every_input_ids):
                tmp_mask = [tmp_idx for tmp_idx, tmp in enumerate(input_masks_i) if tmp != 0]
                tmp_input_ids = [tmp for tmp_idx, tmp in enumerate(input_ids_i) if tmp_idx in tmp_mask]
                tmp_ques_ids = tmp_input_ids[1:tmp_input_ids.index(sep_id)]
                group_questions.append(tokenizer.convert_ids_to_tokens(tmp_ques_ids))
            log["true_label"] = gold_label
            log["pred_label"] = pred_label
            log["doc_tokens"] = every_doc_tokens
            log["questions"] = [" ".join(q) for q in group_questions] # list
            logs.append(log)

        # compute the number of extracted entitities
        num_true = len(truth_entities) # 一个batch中gold truth entity的数目
        # num_extraction = len(final_pred_entities) # 一个batch中抽出来的数目
        num_extraction = [len(pred_i) for pred_i in final_pred_list]
        # num_true_positive = 0
        num_true_positive = [0] * num_ques # 对于当前的input example, 每个问题的tp

        for ques_idx, pred_i in enumerate(final_pred_list):
            for entity_idx in pred_i.keys():
                try:
                    if truth_entities[entity_idx] == pred_i[entity_idx]:
                        num_true_positive[ques_idx] += 1
                except:
                    pass

            dict_match = list(filter(lambda x: x[0] == x[1], zip(pred_label_list[ques_idx], gold_label)))
            accuracy = len(dict_match) / float(len(gold_label))

            # 累加，计算每个问题在整个dataset上的数目
            if every_input_type == "relation":  # relation
                rel_positive[ques_idx] += num_true_positive[ques_idx]  # true_positive
                rel_extracted[ques_idx] += num_extraction[ques_idx]  # num_extraction
                rel_true[ques_idx] += num_true  # num of true entities
                rel_accuracy[ques_idx] += accuracy
                if ques_idx == 0:
                    num_rel += 1
            elif every_input_type == "entity":  # 对于test来说，开始全是entity
                ent_positive[ques_idx] += num_true_positive[ques_idx]
                ent_extracted[ques_idx] += num_extraction[ques_idx]
                ent_true[ques_idx] += num_true
                ent_accuracy[ques_idx] += accuracy
                if ques_idx == 0:
                    num_ent += 1

    ent_results = []
    rel_results = []
    for ques_idx in range(num_ques):
        ent_acc, ent_precision, ent_recall, ent_f1 = compute_f1(ent_accuracy[ques_idx], ent_positive[ques_idx], ent_extracted[ques_idx], ent_true[ques_idx], num_ent)
        rel_acc, rel_ent_precision, rel_recall, rel_f1 = compute_f1(rel_accuracy[ques_idx], rel_positive[ques_idx], rel_extracted[ques_idx], rel_true[ques_idx], num_rel)

        ent_results.append({"accuracy": ent_acc, "precision": ent_precision, "recall": ent_recall, "f1": ent_f1})
        rel_results.append({"accuracy": rel_acc, "precision": rel_ent_precision, "recall": rel_recall, "f1": rel_f1})

    return {"entity":ent_results,
            "relation":rel_results}, logs


def compute_performance(input_ids, doc_tokens, input_masks, pred_labels, gold_labels, label_masks, input_types, label_list, tokenizer=None,
                        ent_weight=[1,1,1], rel_weight=[1,1,1]):
    '''
    :param input_ids: num_all_example, 3, max_seq
    :param doc_tokens: num_all_example, max_seq
    :param input_masks: num_all_example, 3, max_seq
    :param pred_labels: num_all_example, 3, max_seq
    :param gold_labels: num_all_example, max_seq
    :param label_masks: num_all_example, max_seq
    :param input_types:
    :param label_list:
    :param tokenizer:
    :return:
    '''
    label_map = {i: label for i, label in enumerate(label_list)}
    ent_accuracy, ent_positive, ent_extracted, ent_true = 0, 0, 0, 0
    rel_accuracy, rel_positive, rel_extracted, rel_true = 0, 0, 0, 0
    num_rel, num_ent = 0, 0
    logs = []
    for every_input_ids, every_doc_tokens, every_input_masks, every_pred_label, every_golden_label, every_label_masks, every_input_type in \
            zip(input_ids, doc_tokens, input_masks, pred_labels, gold_labels, label_masks, input_types):
        # every_doc_tokens: max_seq
        # batch_pred_label: 3, max_seq
        # extract gold_label
        mask_index = [tmp_idx for tmp_idx, tmp in enumerate(every_label_masks) if tmp != 0]
        gold_label_ids = [tmp for tmp_idx, tmp in enumerate(every_golden_label) if tmp_idx in mask_index]
        gold_label_ids = gold_label_ids[1:-1]
        truth_entities = extract_entities(gold_label_ids, label_list)

        # extract gold_label from multiple question answers
        final_pred_entities = {}
        max_pred_num = -1

        # vote on every token
        final_pred_label = []
        num_ques = len(every_pred_label)
        # every_pred_label: 3, max_seq
        for i in mask_index:  # vote on every token
            answer = [every_pred_label[j][i] for j in range(num_ques)]
            answer = []
            for j in range(num_ques):
                if every_input_type == "entity":
                    answer.extend([every_pred_label[j][i]] * int(ent_weight[j]*100))
                elif every_input_type == "relation":
                    answer.extend([every_pred_label[j][i]] * int(rel_weight[j]*100))
            final_answer = Counter(answer).most_common(1)[0][0]
            final_pred_label.append(final_answer)
        final_pred_label = final_pred_label[1:-1]
        final_pred_entities = extract_entities(final_pred_label, label_list)

        # if every_input_type == "entity":
        #     best_ques_idx = int(max(ent_weight))
        # elif every_input_type == "relation":
        #     best_ques_idx = int(max(rel_weight))
        # pred_label_ids = every_pred_label[best_ques_idx]
        # final_pred_label = [tmp for tmp_idx, tmp in enumerate(pred_label_ids) if tmp_idx in mask_index]
        # final_pred_label = final_pred_label[1:-1]
        # final_pred_entities = extract_entities(final_pred_label, label_list)

        pred_label = ["O"] * len(gold_label_ids)
        for key, ids in final_pred_entities.items():
            try:
                pred_label[ids[0]] = "B"
                for id in ids[1:]:
                    pred_label[id] = "I"
            except:
                print(len(pred_label))
                print(ids)
        pred_label = iob2_to_iobes(pred_label)
        gold_label = [label_map[l] for l in gold_label_ids]
        assert len(gold_label) == len(pred_label)

        if tokenizer is not None:
            log = {}
            group_questions = []
            sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
            for input_masks_i, input_ids_i in zip(every_input_masks, every_input_ids):
                tmp_mask = [tmp_idx for tmp_idx, tmp in enumerate(input_masks_i) if tmp != 0]
                tmp_input_ids = [tmp for tmp_idx, tmp in enumerate(input_ids_i) if tmp_idx in tmp_mask]
                tmp_ques_ids = tmp_input_ids[1:tmp_input_ids.index(sep_id)]
                group_questions.append(tokenizer.convert_ids_to_tokens(tmp_ques_ids))
            log["true_label"] = gold_label
            log["pred_label"] = pred_label
            log["doc_tokens"] = every_doc_tokens
            log["questions"] = [" ".join(q) for q in group_questions]
            logs.append(log)

        # compute the number of extracted entitities
        num_true = len(truth_entities)
        num_extraction = len(final_pred_entities)
        num_true_positive = 0
        for entity_idx in final_pred_entities.keys():
            try:
                if truth_entities[entity_idx] == final_pred_entities[entity_idx]:
                    num_true_positive += 1
            except:
                pass

        dict_match = list(filter(lambda x: x[0] == x[1], zip(pred_label, gold_label)))
        accuracy = len(dict_match) / float(len(gold_label))

        if every_input_type == "relation":  # relation
            rel_positive += num_true_positive  # true_positive
            rel_extracted += num_extraction  # num_extraction
            rel_true += num_true  # num of true entities
            rel_accuracy += accuracy
            num_rel += 1
        elif every_input_type == "entity":  # 对于test来说，开始全是entity
            ent_positive += num_true_positive
            ent_extracted += num_extraction
            ent_true += num_true
            ent_accuracy += accuracy
            num_ent += 1

    ent_accuracy, ent_precision, ent_recall, ent_f1 = compute_f1(ent_accuracy, ent_positive, ent_extracted, ent_true, num_ent)
    rel_accuracy, rel_ent_precision, rel_recall, rel_f1 = compute_f1(rel_accuracy, rel_positive, rel_extracted, rel_true, num_rel)

    return  {"entity": {"accuracy": ent_accuracy, "precision": ent_precision, "recall": ent_recall, "f1": ent_f1},
            "relation": {"accuracy": rel_accuracy, "precision": rel_ent_precision, "recall": rel_recall, "f1": rel_f1}}, logs


def compute_f1(acc, positive, extracted, true, num_example):

    precision = positive / float(extracted) if extracted != 0 else 0
    recall = positive / float(true) if true != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    new_acc = acc / num_example if num_example != 0 else 0
    new_acc, precision, recall, f1 = round(new_acc, 4), round(precision, 4), round(recall, 4), round(f1, 4)

    return new_acc, precision, recall, f1


def createRandomString(len):
    import random
    raw = ""
    range1 = range(58, 65)
    range2 = range(91, 97)

    i = 0
    while i < len:
        seed = random.randint(48, 122)
        if ((seed in range1) or (seed in range2)):
            continue
        raw += chr(seed)
        i += 1
    return raw


def generate_relation_examples(input_ids, doc_tokens, input_masks, pred_labels, gold_labels, label_masks, entity_types, golden_relations, label_list, config, tokenizer, ent_weight=[1,1,1]):
    '''
    :param input_ids: num_all_example, 3, max_seq
    :param doc_tokens: num_all_example, max_seq
    :param input_masks: num_all_example, 3, max_seq
    :param pred_labels: num_all_example, 3, max_seq
    :param gold_labels: num_all_example, max_seq
    :param label_masks: num_all_example, max_seq
    :param entity_types:
    :param golden_relations:
    :param label_list:
    :param config:
    :param tokenizer:
    :return:
    '''

    print("generate_relation_examples...")
    examples = []

    # batch_golden_relation: batch, 3
    for every_input_ids, every_doc_tokens, every_input_masks, every_pred_label, every_golden_label, every_label_masks, entity_type, golden_relation in \
            zip(input_ids, doc_tokens, input_masks, pred_labels, gold_labels, label_masks, entity_types, golden_relations):
        # batch_doc: 3, max_seq
        # batch_pred_label: 3, max_seq
        # extract gold_label
        mask_index = [tmp_idx for tmp_idx, tmp in enumerate(every_label_masks) if tmp != 0]
        gold_label_ids = [tmp for tmp_idx, tmp in enumerate(every_golden_label) if tmp_idx in mask_index]
        gold_label_ids = gold_label_ids[1:-1]
        truth_entities = extract_entities(gold_label_ids, label_list)

        # vote on every token
        final_pred_label = []
        num_ques = len(every_pred_label)
        # every_pred_label: 3, max_seq
        for i in mask_index:  # vote on every token
            answer = [every_pred_label[j][i] for j in range(num_ques)]
            answer = []
            for j in range(num_ques):
                answer.extend([every_pred_label[j][i]] * int(ent_weight[j]*100)) # only entity
            final_answer = Counter(answer).most_common(1)[0][0]
            final_pred_label.append(final_answer)
        final_pred_label = final_pred_label[1:-1]
        final_pred_entities = extract_entities(final_pred_label, label_list)

        # 3.select answer of the best question
        # best_ques_idx = int(max(ent_weight))
        # pred_label_ids = every_pred_label[best_ques_idx]
        # final_pred_label = [tmp for tmp_idx, tmp in enumerate(pred_label_ids) if tmp_idx in mask_index]
        # final_pred_label = final_pred_label[1:-1]
        # final_pred_entities = extract_entities(final_pred_label, label_list)

        for key, indicator in final_pred_entities.items():
            start_idx = int(key.split("_")[0])
            end_idx = int(key.split("_")[-1])
            head_entity_list = every_doc_tokens[start_idx:end_idx+1]
            orig_head_entity_text = " ".join(head_entity_list)

            for relation in entity_relation_map[entity_type]: # live_in, work_for, located_in ...
                questions = []
                for relation_template in question_templates[relation]: # 3 questions
                    question = relation_template.format(orig_head_entity_text)
                    questions.append(question)
                label = extract_relation_answer(golden_relation, relation, orig_head_entity_text, len(every_doc_tokens))
                example = MRCExample(
                    qas_id=createRandomString(10),
                    question_text=questions,
                    doc_tokens=every_doc_tokens,
                    label=label,
                    q_type="relation",
                    entity_type=entity_type,
                    relations=[]
                )
                examples.append(example)

    return examples


def extract_relation_answer(golden_relations, relation, head_entity_text, len_doc):
    tail_entities = []
    for golden_relation in  golden_relations:
        if relation == golden_relation["label"] and golden_relation["e1_text"].startswith(head_entity_text):
            tail_entities.append(golden_relation["e2_ids"])

    label = ["O"] * len_doc
    for ids in tail_entities:
        label[ids[0]] = "B"
        for id in ids[1:]:
            label[id] = "I"
    label = iob2_to_iobes(label)

    return label