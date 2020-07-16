#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm, trange
import torch

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
    SequentialSampler

class DataProcessor(object):
    # base class for data converts for sequence classification datasets
    def get_train_examples(self, data_dir):
        # get a collection of "InputExample" for the train set
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        # gets a collections of "InputExample" for the dev set
        raise NotImplementedError()

    def get_labels(self):
        # gets the list of labels for this data set
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        # reads a tab separated value file.
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

def generate_mini_batch_input(all_features, mini_batch_idx, config):
    batch = [all_features[idx] for idx in mini_batch_idx]
    input_ids = torch.tensor([[f.input_ids for f in group.input_features] for group in batch], dtype=torch.long)
    input_mask = torch.tensor([[f.input_mask for f in group.input_features] for group in batch], dtype=torch.long)
    segment_ids = torch.tensor([[f.segment_ids for f in group.input_features] for group in batch], dtype=torch.long)
    label_ids = torch.tensor([[f.label_id for f in group.input_features] for group in batch], dtype=torch.long)
    valid_ids = torch.tensor([[f.valid_id for f in group.input_features] for group in batch], dtype=torch.long)
    label_mask = torch.tensor([[f.label_mask for f in group.input_features] for group in batch], dtype=torch.long)
    # label_mask = np.array([[f.label_mask for f in group_f] for group_f in batch_features])
    input_types = [group.type for group in batch]
    # entity_types = [[f.entity_type for f in group_f] for group_f in batch]
    # relations = [[f.relations for f in group_f] for group_f in batch]
    entity_types = [group.entity_type for group in batch] # batch_size
    relations = [group.relations for group in batch]
    doc_tokens = [group.doc_tokens for group in batch]

    input_ids = input_ids.view(-1, config.max_seq_length)  # batch * 3, max_seq_length
    input_mask = input_mask.view(-1, config.max_seq_length)
    segment_ids = segment_ids.view(-1, config.max_seq_length)
    label_ids = label_ids.view(-1, config.max_seq_length)
    valid_ids = valid_ids.view(-1, config.max_seq_length)
    label_mask = label_mask.view(-1, config.max_seq_length)
    # label_mask = np.reshape(label_mask, (-1, config.max_seq_length))

    return input_ids, input_mask, segment_ids, label_ids, valid_ids, label_mask, input_types, entity_types, relations, doc_tokens

