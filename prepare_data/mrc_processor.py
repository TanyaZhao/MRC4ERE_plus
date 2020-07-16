import os
import sys
import csv

from .mrc_utils import *

class DataProcessor(object):
    # base class for data converts for sequence classification datasets
    def get_train_examples(self, data_dir):
        # get a collection of "InputExample" for the train set
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        # gets a collections of "InputExample" for the dev set
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

class MRCProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        train_examples = read_squad_examples(data_dir, is_training=True)
        return train_examples

    def get_dev_examples(self, data_dir):
        dev_examples = read_squad_examples(data_dir, is_training=False)
        return dev_examples

    def get_test_examples(self, data_dir):
        test_examples = read_squad_examples(data_dir, is_training=False)
        return test_examples

    def get_labels(self, datasets):
        label_list = ['[CLS]','[SEP]']
        for dataset in datasets:
            for example in dataset:
                for tmp in list(set(example.label)):
                    if tmp not in label_list:
                        label_list.append(tmp)

        return label_list

    def get_entity_types(self, datasets="conll04"):
        return ["loc", "peop", "org", "other"]






