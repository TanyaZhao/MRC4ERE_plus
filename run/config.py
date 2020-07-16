#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import argparse
import json

from configparser import SafeConfigParser

class Configurable:

    def __init__(self, config_file, extra_args, logger):

        config = SafeConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = { k[2:] : v for k, v in zip(extra_args[0::2], extra_args[1::2]) }
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
                    logger.info(section + "-" + k + "-" + v)
        self._config = config
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, "minibatch"))
        config.write(open(self.config_file, "w", encoding="utf8"))
        logger.info("Loaded config file successful.")
        for section in config.sections():
            for k, v in config.items(section):
                logger.info(k + ": " + v)

        bert_config_file = self._config.get('Bert', 'bert_config')
        with open(bert_config_file, "r", encoding='utf-8') as reader:
            self.bert_config_json = json.load(reader)


    @property
    def bert_model(self):
        return self._config.get('Bert', 'bert_model')

    @property
    def bert_config(self):
        return self.bert_config_json["bert_config"]

    @property
    def bert_frozen(self):
        return self.bert_config_json["bert_frozen"]

    @property
    def hidden_size(self):
        return self.bert_config_json["hidden_size"]

    @property
    def hidden_dropout_prob(self):
        return self.bert_config_json["hidden_dropout_prob"]

    @property
    def classifier_sign(self):
        return self.bert_config_json["classifier_sign"]

    @property
    def clip_grad(self):
        return self.bert_config_json["clip_grad"]

    @property
    def use_cuda(self):
        return self._config.getboolean('Run', 'use_cuda')

    @property
    def loss_type(self):
        return self._config.get('Run', 'loss_type')

    @property
    def task_name(self):
        return self._config.get('Run', 'task_name')

    @property
    def do_train(self):
        return self._config.getboolean('Run', 'do_train')

    @property
    def do_eval(self):
        return self._config.getboolean('Run', 'do_eval')

    @property
    def train_batch_size(self):
        return self._config.getint('Run', 'train_batch_size')

    @property
    def dev_batch_size(self):
        return self._config.getint('Run', 'dev_batch_size')

    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def checkpoint(self):
        return self._config.getint('Run', 'checkpoint')

    @property
    def learning_rate(self):
        return self._config.getfloat('Run', 'learning_rate')

    @property
    def epochs(self):
        return self._config.getfloat('Run', 'epochs')

    @property
    def warmup_proportion(self):
        return self._config.getfloat('Run', 'warmup_proportion')

    @property
    def local_rank(self):
        return self._config.getint('Run', 'local_rank')

    @property
    def gradient_accumulation_steps(self):
        return self._config.getint('Run', 'gradient_accumulation_steps')

    @property
    def seed(self):
        return self._config.getint('Run', 'seed')

    @property
    def export_model(self):
        return self._config.getboolean('Run', 'export_model')

    @property
    def max_seq_length(self):
        return self._config.getint('Data', 'max_seq_length')

    @property
    def max_query_length(self):
        return self._config.getint('Data', 'max_query_length')

    @property
    def doc_stride(self):
        return self._config.getint('Data', 'doc_stride')

    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')

    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')

    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')

    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')

    @property
    def output_dir(self):
        return self._config.get('Save', 'output_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def result_dir(self):
        return self._config.get('Save', 'result_dir')



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/default.cfg')
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
