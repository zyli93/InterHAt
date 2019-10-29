"""
Data Loader and Feature Dictionary classes for InterpRecSys

NOTE: some code borrowed from here
    https://github.com/princewen/tensorflow_practice/blob/master/recommendation/Basic-DeepFM-model/data_reader.py

@Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import pickle

from itertools import product
import numpy as np
import pandas as pd


from build_entity_graph import load_graph, load_dict, load_nbr_dict
from const import Constant, Config
DATA_DIR = Constant.PARSE_DIR


class DataLoader:
    def __init__(self
                 , dataset
                 , batch_size):
        """
        :param: dataset: name of dataset
        :param: use_graph: whether need to build graph
        :param: batch_size:
        """

        # ==== params =====
        self.dataset = dataset
        self.cfg = Config(dataset=dataset)

        # ===== sizes =====
        self.batch_size = batch_size
        self.train_size, self.test_size, self.valid_size = 0, 0, 0

        # ===== inner variables =====
        self.batch_index = 0
        self.has_next = False

        # ===== datasets =====
        self.train_ind, self.train_label = self.load_data("train")
        self.test_ind, self.test_label = self.load_data("test")
        self.val_ind, self.val_label = self.load_data("val")

        self.train_size = self.train_label.shape[0]
        self.test_size = self.test_label.shape[0]
        self.val_size = self.val_label.shape[0]
        self.feature_size, self.field_size = self.load_statistics()

        # ===== iter count =====
        self.train_iter_count = self.train_size // self.batch_size

    def load_data(self, usage):
        """
        usage as one of `train`, `test`, `val`

        :param usage:
        :return:
            usage_ind.np.array
            usage_label.np.array
        """
        if usage not in ["train", "test", "val"]:
            raise ValueError
        terms = ["ind", "label"]
        ret_sets = []
        data_dir = Constant.PARSE_DIR + self.dataset + "/"
        for trm in terms:
            ret_sets.append(
                pd.read_csv(
                    data_dir + "{}_{}.csv".format(usage, trm), header=None).values)

        return ret_sets

    def generate_train_batch_ivl(self):
        bs, bi = self.batch_size, self.batch_index
        end_ind = min((bi + 1) * bs, self.train_size)

        b_ind = self.train_ind[bs * bi:end_ind]
        b_label = self.train_label[bs * bi: end_ind]

        self.batch_index += 1
        if self.batch_index == self.train_iter_count:
            self.batch_index = 0
            self.has_next = False

        return b_ind, b_label

    def generate_test_ivl(self):
        bs = self.batch_size
        batch_count = self.test_size // bs + 1

        for bi in range(batch_count):
            end_index = min((bi + 1) * bs, self.test_size)
            batch_ind = self.test_ind[bs * bi: end_index]
            batch_label = self.test_label[bs * bi: end_index]
            yield batch_ind, batch_label

    # TODO in later versions: combine test & val
    def generate_val_ivl(self):
        bs = self.batch_size
        batch_count = self.val_size // bs + 1

        for bi in range(batch_count):
            end_ind = min((bi + 1) * bs, self.val_size)
            batch_ind = self.val_ind[bs * bi: end_ind]
            batch_label = self.val_label[bs * bi: end_ind]
            yield batch_ind, batch_label

    def load_statistics(self):
        with open(Constant.PARSE_DIR + "{}/feat_dict".format(self.dataset), "r") as fin:
            feat_size, field_size = [int(x) for x in fin.readline().split(" ")]
        return feat_size, field_size



