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
        self.train_val, self.train_ind, self.train_label = self.load_data("train")
        self.test_val, self.test_ind, self.test_label = self.load_data("test")
        self.val_val, self.val_ind, self.val_label = self.load_data("val")

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
            usage_val.np.array
            usage_ind.np.array
            usage_label.np.array
        """
        if usage not in ["train", "test", "val"]:
            raise ValueError
        terms = ["value", "ind", "label"]
        ret_sets = []
        data_dir = Constant.PARSE_DIR + self.dataset + "/"
        for trm in terms:
            ret_sets.append(
                pd.read_csv(
                    data_dir + 
                    "{}_{}.csv".format(usage, trm), header=None).values
            )

        return ret_sets

    def generate_train_batch_ivl(self):
        bs, bi = self.batch_size, self.batch_index
        end_ind = min((bi + 1) * bs, self.train_size)

        b_ind = self.train_ind[bs * bi:end_ind]
        b_value = self.train_val[bs * bi: end_ind]
        b_label = self.train_label[bs * bi: end_ind]

        self.batch_index += 1
        if self.batch_index == self.train_iter_count:
            self.batch_index = 0
            self.has_next = False

        return b_ind, b_value, b_label

    def generate_test_ivl(self):
        return self.test_ind, self.test_val, self.test_label

    def generate_val_ivl(self):
        return self.val_ind, self.val_val, self.val_label

    def load_statistics(self):
        with open(Constant.PARSE_DIR + "{}/feat_dict".format(self.dataset), "r") as fin:
            feat_size, field_size = [int(x) for x in fin.readline().split(" ")]
        return feat_size, field_size


# Feature Dictionary of click through
class FeatureDictionary(object):
    def __init__(self,
                 df_train,
                 df_test,
                 df_val,
                 cfg):

        self.dfTrain = df_train
        self.dfTest = df_test
        self.dfVal = df_val
        self.cfg = cfg

        self.feat_dim = 0
        self.feat_dict = {}

        self.gen_feat_dict()

    def gen_feat_dict(self):
        """
        generate feature dictionary

        for categorical feature, do one-hot encoding
            e.g. if col[cat].unique() = 'x', 'y', 'z', tc = 3
                 then feat_dict[cat] = { 'x': 3, 'y': 4, 'z': 5}
        for numeric feature, do
            e.g. if col[num] = 1.4, tc = 10
                 then feat_dict[num] = 10
        """

        df = pd.concat([self.dfTrain, self.dfTest, self.dfVal], sort=False)
        tc = 0
        for col in df.columns:
            if col in self.cfg.IGN_COL:
                continue
            elif col in self.cfg.NUM_COL:
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)

        self.feat_dim = tc

        # * Debug *
        # with open("feat_dict_debug", "wb") as fout:
        #     pickle.dump(self, fout)

    def parse(self, df=None):

        if not self.feat_dict:
            raise ValueError("feat_dict is empty!!")

        dfi = df.copy()

        # discriminate train or test
        # y = dfi['target'].values.tolist()
        # y = dfi['label'].values
        if 'label' in dfi.columns:
            y_col_name = "label"
        elif 'target' in dfi.columns:
            y_col_name = "target"
        elif 'click' in dfi.columns:
            y_col_name = "click"
        else:
            raise KeyError("Cannot find [label] or [target] column in the dataset.")

        y = dfi[y_col_name]
        dfi.drop([y_col_name], axis=1, inplace=True)

        if 'id' in dfi.columns:
            dfi.drop(['id'], axis=1, inplace=True)

        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)

        dfv = dfi.copy()
        for col in dfi.columns:

            if col in self.cfg.IGN_COL:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)

            elif col in self.cfg.NUM_COL:
                # for numeric feature columns, leave dfv[col] == dfi[col]
                dfi[col] = self.feat_dict[col]

            else:
                dfi[col] = dfi[col].map(self.feat_dict[col])
                # `map` the cat feature to an id
                dfv[col] = 1.

        # [Deprecated] Convert pd.DataFrame to np.ndarray to a list of list
        # xi = dfi.values.tolist()
        # xv = dfv.values.tolist()

        # [New] convert to np.ndarray instead of list
        # xi = dfi.values
        # xv = dfv.values

        # return xi, xv, y
        return dfi, dfv, y
