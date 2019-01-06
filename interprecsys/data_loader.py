"""
Data Loader and Feature Dictionary classes for InterpRecSys

NOTE: some code borrowed from here
    https://github.com/princewen/tensorflow_practice/blob/master/recommendation/Basic-DeepFM-model/data_reader.py

@Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

from itertools import product
import numpy as np
import pandas as pd

from .build_entity_graph import load_graph, load_dict, load_nbr_dict
from .const import Constant, Config


DATA_DIR = Constant.PARSE_DIR  # TODO: manage folder tree


class DataLoader:
    def __init__(self
                 , dataset
                 , use_graph
                 , entity_graph_threshold
                 , batch_size):
        """
        :param: dataset: name of dataset
        :param: use_graph: whether need to build graph
        :param: entity_graph_threshold:
        :param: batch_size:
        """

        # ==== params =====
        self.use_graph = use_graph
        self.dataset = dataset

        self.batch_size = batch_size

        # ===== inner variables =====
        self.batch_index = 0
        self.has_next = False

        # ===== load dataset =====
        # TODO: X_train, X_test not used.
        self.df_train, self.df_test, \
            self.X_train, self.y_train, \
            self.X_test, self.ids_test, \
            self.cat_feat_idx = self.load_data()

        # ===== create feature dictionary =====
        self.feat_dict = FeatureDictionary(df_train=self.df_train,
                                           df_test=self.df_test)

        self.X_train_ind, self.X_train_val, self.y_train = \
            self.parse(df=self.df_train, has_label=True)
        self.X_test_ind, self.X_test_val, self.ids_test = \
            self.parse(df=self.df_train)

        self.feature_size = self.feat_dict.feat_dim
        self.field_size = len(self.X_train_ind[0])

        self.trainset_size = len(self.X_train_ind[1])

        # version with GraphSAGE (complex)
        # TODO: cannot connect to read csv file
        if self.use_graph:

            self.cus_G = load_graph(dataset, entity_graph_threshold, is_cus=True)  # 1 for cus
            self.cus_dict = load_dict(dataset, is_cus=True)
            self.cus_nbr = load_nbr_dict(dataset, entity_graph_threshold, is_cus=True)

            self.obj_G = load_graph(dataset, entity_graph_threshold, is_cus=False)  # 0 for obj
            self.obj_dict = load_dict(dataset, is_cus=False)
            self.obj_nbr = load_nbr_dict(dataset, entity_graph_threshold, is_cus=False)

            """
            TODO: here's a problem. what exactly do graph-version need for input. Is it all raw features?
            Or just column from embedding matrix.
            """


    def generate_train_batch(self):
        """
        generate training batch

        :return:
            - mixed batch of clk-thru and non-clk-thru
            - there labels

        """
        bs, bi = self.batch_size, self.batch_index
        if (bi + 1) * bs < self.trainset_size:
            batch_ind = self.X_train_ind[bi * bs: (bi + 1) * bs]
            batch_val = self.X_train_val[bi * bs: (bi + 1) * bs]
            self.batch_index += 1
        else:
            batch_ind = self.X_train_ind[bi * bs:]
            batch_val = self.X_train_val[bi * bs:]
            self.batch_index = 0
            self.has_next = False
            self._shuffle_data(is_train=True)

        if self.use_graph:
            raise NotImplementedError
        else:
            return batch_ind, batch_val

    def generate_test_batch(self):
        """
        TODO: simple version for now.
        """
        return self.X_test_ind, self.X_test_val, self.ids_test

    def _shuffle_data(self, is_train):
        """
        Shuffle data.

        [NOTE] self.X_train, self.X_test not used.
        :param is_train:
        """
        s = np.arange(self.trainset_size)
        np.random.shuffle(s)
        if is_train:
            self.X_train_val = self.X_train_val[s]
            self.X_train_ind = self.X_train_ind[s]
            self.y_train = self.y_train[s]
        else:
            self.X_test_val = self.X_test_val[s]
            self.X_test_ind = self.X_test_ind[s]
            self.ids_test = self.ids_test[s]

    def load_data(self):
        """
        load mixed train and test data
        """

        df_train = pd.read_csv(Config.TRAIN_FILE)
        df_test = pd.read_csv(Config.TEST_FILE)

        # process missing feature
        cols = [c for c in df_train.columns if c not in ['id', 'target']]

        df_train["missing_feat"] = np.sum((df_train[cols] == -1).values, axis=1)
        df_test["missing_feat"] = np.sum((df_test[cols] == -1).values, axis=1)

        # convert pd.DataFrame to np.ndarray
        X_train = df_train[cols].values
        y_train = df_train['target'].values

        X_test = df_test[cols].values
        ids_test = df_test['id'].values

        cat_feat_idx = [i for i, c in enumerate(cols) if c in Config.CAT_COL]

        return df_train, df_test, X_train, y_train, X_test, ids_test, cat_feat_idx

    def parse(self, df=None, has_label=False):

        dfi = df.copy()

        # discriminate train or test
        if has_label:
            # y = dfi['target'].values.tolist()
            y = dfi['target'].values
            dfi.drop(['id', 'target'], axis=1, inplace=True)
        else:
            # ids = dfi['id'].values.tolist()
            ids = dfi['id'].values
            dfi.drop(['id'], axis=1, inplace=True)

        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)

        dfv = dfi.copy()
        for col in dfi.columns:

            if col in Config.IGN_COL:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)

            elif col in Config.NUM_COL:
                # for numeric feature columns, leave dfv[col] == dfi[col]
                dfi[col] = self.feat_dict.feat_dict[col]

            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                # `map` the cat feature to an id
                dfv[col] = 1.

        # [Deprecated] Convert pd.DataFrame to np.ndarray to a list of list
        # xi = dfi.values.tolist()
        # xv = dfv.values.tolist()

        # [New] convert to np.ndarray instead of list
        xi = dfi.values
        xv = dfv.values

        if has_label:  # train
            return xi, xv, y
        else:  # test
            return xi, xv, ids


# Feature Dictionary of click through
class FeatureDictionary(object):
    def __init__(self,
                 df_train,
                 df_test):

        self.dfTrain = df_train
        self.dfTest = df_test

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

        df = pd.concat([self.dfTrain, self.dfTest])
        tc = 0
        for col in df.columns:
            if col in Config.IGN_COL:
                continue
            elif col in Config.NUM_COL:
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)

        self.feat_dim = tc
