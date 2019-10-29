"""
Preprocessing functions

@Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os, sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from const import Constant, Config

DATA_DIR = Constant.GRAPH_DIR
NUM_BUCKET = 0


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

        [OLD] for numerical feature, do
            e.g. if col[num] = 1.4, tc = 10
                 then feat_dict[num] = 10

        [NOW] for numerical feature, do
            e.g. if norm(col[num]) = 0.24643, num_bucket = 10
                 then it would be mapped to 2.0
        """

        df = pd.concat([self.dfTrain, self.dfTest, self.dfVal], sort=False)
        tc = 0
        for col in df.columns:
            if col in self.cfg.IGN_COL:
                continue

            # commented below code because of bucketing mechanism
            # elif col in self.cfg.NUM_COL:
            #     self.feat_dict[col] = tc
            #     tc += 1

            else:
                us = df[col].unique()  # unique sample
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)

        self.feat_dim = tc

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
        elif 0 in dfi.columns:
            y_col_name = 0
        else:
            raise KeyError("Cannot find [label] or [target] column in the dataset.")

        y = dfi[y_col_name]
        dfi.drop([y_col_name], axis=1, inplace=True)

        if 'id' in dfi.columns:
            dfi.drop(['id'], axis=1, inplace=True)

        # [OLD]
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        # dfv = dfi.copy()

        for col in dfi.columns:

            if col in self.cfg.IGN_COL:
                dfi.drop(col, axis=1, inplace=True)
                # dfv.drop(col, axis=1, inplace=True)  # [OLD]

            # commented below lines because of bucketing
            # elif col in self.cfg.NUM_COL:
            #     # for numeric feature columns, leave dfv[col] == dfi[col]
            #     dfi[col] = self.feat_dict[col]

            else:
                dfi[col] = dfi[col].map(self.feat_dict[col])
                # `map` the cat feature to an id
                # dfv[col] = 1  # [OLD]

        # [Deprecated] Convert pd.DataFrame to np.ndarray to a list of list
        # xi = dfi.values.tolist()
        # xv = dfv.values.tolist()

        # [New] convert to np.ndarray instead of list
        # xi = dfi.values
        # xv = dfv.values

        # return xi, xv, y
        # return dfi, dfv, y  # [OLD]
        return dfi, y


def parse_criteo():
    """
    parse dataset criteoDAC

    """
    input_file = "criteoDAC/train.txt"
    input_dir = Constant.RAW_DIR + input_file

    exceptions = [16, 17, 25, 29, 34]
    cat_col = [x for x in range(14, 40)]
    num_col = [x for x in range(1, 14)]
    criteo_columns = ['label'] + num_col + cat_col

    print("Preprocessing criteo dataset ...")

    # load dataset
    print("\tLoading dataset ...")
    df = pd.read_csv(input_dir, header=None, sep="\t", names=criteo_columns)

    # drop columns
    df = df.drop(exceptions, axis=1)

    # fix missing value
    print("\tFixing missing values ...")
    df = _fix_missing_values(df)

    print("\tNormalizing numerical features ...")
    df = _norm_bucket_numerical(df, num_col)

    # split train, valid, and test
    print("\tSplitting Train, Valid, and Test dataset ...")
    df_train, df_val, df_test = _split_train_validation_test(df)

    # split ind, val, and label
    print("\tSplitting Index, Value, and Labels ...")
    full_splits = _split_ind_val_label(dataset="criteoDAC",
                                       df_train=df_train,
                                       df_test=df_test,
                                       df_val=df_val)

    # save 3X3 dataframes to `parsed` folder
    print("\tSaving all splited matrices ...")
    _save_splits(full_splits, dataset="criteoDAC")


def parse_avazu():
    input_file = "avazu/train"
    input_dir = Constant.RAW_DIR + input_file

    exceptions = ["device_id", "device_ip", "id"]

    print("Preprocessing avazu dataset ...")

    # load dataset
    print("\tLoading dataset ...")
    df = pd.read_csv(input_dir)

    # fix missing value -- no missing value

    # drop columns
    df = df.drop(Config("avazu").IGN_COL, axis=1)

    # normalizing features -- no numerical cols

    # split train, valid, and test
    print("\tSplitting Train, Valid, and Test Dataset ...")
    df_train, df_val, df_test = _split_train_validation_test(df)

    # split ind, val, and test
    print("\tSplitting Index, Value, and Labels ...")
    full_splits = _split_ind_val_label(dataset="avazu",
                                       df_train=df_train,
                                       df_test=df_test,
                                       df_val=df_val)

    # save 3X3 dataframes to `parsed` folder
    print("\tSaving all splited matrices ...")
    _save_splits(full_splits, dataset="avazu")


def parse_frappe():
    input_file = "frappe/train.csv"
    input_dir = Constant.RAW_DIR + input_file


    print("Preprocessing avazu dataset ...")

    # load dataset
    print("\tLoading dataset ...")
    df = pd.read_csv(input_dir, header=None)

    # fix missing value -- no missing value

    # drop columns
    df = df.drop(Config("frappe").IGN_COL, axis=1)

    # normalizing features -- no numerical cols

    # split train, valid, and test
    print("\tSplitting Train, Valid, and Test Dataset ...")
    df_train, df_val, df_test = _split_train_validation_test(df)

    # split ind, val, and test
    print("\tSplitting Index, Value, and Labels ...")
    full_splits = _split_ind_val_label(dataset="frappe",
                                       df_train=df_train,
                                       df_test=df_test,
                                       df_val=df_val)

    # save 3X3 dataframes to `parsed` folder
    print("\tSaving all splited matrices ...")
    _save_splits(full_splits, dataset="frappe")


def parse_vis():
    input_file = "vis/train.csv"
    input_dir = Constant.RAW_DIR + input_file


    print("Preprocessing avazu dataset ...")

    # load dataset
    print("\tLoading dataset ...")
    df = pd.read_csv(input_dir, header=None)

    # fix missing value -- no missing value

    # drop columns
    df = df.drop(Config("vis").IGN_COL, axis=1)

    # normalizing features -- no numerical cols

    # split train, valid, and test
    print("\tSplitting Train, Valid, and Test Dataset ...")
    df_train, df_val, df_test = _split_train_validation_test(df)

    # split ind, val, and test
    print("\tSplitting Index, Value, and Labels ...")
    full_splits = _split_ind_val_label(dataset="vis",
                                       df_train=df_train,
                                       df_test=df_test,
                                       df_val=df_val)

    # save 3X3 dataframes to `parsed` folder
    print("\tSaving all splited matrices ...")
    _save_splits(full_splits, dataset="vis")


def parse_safe_driver():
    dataset = "safedriver"
    input_file = "safedriver/train.csv"
    input_dir = Constant.RAW_DIR + input_file

    print("Preprocessing safedriver_medium dataset ...")

    # load dataset
    print("\tLoading dataset ...")
    df = pd.read_csv(input_dir)

    # fix missing value

    # split train, valid, and test
    print("\tSplitting Train, Valid, and Test Dataset ...")
    df_train, df_val, df_test = _split_train_validation_test(df)

    # split ind, val, and test
    print("\tSplitting Index, Value, and Labels ...")
    full_splits = _split_ind_val_label(dataset=dataset,
                                       df_train=df_train,
                                       df_test=df_test,
                                       df_val=df_val)

    # save 3X3 dataframes to `parsed` folder
    print("\tSaving all splited matrices ...")
    _save_splits(full_splits, dataset=dataset)


def _fix_missing_values(df):
    nan_convert_map = {
        'int64': 0,
        'float64': 0.0,
        'O': "00000000",
        'object': "00000000"
    }
    for col in df.columns:
        patch = nan_convert_map[str(df[col].dtype)]
        df[col] = df[col].fillna(patch)

    return df


def _split_train_validation_test(df):
    """
    split train, validation, test set.
    Using `train_test_split` twice.
    """

    df_train_val, df_test = train_test_split(
        df,
        test_size=0.1,
        shuffle=False)
        # random_state=666)

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=0.111,
        shuffle=False)

    return df_train, df_val, df_test


def _split_ind_val_label(dataset, df_train, df_test, df_val):
    feat_dict = FeatureDictionary(df_train=df_train,
                                  df_test=df_test,
                                  df_val=df_val,
                                  cfg=Config(dataset=dataset))

    # parse datasets
    df_train_split = feat_dict.parse(df=df_train)
    df_test_split = feat_dict.parse(df=df_test)
    df_val_split = feat_dict.parse(df=df_val)

    return df_train_split, df_val_split, df_test_split, feat_dict


def _save_splits(splits, dataset):

    usage = ["train", "val", "test"]
    term = ["ind", "label"]

    if not os.path.isdir(Constant.PARSE_DIR + dataset):
        os.mkdir(Constant.PARSE_DIR + dataset)

    for i, u in enumerate(usage):
        for j, t in enumerate(term):
            # np.savetxt(
            #     fname=Constant.PARSE_DIR + "{}/{}_{}.csv".format(dataset, u, t),
            #     X=splits[i][j],
            #     delimiter=','
            # )
            splits[i][j].to_csv(
                Constant.PARSE_DIR + "{}/{}_{}.csv".format(dataset, u, t),
                index=False,
                header=False
            )

    with open(Constant.PARSE_DIR + "{}/feat_dict".format(dataset), "w") as fout:
        # feature size field size
        fout.write("{} {}".format(splits[3].feat_dim, splits[0][0].shape[1]))


def _norm_bucket_numerical(df, num_col):
    """Normalizing and bucketing numerical features

    algo: target_bucket = normalized_value * num_bucket
    e.g.: 5 = 0.5 * 10

    """
    mms = MinMaxScaler(feature_range=(0, 1))
    df[num_col] = mms.fit_transform(df[num_col])

    # fixed size bucketing
    df[num_col] = np.floor(df[num_col] * NUM_BUCKET)
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2 + 1:
        sys.exit("format: python preprocess.py [dataset] [num_bucket]")

    dataset = sys.argv[1]
    NUM_BUCKET = int(sys.argv[2])
    print("preprocess: {} with num bucket {}".format(dataset, NUM_BUCKET))

    if dataset == "criteo":
        parse_criteo()
    elif dataset == "avazu":
        parse_avazu()
    elif dataset == "safedriver":
        parse_safe_driver()
    elif dataset == "movielens":
        raise NotImplementedError("To Be Implemented.")
    elif dataset == "frappe":
        parse_frappe()
    elif dataset == "vis":
        parse_vis()
    else:
        raise ValueError("Not supported `dataset` {}".format(dataset))
