"""
Preprocessing functions

@Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os, sys
import pickle
import networkx as nx
import json

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from const import Constant, Config
from build_entity_graph import load_graph
from data_loader import FeatureDictionary

from itertools import product


DATA_DIR = Constant.GRAPH_DIR


def clk_thru_neg_sample(dataset):
    """
    Create negative samples for Click Through
    """
    pass


# graph
def parse_two_order_neighbors(dataset, threshold, is_cus):
    """
    Load Graphs and generate all 1st order and 2nd order neighbors
    for all nodes.

    The output file
    """
    G = load_graph(dataset, threshold, is_cus=is_cus)

    order_1st = {}  # First order neighbors
    order_2nd = {}  # Second order neighbors

    for node in G.nodes():
        order_1st[node] = list(G[node])

    for node in G.nodes():
        node_second_neighbor = set()
        for node_2 in order_1st[node]:
            node_second_neighbor.update(order_1st[node_2])
        order_2nd[node] = list(node_second_neighbor)

    node_nbr_dict = dict()
    for node in G.nodes():
        node_nbr_dict[node] = {
            "1st_order_nbr": order_1st[node],
            "2nd_order_nbr": order_2nd[node]
        }

    role = "cus" if is_cus else "obj"
    output_path = os.path.join(Constant.GRAPH_DIR, dataset, "{}_{}_nbr.pkl".format(role, threshold))
    with open(output_path, "wb") as fout:
        pickle.dump(node_nbr_dict, fout)

    print("Parsed neighbor files saved at {}".format(output_path))


# clk_thru
def parse_features(infile, dataset):
    """
    Load entity features and transfer

    NOT USED
    """

    # Notes: (1) return pandas dataframe
    #        (2) split cus and obj feature by prefix in cols
    def parse_dataset_a():
        input_path = Constant.RAW_DIR + "dsA"  # Or maybe more than 1 file?
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def parse_dataset_b():
        input_path = Constant.RAW_DIR + "dsB"  # Or maybe more than 1 file?
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    output_path = os.path.join(Constant.DATA_DIR, dataset)

    df_ct_train, df_ct_test, df_cus, df_obj = None, None, None, None
    if dataset == "A":
        df_ct, df_cus, df_obj = parse_dataset_a()
    elif dataset == "B":
        df_ct, df_cus, df_obj = parse_dataset_b()

    df_ct_train.to_csv(output_path + "/train.csv")
    df_ct_test.to_csv(output_path + "/test.csv")
    df_cus.to_csv(output_path, "/customer.csv")
    df_obj.to_csv(output_path, "/object.csv")

    print("Parsed Features has been saved in {}".format(output_path))


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
    df = _normalizing_numerical(df, num_col)

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
        # shuffle=False)
        random_state=666)

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
    term = ["ind", "value", "label"]

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


def _normalizing_numerical(df, num_col):
    mms = MinMaxScaler(feature_range=(0, 1))
    df[num_col] = mms.fit_transform(df[num_col])
    return df


if __name__ == "__main__":
    if len(sys.argv) < 1 + 1:
        sys.exit("format: python preprocess.py [dataset]")

    dataset = sys.argv[1]

    if dataset == "criteo":
        parse_criteo()
    elif dataset == "avazu":
        parse_avazu()
    elif dataset == "safedriver":
        parse_safe_driver()
    elif dataset == "movielens":
        raise NotImplementedError("To Be Implemented.")
    else:
        raise ValueError("Not supported `dataset` {}".format(dataset))
