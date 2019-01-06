"""
Preprocessing functions

@Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os
import pickle
import networkx as nx
import json

import pandas as pd

from .const import Constant
from .build_entity_graph import load_graph

DATA_DIR = Constant.GRAPH_DIR


def clk_thru_neg_sample(dataset):
    """
    TODO: What is this?
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

    TODO:
        (1) decide whether to split cus and obj
        (2) to implement
    """

    # TODO: implement dataset specific parsing func
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

    # TODO: split train and test in df_ct

    df_ct_train.to_csv(output_path + "/clk_thru_train.csv")
    df_ct_test.to_csv(output_path + "/clk_thru_test.csv")
    df_cus.to_csv(output_path, "/customer.csv")
    df_obj.to_csv(output_path, "/object.csv")

    print("Parsed Features has been saved in {}".format(output_path))
