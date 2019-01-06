"""
Constants file for InterpRecSys

@Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os


class Constant(object):
    """
    Constants of InterpRecSys
    """

    # Directory Structure
    # - data
    #   - graph: save graph objects
    #   - raw: all raw files
    #   - parse: all parsed files (order neighbors)

    DATA_DIR = os.getcwd() + "/data/"

    GRAPH_DIR = os.getcwd() + "/data/graph/"
    RAW_DIR = os.getcwd() + "/data/raw/"
    PARSE_DIR = os.getcwd() + "./data/parse/"

    RANDOM_SEED = 723


class Config(object):
    """
    Specific information for Dataset A
    """

    dataset = "a_dataset"

    TRAIN_FILE = Constant.PARSE_DIR + dataset + "/xxx_train.csv"
    TEST_FILE = Constant.PARSE_DIR + dataset + "/xxx_test.csv"
    CUS_FILE = Constant.PARSE_DIR + dataset + "/xxx_cus.csv"
    OBJ_FILE = Constant.PARSE_DIR + dataset + "/xxx_obj.csv"


    # Columns of categorical features
    CAT_COL = [
        "col_1",
        "col_2"
    ]

    # Columns of numeric features
    NUM_COL = [
        "num_1",
        "num_2"
    ]

    # Columns to be ignored
    IGN_COL = [
        "ign_1",
        "ign_2"
    ]
