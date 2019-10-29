"""
Constants file for InterpRecSys

Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
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
    PARSE_DIR = os.getcwd() + "/data/parse/"
    LOG_DIR = os.getcwd() + "/log/"

    RANDOM_SEED = 666


class Config:
    def __init__(self, dataset):
        self.dataset = dataset

        self.TRAIN_FILE = Constant.PARSE_DIR + dataset + "/train.csv"
        self.TEST_FILE = Constant.PARSE_DIR + dataset + "/test.csv"
        self.CUS_FILE = Constant.PARSE_DIR + dataset + "/cus.csv"
        self.OBJ_FILE = Constant.PARSE_DIR + dataset + "/obj.csv"

        if self.dataset == "criteoDAC":
            self.CAT_COL = [x for x in range(14, 40)]
            self.NUM_COL = [x for x in range(1, 14)]
            self.IGN_COL = [16, 17, 25, 29, 34]

        elif self.dataset == "avazu":
            self.CAT_COL = [
               "hour",
               "C1",
               "banner_pos",
               "site_id",
               "site_domain",
               "site_category",
               "app_id",
               "app_domain",
               "app_category",
               "device_model",
               "device_type",
               "device_conn_type"] + \
               ["C{}".format(str(x)) for x in range(14, 22)]

            self.NUM_COL = []

            self.IGN_COL = [
                "device_ip",
                "device_id",
                "id"
            ]

        else:
            raise ValueError("Invalid dataset {}".format(dataset))


