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
    RAW_DIR = os.getcwd() + "/data/raw/"
    PARSE_DIR = os.getcwd() + "/data/parse/"

    LOG_DIR = os.getcwd() + "/log/"

    PERF_DIR = os.getcwd() + "/performance/"

    RANDOM_SEED = 666


class Config:
    def __init__(self, dataset):
        self.dataset = dataset

        self.TRAIN_FILE = Constant.PARSE_DIR + dataset + "/train.csv"
        self.TEST_FILE = Constant.PARSE_DIR + dataset + "/test.csv"
        self.CUS_FILE = Constant.PARSE_DIR + dataset + "/cus.csv"
        self.OBJ_FILE = Constant.PARSE_DIR + dataset + "/obj.csv"

        # Columns of categorical features
        # CAT_COL, NUM_COL, IGN_COL
        if self.dataset == "safedriver":
            self.CAT_COL = [
                # ind
                'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat',
                "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin",
                "ps_ind_09_bin", "ps_ind_10_bin", "ps_ind_11_bin",
                "ps_ind_12_bin", "ps_ind_13_bin", "ps_ind_16_bin",
                "ps_ind_17_bin", "ps_ind_18_bin",
                # car
                'ps_car_01_cat', 'ps_car_02_cat',
                # 'ps_car_03_cat',
                'ps_car_04_cat',
                # 'ps_car_05_cat',
                'ps_car_06_cat',
                'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
                'ps_car_10_cat', 'ps_car_11_cat',
            ]

            self.NUM_COL = [
                # # binary
                # "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
                # "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin",
                # numeric
                "ps_ind_01", "ps_ind_03", "ps_ind_14", "ps_ind_15"
                "ps_reg_01", "ps_reg_02", "ps_reg_03",
                "ps_car_11", "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
            ]

            self.IGN_COL = [
                "id",
                "target",
                "ps_car_03_cat",
                "ps_car_05_cat",  # more than 40% missing
                "ps_calc_01",
                "ps_calc_02",
                "ps_calc_03",
                "ps_calc_04",
                "ps_calc_05",
                "ps_calc_06",
                "ps_calc_07",
                "ps_calc_08",
                "ps_calc_09",
                "ps_calc_10",
                "ps_calc_11",
                "ps_calc_12",
                "ps_calc_13",
                "ps_calc_14",
                "ps_calc_15_bin",
                "ps_calc_16_bin",
                "ps_calc_17_bin",
                "ps_calc_18_bin",
                "ps_calc_19_bin",
                "ps_calc_20_bin"  # because not correlated
            ]

        elif self.dataset == "criteoDAC":
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
        elif dataset == "frappe":
            self.CAT_COL = list(range(1, 11))
            self.IGN_COL = []
            self.NUM_COL = []

        elif dataset == "vis":
            self.CAT_COL = list(range(1, 11))
            self.IGN_COL = []
            self.NUM_COL = []

        else:
             raise ValueError("Invalid dataset {}".format(dataset))



