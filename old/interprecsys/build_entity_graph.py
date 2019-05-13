"""
Build Entity Graph for Customer and Object

@Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

LOG:
    18-12-22:
        since we changed the starting point from graph to trm,
        we will keep the code in the current status and come back
        when trm part is done.
"""

import networkx as nx
import pandas as pd

import os
import sys
from itertools import combinations
import pickle

from const import Constant
from utils import entity_similarity


def _load_pickle(path):
    with open(path, "rb") as fin:
        ret = pickle.load(fin)
    return ret


def load_graph(data_set, threshold, is_cus):
    """
    Load full/partial graph.
    """

    role = "cus" if is_cus else "obj"
    in_path = os.path.join(Constant.GRAPH_DIR, data_set,  "/{}_{}.pkl".format(role, str(threshold)))
    return _load_pickle(in_path)


def load_dict(data_set, is_cus):
    """
    Load full/partial graph.
    """

    role = "cus" if is_cus else "obj"
    in_path = os.path.join(Constant.GRAPH_DIR, data_set, "/{}_id2ent.pkl".format(role))
    return _load_pickle(in_path)


def load_nbr_dict(data_set, threshold, is_cus):
    """
    Load 2 order neighbor graph
    """

    role = "cus" if is_cus else "obj"
    in_path = os.path.join(Constant.GRAPH_DIR, data_set, "/{}_{}_nbr.pkl".format(role, str(threshold)))
    return _load_pickle(in_path)


def create_or_load_graph_profile(dataset, load, save, is_cus, sim_metric="jaccard"):
    """
    load cus/obj from csv files

    :param dataset:
    :param load: boolean. If True, load graph; else, create graph
    :param save: boolean. Used when load_full is True.
    :param is_cus: boolean. If True, work on `cus` else `obj`.
    :param sim_metric: string. Similarity measurement

    :return: networkx object of G and dict of dict-
    """
    role = "_cus" if is_cus else "_obj"

    if not load:

        # ====== Read input files =====

        in_path = os.path.join(Constant.PARSE_DIR, dataset, role + ".csv")
        df_entities = pd.read_csv(in_path)

        # ====== Generate instance list and dict =====

        entities = df_entities.values.tolist()
        dict_entities = dict(enumerate(entities))  # build [id: feature] dict

        print("Loading Done!")

        # ===== Create a full graph =====

        G = nx.Graph()

        for entity_1, entity_2 in combinations(range(len(entities)), 2):
            feature_1, feature_2 = entities[entity_1], entities[entity_2]
            G.add_edge(entity_1, entity_2,
                       weight=entity_similarity(sim_metric, feature_1, feature_2))
        print("Building Full Weighted Graph Done!")

        if save:

            # check if fold exists
            out_path_prefix = os.path.join(Constant.GRAPH_DIR, dataset)
            if os.path.isdir(out_path_prefix):
                os.mkdir(out_path_prefix)

            # save [id: feature] dict and full graph
            with open(out_path_prefix + "/{}_id2ent.pkl".format(role), "wb") as fout:
                pickle.dump(dict_entities, fout)

            with open(out_path_prefix + "/{}_full_graph.pkl".format(role), "wb") as fout:
                pickle.dump(G, fout)

            print("Full graph & Dict stored at {}".format(out_path_prefix))

    else:
        # ===== Loading graph from file =====

        in_path_prefix = os.path.join(Constant.GRAPH_DIR, dataset)

        with open(in_path_prefix + "/{}_id2ent.pkl".format(role), "rb") as fin:
            dict_entities = pickle.load(fin)

        with open(in_path_prefix + "/{}_full_graph.pkl".format(role), "rb") as fin:
            G = pickle.load(fin)

        print("Loading Graph and Dict Done!")

    return G, dict_entities


def generate_subgraph(fullG, is_cus, edge_threshold):

    # ===== Filter qualified edges and building subgraph =====

    role = "cus" if is_cus else "obj"

    qualified_edges = [(u, v) for (u, v, d) in fullG.edges(data=True) if d["weight"] >= edge_threshold]
    subG = fullG.edge_subgraph(qualified_edges).copy()

    print("Build Subgraph Done!")

    # ===== Save the subgraph ======

    out_path = os.path.join(Constant.GRAPH_DIR, dataset, "{}_{}.pkl".format(role, str(edge_threshold)))
    with open(out_path, "wb") as fout:
        pickle.dump(subG, fout)

    print("All Done! The new subgraph has been stored at {}".format(out_path))


if __name__ == "__main__":
    """Generate a new Graph"""
    
    if len(sys.argv) < 1 + 1:
        print("Insufficient Parameters!")
        print("\tParameter list: [dataset], [if_load_full], [if_save_full], [edge_threshold]")
        print("\t[if_load] (bool) - 0: create; otherwise: load.")
        sys.exit(1)

    dataset = sys.argv[1]
    load_full = bool(sys.argv[2])  # bool: load full graph from file
    save_full = bool(sys.argv[3])  # bool: save full graph to file
    threshold = float(sys.argv[3])

    for is_cus in [True, False]:
        G, dict_ent = create_or_load_graph_profile(dataset,
                                                   load=load_full,
                                                   save=save_full,
                                                   is_cus=is_cus)

        generate_subgraph(G, is_cus=is_cus, edge_threshold=threshold)

