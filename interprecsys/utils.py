"""
Util functions
"""

import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import roc_auc_score, log_loss

from const import Constant


# ===== Entity Similarity Measurement =====
def entity_similarity(algo, entity1, entity2):
    """
    measure the similarity of the two entities
    :param algo: the algorithm to measure similarity
    :param entity1, entity2: two entities to measure analogy
    :return: [float] in [0. 1)
    """
    if algo == "jaccard":
        return jaccard_similarity_score(entity1, entity2)
    else:
        raise NotImplementedError


# ===== Activation Function Factory =====
def activation_options(activation_func):
    """
    Return activation function option

    Example::

        >>> actv_func = activation_options("relu")
        >>> before_actv = None
        >>> after_actv = actv_func(before_actv, name="Activation")

    """
    if activation_func == "relu":
        return tf.nn.relu
    elif activation_func == "tanh":
        return tf.nn.tanh
    else:
        raise ValueError("Unrecognized Activation Function {}".format(activation_func))


# ===== Plotting =====
def plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1] + 1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d" % (i + 1))
        legends.append("valid-%d" % (i + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s" % model_name)
    plt.legend(legends)
    plt.savefig("fig/%s.png" % model_name)
    plt.close()


def create_folder_tree(dataset):
    for x in [
        Constant.DATA_DIR,
        Constant.GRAPH_DIR,
        Constant.RAW_DIR,
        Constant.PARSE_DIR,
        Constant.LOG_DIR,
        Constant.PARSE_DIR + "/{}".format(dataset)
    ]:
        if not os.path.isdir(x):
            os.mkdir(x)


def evaluate_metrics(y_true, y_predict):
    """
    evaluate performance by AUC and LogLoss
    :param y_true: 1D numpy array
    :param y_predict: 1D numpy array
    :return:
    """

    auc = roc_auc_score(y_true=y_true, y_score=y_predict)
    lloss = log_loss(y_true=y_true, y_pred=y_predict)

    return auc, lloss


def build_msg(stage,
              epoch,
              iteration=None,
              global_step=None,
              logloss=None,
              regloss=None,
              auc=None):
    # build msg
    time = datetime.now().isoformat()[:24]
    msg = ("[{},{}] epoch:{} iter:{} global_step:{} logloss:{:.6f} "
           "regloss:{:.6f} AUC:{:.6f}"
           .format(stage,
                   time,
                   epoch,
                   iteration if iteration else "NONE",
                   global_step if global_step else "NONE",
                   logloss,
                   regloss if regloss else 0.0,
                   auc)
           )

    return msg
