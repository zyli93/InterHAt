"""Hybrid model of Attentive FM and Graph Attention Network

This is a complex hybrid model of "Attentive Factorization Machine and Graph Attention Network",
    "Graph Attention Network", and "Graph Convolutional Network".

@author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import tensorflow as tf

from const import Constant
from modules import multihead_attention, feedforward, embedding
from module2 import single_attention


class Interprecsys:
    def __init__(self
                 , hidden_dimension
                 , num_cus_feat
                 , num_obj_feat
                 , ):
        """

        :param hidden_dimension: Dimension of hidden variable features
        :param num_cus_feat: int
        :param num_obj_feat: int
        """

        # ===== parameters =====
        self.hidden_dimension = hidden_dimension
        self.num_cus_feat = num_cus_feat
        self.num_obj_feat = num_obj_feat

        # ===== tensors/placeholders to be initialized =====

        #   placeholders
        self.input_feature_placeholders = {}
        self.input_label_placeholder = None

        #   embeddings
        self.feat_hid_emb_lookup = None
        self.feat_embedding_tensors = {}

    def reg_placeholders_variables(self):
        """
        Register all placeholders/trainable_params needed later.

        :return: None
        """
        # ===== Define placeholders for the input ======

        with tf.name_scope("input") as scope:
            # convert all continuous features to discrete features and therefore no `cat/obj_num_feat`.
            self.input_feature_placeholders = {
                "cus_2nd_feat": tf.placeholder(dtype=tf.int32, shape=[None, self.num_cus_feat],
                                               name="customer_2nd_neighbor_feat"),
                "obj_2nd_feat": tf.placeholder(dtype=tf.int32, shape=[None, self.num_obj_feat],
                                               name="object_2nd_neighbor_feat"),
                "cus_1st_feat": tf.placeholder(dtype=tf.int32, shape=[None, self.num_cus_feat],
                                               name="customer_1st_neighbor_feat"),
                "obj_1st_feat": tf.placeholder(dtype=tf.int32, shape=[None, self.num_obj_feat],
                                               name="object_1st_neighbor_feat"),
                "cus_feat": tf.placeholder(dtype=tf.int32, shape=[None, self.num_cus_feat],
                                           name="customer_feature"),
                "obj_feat": tf.placeholder(dtype=tf.int32, shape=[None, self.num_obj_feat],
                                           name="object_feature")
            }
            self.input_label_placeholder = tf.placeholder(dtype=tf.int32, shape=None, name="clk_thru_label")

        # ===== Define embedding layers =====

        self.feat_hid_emb_lookup = tf.get_variable(name="feature_hidden_embeddings",
                                                   shape=[self.num_cus_feat + self.num_obj_feat,
                                                          self.hidden_dimension])

        # ===== Define AGG kernels =====

        # ===== Define Attention kernels =====

    def build_graph(self):
        """
        Building computational graph, modularize by `with`.

        TODO: implementation
        """

        # ===== Get embeddings for input features =====

        with tf.name_scope("input_to_embeddings") as scope:
            for feat_name, feat_tensor in self.input_feature_placeholders.items():
                self.feat_embedding_tensors[feat_name] = tf.nn.embedding_lookup(self.feat_hid_emb_lookup, feat_tensor)

        # ===== Apply GraphSAGE AGG/CONCAT/NONLINEAR =====


# ===== InterpRecSys Base Model =====
class InterprecsysBase:

    """
    TODO: write tensorboard summary
    """
    def __init__(self
                 , embedding_dim
                 , field_size
                 , feature_size
                 , learning_rate
                 , batch_size
                 , num_block
                 , num_head
                 , dropout_rate
                 , regularization_weight
                 , random_seed=Constant.RANDOM_SEED
                 , scale_embedding=False
                 ):
        # config parameters
        self.embedding_dim = embedding_dim  # the C
        self.scale_embedding = scale_embedding  # bool
        self.field_size = field_size
        self.feat_size = feature_size  # the T TODO: some places misused

        self.dropout_rate = dropout_rate
        self.random_seed = random_seed
        self.num_block = num_block  # num of blocks of multi-head attn
        self.num_head = num_head  # num of heads
        self.regularization_weight = regularization_weight

        # training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # variables [None]
        self.embedding_lookup = None
        self.emb = None  # raw features

        # placeholders
        self.X_ind, self.X_val, self.label = None, None, None
        self.is_training = None

        # ports to the outside
        self.loss, self.mean_loss = None, None
        self.predict, self.acc = None, None

        # global training steps
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # operations
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=0.9,
                                                beta2=0.98,
                                                epsilon=1e-8)
        self.train_op, self.merged = None, None

        self.build_graph()

    def build_graph(self):

        # Define input
        with tf.name_scope("input_ph"):
            self.X_ind = tf.placeholder(dtype=tf.int32, shape=[None, self.field_size], name="X_index")
            self.X_val = tf.placeholder(dtype=tf.float32, shape=[None, self.field_size], name="X_value")
            self.label = tf.placeholder(dtype=tf.float32, shape=None, name="label")
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name="is_training")

        # lookup and process embedding
        with tf.name_scope("embedding"):
            self.emb = embedding(inputs=self.X_ind,
                                 vocab_size=self.feat_size,
                                 num_units=self.embedding_dim,
                                 scale=self.scale_embedding,
                                 scope="embedding_process")

            # multiply embeddings with values (for numeric features)
            self.emb = tf.multiply(self.emb, tf.expand_dims(self.X_val, axis=2))

        # self.emb: raw embedding, features: used for later
        features = self.emb

        # apply dropout on embedding
        with tf.name_scope("dropout"):
            features = tf.layers.dropout(features,
                                         rate=self.dropout_rate,
                                         training=self.is_training)

        # multi-layer, multi-head attention
        with tf.name_scope("multilayer_attn"):
            for i_block in range(self.num_block):
                with tf.variable_scope("attn_head_{}".format(str(i_block))) as scope:
                    # Multihead Attention
                    features = multihead_attention(queries=features,
                                                   keys=features,
                                                   num_units=self.embedding_dim,
                                                   num_heads=self.num_head,
                                                   dropout_rate=self.dropout_rate,
                                                   causality=False,
                                                   is_training=self.is_training)

                    # Feed Forward
                    features = feedforward(inputs=features,
                                           num_units=[4 * self.embedding_dim, self.embedding_dim],
                                           scope=scope)  # (N, T, C)

        # build second order cross
        # TODO: could be other functions
        # TODO: find out difference between name_scope and variable_scope
        # TODO: some of var names are `over-scoped`.
            # with tf.name_scope("sec_order_cross"):

        with tf.variable_scope("sec_order_cross") as scope:
            second_cross = tf.layers.conv1d(inputs=tf.transpose(features, [0, 2, 1]),  # transpose: (N, C, T)
                                            filters=1,
                                            kernel_size=1,
                                            activation=tf.nn.relu,
                                            use_bias=True)  # (N, C, 1)
            second_cross = tf.transpose(second_cross, [0, 2, 1])  # (N, 1, C)

            third_cross = single_attention(queries=second_cross,
                                           keys=self.emb,
                                           values=self.emb,
                                           scope=scope,
                                           regularize=True)  # (N, 1, C)

        with tf.name_scope("combined_features"):
            # concatenate enc, second_cross, and third_cross
            all_features = tf.concat([self.emb, second_cross, third_cross],
                                     axis=1,
                                     name="combined_features")  # (N, (T+2), C)

            # TODO: other options of condensing information
            all_features = tf.layers.conv1d(inputs=all_features,
                                            filters=1,
                                            kernel_size=1,
                                            activation=tf.nn.relu,
                                            use_bias=True)  # (N, (T+2), 1)

            all_features = tf.squeeze(all_features, axis=2)  # (N, (T+2))
            logits = tf.squeeze(tf.layers.dense(inputs=all_features,
                                                units=1,
                                                activation=tf.nn.relu), axis=1)  # N

        self.predict = tf.to_int32(tf.round(tf.sigmoid(logits)), name="predicts")

        with tf.name_scope("Accuracy"):
            if self.label is None:
                half_size = self.batch_size / 2
                pseudo_label = tf.constant([1.0] * half_size + [0.0] * half_size)  # half 1 & half 0
                self.acc, _ = tf.metrics.accuracy(labels=pseudo_label, predictions=self.predict)
            else:
                self.acc, _ = tf.metrics.accuracy(labels=self.label, predictions=self.predict)
            tf.summary.scalar("Accuracy", self.acc)

        # ===== ADD OTHER METRICS HERE =====

        """
        if label is set: cross entropy loss
        else: upper half positive and lower half negative, use subtract
        """
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        if self.label is None:
            pos_label, neg_label = tf.split(2, tf.sigmoid(logits), axis=0)
            self.loss = tf.add(
                tf.reduce_sum(
                    tf.subtract(neg_label, pos_label, name="training_loss_nolabel")),
                regularization_loss,
                name="training_loss_nolabel_reg"
            )
            self.mean_loss = tf.divide(self.loss,
                                       tf.to_float(self.batch_size/2),
                                       name="mean_loss_nolabel_reg")
        else:
            self.loss = tf.add(
                tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.label,
                        logits=logits,
                        name="training_loss_label")),
                regularization_loss,
                name="training_loss_label_reg"
            )

            self.mean_loss = tf.divide(self.loss,
                                       tf.to_float(self.batch_size),
                                       name="mean_loss_label_reg")

            tf.summary.scalar("Mean_Loss", self.mean_loss)

            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
            self.merged = tf.summary.merge_all()  # TODO: other metrics outside model

