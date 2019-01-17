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
        """

        # ===== Get embeddings for input features =====

        with tf.name_scope("input_to_embeddings") as scope:
            for feat_name, feat_tensor in self.input_feature_placeholders.items():
                self.feat_embedding_tensors[feat_name] = tf.nn.embedding_lookup(self.feat_hid_emb_lookup, feat_tensor)

        # ===== Apply GraphSAGE AGG/CONCAT/NONLINEAR =====


# ===== InterpRecSys Base Model =====
class InterprecsysBase:

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
                 , merge_feat_channel
                 , random_seed=Constant.RANDOM_SEED
                 , scale_embedding=False
                 ):
        # config parameters
        self.embedding_dim = embedding_dim  # the C
        self.scale_embedding = scale_embedding  # bool
        self.field_size = field_size
        self.feat_size = feature_size  # the T

        self.dropout_rate = dropout_rate
        self.random_seed = random_seed
        self.num_block = num_block  # num of blocks of multi-head attn
        self.num_head = num_head  # num of heads
        self.regularization_weight = regularization_weight
        self.merge_feat_channel = merge_feat_channel

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

        # merging intermediate results
        self.all_features, self.weight_all_feat, self.weighted_sum_all_feature = None, None, None
        self.logits = None

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
        with tf.name_scope("Multilayer_attn"):
            for i_block in range(self.num_block):
                with tf.variable_scope("attn_head_{}".format(str(i_block))) as scope:
                    # Multihead Attention
                    features = multihead_attention(queries=features,
                                                   keys=features,
                                                   num_units=self.embedding_dim,
                                                   num_heads=self.num_head,
                                                   dropout_rate=self.dropout_rate,
                                                   causality=False,
                                                   is_training=self.is_training,
                                                   scope="multihead_attn")

                    # Feed Forward
                    features = feedforward(inputs=features,
                                           num_units=[4 * self.embedding_dim, self.embedding_dim],
                                           scope="feed_forward")  # (N, T, C)

        # build second order cross
        # TODO: could be other functions
        # TODO: some of var names are `over-scoped`.
            # with tf.name_scope("sec_order_cross"):

        with tf.name_scope("Third_order") as scope:
            second_cross = tf.layers.conv1d(inputs=tf.transpose(features, [0, 2, 1]),  # transpose: (N, C, T)
                                            filters=1,
                                            kernel_size=1,
                                            activation=tf.nn.relu,
                                            use_bias=True)  # (N, C, 1)
            second_cross = tf.transpose(second_cross, [0, 2, 1])  # (N, 1, C)

            third_cross = single_attention(queries=second_cross,
                                           keys=self.emb,
                                           values=self.emb,
                                           scope="single_attn",
                                           regularize=True)  # (N, 1, C)

        with tf.name_scope("Merged_features"):
            # concatenate enc, second_cross, and third_cross
            self.all_features = tf.concat([self.emb, second_cross, third_cross],
                                           axis=1,
                                           name="concat_feature")  # (N, (T+2), C)

            # ===== Generate weights of all features =====
            # Column wise Conv-1D, ReLU, and Softmax (sum up to one)

            # TODO: tune - conv1d's activation function, might be negative

            # +++ Del later +++
            self.linear_act_conv1d = tf.layers.conv1d(
                inputs=self.all_features, filters=1, kernel_size=1, activation=tf.nn.relu, use_bias=True)

            self.weight_all_feat = tf.nn.softmax(self.linear_act_conv1d)

            # +++ Until here +++

            # +++ Uncomment later +++
            # self.weight_all_feat = tf.nn.softmax(
            #     tf.layers.conv1d(inputs=self.all_features,
            #                      filters=1,
            #                      kernel_size=1,
            #                      activation=tf.nn.relu,
            #                      use_bias=True),
            #     name="Weight_of_All_Features")

            """
            logits = tf.squeeze(tf.layers.dense(inputs=all_features,
                                                units=1,
                                                activation=tf.nn.relu), axis=1)  # N
                                                # activation=None), axis=1)
            """

            # ===== Weighted Sum of Features =====
            self.weighted_sum_all_feature = tf.reduce_sum(
                tf.multiply(self.all_features, self.weight_all_feat),  # (N, (T+2), C)
                axis=2,
                name="Weighted_Sum_of_All_Features"
            )  # (N, (T+2))

            # ===== Dense layers: merging from T+2 to 1 =====
            # ** Output Domain: [-inf, +inf] **

            # TODO: tune - dense's activation function

            self.logits = tf.squeeze(tf.layers.dense(
                inputs=self.weighted_sum_all_feature,
                units=1,
                activation=None,
                use_bias=True,
                name="Logits"), axis=1
            )  # (N)

        self.logits_sigmoid = tf.nn.sigmoid(self.logits)

        # ===== Accuracy =====
        with tf.name_scope("Accuracy"):
            self.predict = tf.to_int32(tf.round(self.logits), name="predicts")
            self.acc, _ = tf.metrics.accuracy(labels=self.label, predictions=self.predict)
            tf.summary.scalar("Accuracy", self.acc)

        # ===== ADD OTHER METRICS HERE =====

        """
        if label is set: cross entropy loss
        else: upper half positive and lower half negative, use subtract
        """
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.reg_term = regularization_loss

        with tf.name_scope("Mean_loss"):
            self.loss = tf.add(
                tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.label,
                        logits=self.logits,
                        name="training_loss_label")),
                self.regularization_weight * regularization_loss,
                name="training_loss_label_reg"
            )

            self.mean_loss = tf.divide(self.loss,
                                       tf.to_float(self.batch_size),
                                       name="mean_loss_label_reg")

            tf.summary.scalar("Mean_Loss", self.mean_loss)

        self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
        self.merged = tf.summary.merge_all()
