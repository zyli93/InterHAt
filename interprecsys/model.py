"""Hybrid model of Attentive FM and Graph Attention Network

This is a complex hybrid model of "Attentive Factorization Machine and Graph Attention Network",
    "Graph Attention Network", and "Graph Convolutional Network".

@author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import tensorflow as tf

from const import Constant
from modules import multihead_attention, feedforward, embedding
from module2 import recur_attention, agg_attention


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
        Register all placeholders/trainable_params ne],

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
                 , pool_filter_size
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
        self.pool_filter_size = pool_filter_size

        # training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # ===== Create None variables for object =====
        # variables [None]
        self.embedding_lookup = None
        self.emb = None  # raw features
        self.attn = tf.get_variable(
            name="attention_factors",
            shape=self.pool_filter_size,
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32)


        # placeholders
        self.X_ind, self.X_val, self.label = None, None, None
        self.is_training = None

        # ports to the outside
        self.sigmoid_logits = None
        self.regularization_loss = None
        self.logloss, self.mean_logloss = None, None
        self.overall_loss = None


        # train/summary operations
        self.train_op, self.merged = None, None

        # intermediate results
        self.feature_weights = None
        self.sigmoid_logits = None

        # global training steps
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # operations
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=0.9,
                                                beta2=0.98,
                                                epsilon=1e-8)

        self.build_graph()

    def build_graph(self):

        # Define input
        with tf.name_scope("input_ph"):
            self.X_ind = tf.placeholder(dtype=tf.int32, 
                                        shape=[None, self.field_size], 
                                        name="X_index")
            self.X_val = tf.placeholder(dtype=tf.float32, 
                                        shape=[None, self.field_size], 
                                        name="X_value")
            self.label = tf.placeholder(dtype=tf.float32, 
                                        shape=[None],
                                        name="label")
            self.is_training = tf.placeholder(dtype=tf.bool, 
                                              shape=(), 
                                              name="is_training")

        # lookup and process embedding
        with tf.name_scope("embedding"):
            self.emb = embedding(inputs=self.X_ind,
                                 vocab_size=self.feat_size,
                                 num_units=self.embedding_dim,
                                 scale=self.scale_embedding,
                                 scope="embedding_process")

            # multiply embeddings with values (for numeric features)
            self.emb = tf.multiply(self.emb,
                                   tf.expand_dims(self.X_val, axis=2))

        # self.emb: raw embedding, features: used for later
        features = self.emb

        # apply dropout on embedding
        with tf.name_scope("dropout"):
            features = tf.layers.dropout(features,
                                         rate=self.dropout_rate,
                                         training=self.is_training)

        # multi-layer, multi-head attention, Version 1
        # with tf.name_scope("Multilayer_attn"):
        #     for i_block in range(self.num_block):
        #         with tf.variable_scope("attn_head_{}".format(str(i_block))) as scope:
        #             # multihead attention
        #             features, _ = multihead_attention(queries=features,
        #                                            keys=features,
        #                                            num_units=self.embedding_dim,
        #                                            num_heads=self.num_head,
        #                                            dropout_rate=self.dropout_rate,
        #                                            is_training=self.is_training,
        #                                            scope="multihead_attn")
        #
        #             # feed forward
        #             features = feedforward(inputs=features,
        #                                    num_units=[4 * self.embedding_dim,
        #                                               self.embedding_dim],
        #                                    scope="feed_forward")  # (N, T, C)
        # End of version 1

        # multi-layer, multi-head attention Version 2
        # major difference: remove multi-block
        with tf.name_scope("Multilayer_attn"):
            with tf.variable_scope("attention_head") as scope:
                features, _ = multihead_attention(
                    queries=features,
                    keys=features,
                    num_units=self.embedding_dim,
                    num_heads=self.num_head,
                    dropout_rate=self.dropout_rate,
                    is_training=self.is_training,
                    scope="multihead_attention"
                )

                features = feedforward(
                    inputs=features,
                    num_units=[4 * self.embedding_dim,
                               self.embedding_dim],
                    scope="feed_forward"
                )

        # build second order cross
        with tf.name_scope("Second_order") as scope:
            # second_cross = tf.layers.conv1d(inputs=tf.transpose(
            #     features, [0, 2, 1]),
            #     # transpose: (N, C, T)
            #     filters=1,
            #     kernel_size=1,
            #     activation=tf.nn.relu,
            #     use_bias=True)  # (N, C, 1)

            second_cross_context = tf.get_variable(
                name="second_cross_attn_context",
                shape=(self.embedding_dim),
                dtype=tf.float32
            )

            # generate second cross features
            # the _ is heat (attentions)
            # TODO: remove regularization weight in the end
            second_cross, _ = agg_attention(
                query=second_cross_context,
                keys=features,
                values=features,
                attention_size=self.embedding_dim,
                regularize_scale=self.regularization_weight
            )  # [N, dim]=[N, C]

            print("second_cross dim", second_cross.get_shape())

        # build third order cross
        with tf.name_scope("Third_order") as scope:
            # second_cross = tf.transpose(second_cross, [0, 2, 1])  # (N, 1, C), old
            second_cross = tf.expand_dims(second_cross, axis=1)  # [N, 1, C]

            # old
            third_cross = recur_attention(queries=second_cross,
                                          keys=self.emb,
                                          values=self.emb,
                                          scope="third_order",
                                          regularize_scale=self.regularization_weight
                                          )  # (N, 1, C)

        with tf.name_scope("Merged_features"):

            # concatenate [enc, second_cross, third_cross]
            all_features = tf.concat(
                [self.emb, second_cross, third_cross],
                axis=1, name="concat_feature")  # (N, (T+2), C)

            # # Version 2
            # ===== Generate weights of all features =====

            # column wise Conv-1D, ReLU, and Softmax (sum up to one)
            # TODO: can be relu+softmax, or direct softmax
            # self.feature_weights = tf.nn.softmax(
            #     tf.layers.conv1d(inputs=all_features,
            #                      filters=1,
            #                      kernel_size=1,
            #                      activation=tf.nn.relu,
            #                      use_bias=True),  # (N, (T+2), 1)
            #     name="Feature_attentive_weights"
            # )  # (N, (T+2), 1)

            # # condense features with pooling - feature abstracts
            # condense_feature = tf.reduce_max(
            #     tf.layers.conv1d(inputs=all_features,
            #                      filters=self.pool_filter_size,
            #                      kernel_size=1,
            #                      activation=tf.nn.relu,
            #                      use_bias=True),  # (N, (T+2), pf_size)
            #     axis=2,
            #     name="Feature_abstracts"
            # )  # (N, (T+2), 1)

            # # predictions in terms of logits
            # logits = tf.reduce_sum(
            #     tf.multiply(
            #         tf.squeeze(self.feature_weights),
            #         condense_feature
            #     ),  # (N, T+2)
            #     axis=1,
            #     name="Logits"
            # )  # (N)

            # Version 1
            # ===== Weighted Sum of Features =====
             
            # weighted_sum_all_feature = tf.reduce_sum(
            #     tf.multiply(all_features,
            #                self.feature_weights),  # (N, (T+2), C)
            #     axis=2,
            #     name="Weighted_Sum_of_All_Features"
            # )  # (N, (T+2))
            # 
            #  # ===== Dense layers: merging from T+2 to 1 =====
            # 
            #  # TODO: tune - dense's activation function
            # 
            # logits = tf.squeeze(
            #         tf.layers.dense(
            #             inputs=weighted_sum_all_feature,
            #             units=1,
            #             activation=None,
            #             use_bias=True,
            #             name="Logits"),
            #         axis=1)  # (N)

        # Version 3
        # map C to pool_filter_size dimension
        mapped_all_feature = tf.layers.conv1d(
            inputs=all_features,
            filters=self.pool_filter_size,
            kernel_size=1,
            use_bias=True,
            name="Mapped_all_feature"
        )  # (N, T+2, pf_size)
        
        # apply context vector
        feature_weights = tf.nn.softmax(
            tf.squeeze(
                tf.layers.dense(
                    mapped_all_feature,
                    units=1,
                    activation=None,
                    use_bias=False
                ),  # (N, T+2, 1),
                [2]
            ), # (N, T+2)
        )  # (N, T+2)
        
        # weighted sum
        weighted_sum_feat = tf.reduce_sum(
            tf.multiply(
                all_features,
                tf.expand_dims(feature_weights, axis=2),
            ),  # (N, T+2, C)
            axis=[1],
            name="Attn_weighted_sum_feature"
        )  # (N, C)
        
        # last non-linear
        hidden_logits = tf.layers.dense(
            weighted_sum_feat,
            units=self.embedding_dim // 2,
            activation=tf.nn.relu,
            use_bias=False,
            name="HiddenLogits"
        )  # (N, C/2)

        # the last dense for logits
        logits = tf.squeeze(
            tf.layers.dense(
                hidden_logits,
                units=1,
                activation=None,
                use_bias=False,
                name="Logits"
            ),  # (N, 1)
            axis=[1]
        )  # (N,)

        # Generate Logits here

        # sigmoid logits
        self.sigmoid_logits = tf.nn.sigmoid(logits)

        # regularization term
        # self.regularization_loss = tf.reduce_sum(
        #         tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) \
        #         * self.regularization_weight
        self.regularization_loss = tf.losses.get_regularization_loss()

        # sum logloss
        print("label shape", self.label.get_shape())
        print("logit shape", logits.shape)
        self.logloss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.expand_dims(self.label, -1),
                logits=tf.expand_dims(logits, -1),
                name="SumLogLoss"))

        # mean logloss [old]
        self.mean_logloss = tf.divide(self.logloss,
                                      tf.to_float(self.batch_size),
                                      name="MeanLogLoss")

        # mean logloss
        # self.mean_logloss = tf.reduce_mean(self.logloss,
        #                                    name="MeanLogLoss")

        # overall loss
        self.overall_loss = tf.add(
            self.mean_logloss,
            self.regularization_loss,
            name="OverallLoss"
        )
        
        # mean loss
        # with tf.name_scope("Mean_loss"):
        #     self.loss = tf.add(
        #         tf.reduce_sum(
        #             tf.nn.sigmoid_cross_entropy_with_logits(
        #                 labels=self.label,
        #                 logits=logits,
        #                 name="Cross_Entropy_Loss")),
        #         self.regularization_loss,
        #         name="Overall_Loss"
        #     )

        tf.summary.scalar("Mean_LogLoss", self.mean_logloss)
        tf.summary.scalar("Reg_Loss", self.regularization_loss)
        tf.summary.scalar("Overall_Loss", self.overall_loss)

        self.train_op = self.optimizer.minimize(self.overall_loss, 
                                                global_step=self.global_step)
        self.merged = tf.summary.merge_all()
