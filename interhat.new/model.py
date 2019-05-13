"""@author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import tensorflow as tf

from const import Constant
from modules import multihead_attention, feedforward, embedding
from module2 import agg_attention


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
                 , attention_size
                 , pool_filter_size
                 , temperature
                 , dropout_rate
                 , regularization_weight
                 , highest_order
                 , pred_mlp_layers
                 , random_seed=Constant.RANDOM_SEED
                 ):

        # =========================
        #     config parameters
        # =========================

        self.embedding_dim = embedding_dim  # the C
        self.field_size = field_size
        self.feat_size = feature_size  # the T

        self.dropout_rate = dropout_rate
        self.random_seed = random_seed
        self.num_block = num_block  # num of blocks of multi-head attn
        self.num_head = num_head  # num of heads
        self.attention_size = attention_size
        self.attention_size_last = pool_filter_size  # TODO: what is this?
        self.highest_order = highest_order
        self.temperature = temperature

        # training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # ==================================
        #   `None` variables for objects
        # ==================================

        self.emb = None  # raw features

        self.pred_layers = [int(x) for x in pred_mlp_layers.strip().split(",")]
        assert self.pred_layers[-1] == 2, "CTR Pred last layer should be 2!"

        # placeholders
        self.X_ind, self.X_val, self.label = None, None, None
        self.is_training = None

        # pred & losses
        self.reg_loss, self.log_loss, self.loss = None, None, None
        self.pred = None

        # train operator
        self.train_op = None

        # attentions and weights
        self.agg_feature_list = []
        self.attn_weight_list = []
        self.final_attn = None
        self.multi_order_attn = None

        # global training steps
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # regularizer
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=regularization_weight)

        self.build_graph()

    def build_graph(self):

        # ==================
        #   placeholders
        # ==================

        with tf.name_scope("Placeholders"):
            self.X_ind = tf.placeholder(dtype=tf.int32, shape=[None, self.field_size])
            # self.X_val = tf.placeholder(dtype=tf.float32, shape=[None, self.field_size]) [OLD]
            self.label = tf.placeholder(dtype=tf.float32, shape=[None])
            self.is_training = tf.placeholder(dtype=tf.bool, shape=())

        # ==================
        #   raw embedding
        # ==================

        # lookup and process embedding
        with tf.name_scope("Raw_embeddings"):
            self.emb = embedding(inputs=self.X_ind, vocab_size=self.feat_size,
                                 num_units=self.embedding_dim, scope="embedding_process")

            # [OLD] multiply embeddings with values (for ic features)
            # self.emb = tf.multiply(self.emb, tf.expand_dims(self.X_val, axis=2))

        features = self.emb

        # ==================
        #   transformer
        # ==================

        # multi-block multi-head transformer
        with tf.name_scope("Transformer"):
            for i in range(self.num_block):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    features, _ = multihead_attention(
                        queries=features, keys=features, values=features,
                        num_units=self.attention_size, num_heads=self.num_head,
                        dropout_rate=self.dropout_rate, is_training=self.is_training)

                    features = feedforward(
                        inputs=features,
                        num_units=[4 * self.embedding_dim, self.embedding_dim])  # [N, T, dim]

        # =======================
        #   high/cross features
        # =======================

        # cross feature with a for loop
        with tf.name_scope("Attentional_Aggregators"):
            with tf.variable_scope("cross_feature_1", reuse=tf.AUTO_REUSE):
                agg_feature, attn_weight = agg_attention(
                    keys=features, values=features,
                    attention_size=self.attention_size,
                    temperature=self.temperature,
                    regularizer=self.regularizer)
                self.agg_feature_list.append(agg_feature)
                self.attn_weight_list.append(attn_weight)

            for i in range(1, self.highest_order):
                with tf.variable_scope("cross_feature_{}".format(i+1), reuse=tf.AUTO_REUSE):

                    temp_feat = tf.multiply(
                        features,
                        tf.expand_dims(self.agg_feature_list[i-1], axis=1))  # [N, T, dim]

                    agg_feature, attn_weight = agg_attention(
                        keys=temp_feat, values=temp_feat,
                        attention_size=self.attention_size,
                        regularizer=self.regularizer)

                    # RESIDUAL
                    agg_feature += self.agg_feature_list[i-1]

                    self.agg_feature_list.append(agg_feature)
                    self.attn_weight_list.append(attn_weight)

        # concatenate, C: attention_size, k: highest order
        stacked_multi_order_feat = tf.stack(self.agg_feature_list, axis=1)  # (N, k, C)

        # map C to pool_filter_size dimension

        # ========================
        #   last attention layer
        # ========================

        with tf.name_scope("Final_layer_attention"):
            # F -> tanh(MF),
            feature_weights = tf.layers.dense(
                inputs=stacked_multi_order_feat,
                units=self.attention_size_last,
                use_bias=False, activation=tf.nn.tanh)  # (N, k, pf_size)

            # apply context vector, tanh(MF) -> W^Ttanh(MF)
            feature_weights = tf.nn.softmax(
                tf.layers.dense(feature_weights, units=1, use_bias=False),
                axis=1)  # (N, k, 1)

            self.multi_order_attn = feature_weights
            #  This attention is exp(w^T tanh(A.feat)), one of the attentions in MW paper

            attn_multi_order_feats = tf.multiply(stacked_multi_order_feat, feature_weights)  # (N, k, C)

        # ==================
        #   CTR pred MLP
        # ==================

        # ctr_pred_feats: flattened attned_multi_order_feats
        # final output should have two dimension
        with tf.name_scope("ctr_prediction"):
            ctr_pred_feats = tf.contrib.layers.flatten(attn_multi_order_feats)  # (N, k*C)
            for nn in self.pred_layers:
                # use batch_normalization: fc - bn - activation
                ctr_pred_feats = tf.layers.dense(
                    ctr_pred_feats, units=nn, use_bias=False,
                    kernel_regularizer=self.regularizer)
                ctr_pred_feats = tf.layers.batch_normalization(
                    ctr_pred_feats, training=self.is_training)
                ctr_pred_feats = tf.nn.relu(ctr_pred_feats)

        # ==================
        #   pred & loss
        # ==================

        # prediction
        self.pred = tf.argmax(ctr_pred_feats, axis=-1)

        # cross entropy loss, log loss
        self.log_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(self.label, depth=2),
            logits=ctr_pred_feats)

        # regularization loss (weight already multiplied)
        self.reg_loss = tf.losses.get_regularization_loss()

        # total loss
        self.loss = self.log_loss + self.reg_loss

        # ==================
        #   optimization
        # ==================

        # optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                                    .minimize(self.loss, global_step=self.global_step)
