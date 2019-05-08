"""@author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import tensorflow as tf

from const import Constant
from modules import multihead_attention, feedforward, embedding
from module2 import recur_attention, agg_attention


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
                 , dropout_rate
                 , regularization_weight
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
        self.attention_size = attention_size
        self.regularization_weight = regularization_weight
        self.pool_filter_size = pool_filter_size

        # training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # ===== Create None variables for object =====
        # variables [None]
        self.embedding_lookup = None
        self.emb = None  # raw features

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

        # attns
        self.attn_1, self.attn_2, self.attn_3, self.attn_4, self.attn_5 = \
        None, None, None, None, None

        self.attn_k = None

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
        # with tf.name_scope("dropout"):
        #     features = tf.layers.dropout(features,
        #                                  rate=self.dropout_rate,
        #                                  training=self.is_training)

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

        # | multi-layer, multi-head attention Version 2
        # | major difference: remove multi-block

        with tf.name_scope("Multilayer_attn"):
            with tf.variable_scope("attention_head") as scope:
                features, _ = multihead_attention(
                    queries=features,
                    keys=features,
                    num_units=self.attention_size*self.num_head,
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
                )  # [N, T, dim]
                

        # multi-head feature to agg 1st order feature
        with tf.name_scope("Agg_first_order") as scope:
            ctx_order_1 = tf.get_variable(
                name="context_order_1",
                shape=(self.attention_size),
                dtype=tf.float32)

            agg_feat_1, self.attn_1 = agg_attention(
                query=ctx_order_1,
                keys=features,
                values=features,
                attention_size=self.attention_size,
                regularize_scale=self.regularization_weight
                )  # [N, dim]


        # build second order cross
        with tf.name_scope("Second_order") as scope:
            feat_2 = tf.multiply(
                features,
                tf.expand_dims(agg_feat_1, axis=1)
                )  # [N, T, dim]

            feat_2 += features  # Add the residual, [N, T, dim]

            ctx_order_2 = tf.get_variable(
                name="context_order_2",
                shape=(self.attention_size),
                dtype=tf.float32
                )

            agg_feat_2, self.attn_2 = agg_attention(
                query=ctx_order_2,
                keys=feat_2,
                values=feat_2,
                attention_size=self.attention_size,
                regularize_scale=self.regularization_weight
                )

        # build third order cross
        with tf.name_scope("Third_order") as scope:
            feat_3 = tf.multiply(
                features,
                tf.expand_dims(agg_feat_2, axis=1)
                )  # [N, T, dim]

            feat_3 += feat_2  # Add the residual, [N, T, dim]

            ctx_order_3 = tf.get_variable(
                name="context_order_3",
                shape=(self.attention_size),
                dtype=tf.float32
                )

            agg_feat_3, self.attn_3 = agg_attention(
                query=ctx_order_3,
                keys=feat_3,
                values=feat_3,
                attention_size=self.attention_size,
                regularize_scale=self.regularization_weight
                )

        with tf.name_scope("Fourth_order") as scope:
            feat_4 = tf.multiply(
                features,
                tf.expand_dims(agg_feat_3, axis=1)
                )  # [N, T, dim]

            feat_4 += feat_3  # Add the residual, [N, T, dim]

            ctx_order_4 = tf.get_variable(
                name="context_order_4",
                shape=(self.attention_size),
                dtype=tf.float32
                )

            agg_feat_4, self.attn_4 = agg_attention(
                query=ctx_order_4,
                keys=feat_4,
                values=feat_4,
                attention_size=self.attention_size,
                regularize_scale=self.regularization_weight
                )

        with tf.name_scope("Fifth_order") as scope:
            feat_5 = tf.multiply(
                features,
                tf.expand_dims(agg_feat_4, axis=1)
                )  # [N, T, dim]

            feat_4 += feat_3  # Add the residual, [N, T, dim]

            ctx_order_5 = tf.get_variable(
                name="context_order_5",
                shape=(self.attention_size),
                dtype=tf.float32
                )

            agg_feat_5, self.attn_5 = agg_attention(
                query=ctx_order_5,
                keys=feat_5,
                values=feat_5,
                attention_size=self.attention_size,
                regularize_scale=self.regularization_weight
                )


        print("agg 1 shape", agg_feat_1.get_shape().as_list())
        print("agg 2 shape", agg_feat_2.get_shape().as_list())
        print("agg 1 shape", agg_feat_3.get_shape().as_list())
        print("agg 4 shape", agg_feat_4.get_shape().as_list())
        print("agg 5 shape", agg_feat_5.get_shape().as_list())

        with tf.name_scope("Merged_features"):

            # concatenate [enc, second_cross, third_cross]
            # TODO: can + multihead_features
            all_features = tf.stack([
                agg_feat_1,
                agg_feat_2,
                agg_feat_3
                # agg_feat_4,
                # agg_feat_5
                ],
                axis=1, name="concat_feature")  # (N, k, C)

        # Version 3
        # map C to pool_filter_size dimension
        mapped_all_feature = tf.layers.conv1d(
            inputs=all_features,
            filters=self.pool_filter_size,
            kernel_size=1,
            use_bias=True,
            name="Mapped_all_feature"
        )  # (N, k, pf_size)
        
        # apply context vector
        feature_weights = tf.nn.softmax(
            tf.squeeze(
                tf.layers.dense(
                    mapped_all_feature,
                    units=1,
                    activation=None,
                    use_bias=False
                ),  # (N, k, 1),
                [2]
            ), # (N, k)
        )  # (N, k)

        self.attn_k = feature_weights
        
        # weighted sum
        weighted_sum_feat = tf.reduce_sum(
            tf.multiply(
                all_features,
                tf.expand_dims(feature_weights, axis=2),
            ),  # (N, k, C)
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
        self.regularization_loss = tf.losses.get_regularization_loss()

        # sum logloss
        # print("label shape", self.label.get_shape())
        # print("logit shape", logits.shape)
        self.logloss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.expand_dims(self.label, -1),
                logits=tf.expand_dims(logits, -1),
                name="SumLogLoss"))

        self.mean_logloss = tf.divide(
            self.logloss,
            tf.to_float(self.batch_size),
            name="MeanLogLoss"
            )

        # overall loss
        self.overall_loss = tf.add(
            self.mean_logloss,
            self.regularization_loss,
            name="OverallLoss"
        )
        
        tf.summary.scalar("Mean_LogLoss", self.mean_logloss)
        tf.summary.scalar("Reg_Loss", self.regularization_loss)
        tf.summary.scalar("Overall_Loss", self.overall_loss)

        self.train_op = self.optimizer.minimize(self.overall_loss, 
                                                global_step=self.global_step)
        self.merged = tf.summary.merge_all()
