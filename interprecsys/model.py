"""Hybrid model of Attentive FM and Graph Attention Network

This is a complex hybrid model of "Attentive Factorization Machine and Graph Attention Network",
    "Graph Attention Network", and "Graph Convolutional Network".

@author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import tensorflow as tf

class Interprecsys:
    def __init__(self
                 , hidden_dimension
                 , attr_dimension):
        # placeholders to initialize
        self.user_emb = None
        self.attr_emb = None

        # vars to initialize


        # settings
        self.dimension = hidden_dimension
        self.attribute_dimension = attr_dimension

    def build_graph(self):
        self._build_att()
        self._build_gcn()

    def _build_att(self):
        pass

    def _build_gcn(self):
        pass

    def param_initializer(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # TODO: check correctness

    def register_placeholder(self):
        # user-item features
        dim = self.dimension
        attr_dim = self.attribute_dimension
        with tf.name_scope("input") as scope:
            self.user_emb = tf.placeholder(dtype=tf.float32, shape=[None, None, dim], name="user_embedding")
            self.attr_emb = tf.placeholder(dtype=tf.float32,
                                           shape=[None, attr_dim, dim], name="attr_embedding")
            # TODO: other placeholders

