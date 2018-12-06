"""
Factorization Machine Model

@Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import tensorflow as tf


class FMModel():
    """Factorization Machine Model
    This class implements L2-regularized Second-order FM model

    TODO:
        - [ ] Add support to tensorboard
    """

    def __init__(self,
                 lr,
                 lambda_1,
                 lambda_2,
                 num_latent_factor=None,
                 num_features=None,

                 ):
        self.k = num_latent_factor
        self.p = num_features
        self.seed = None # TODO: what is this?

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        self.w0, self.W, self.V = None, None, None
        self.X, self.y = None, None

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.init_all_vars = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.trainer = None
        self.target = None

        self.graph = None

    # ======== Generate Graph =======
    def reg_learnable_params(self):
        with tf.name_scope("learnable_params") as scope:
            self.w0 = tf.Variable(tf.zeros(shape=[1]), dtype=tf.float32, name="bias_term")
            self.W = tf.Variable(tf.zeros(shape=[self.p]),
                                 dtype=tf.float32, name="first_order_term")
            self.V = tf.Variable(tf.random_normal([self.k, self.p], stddev=0.01))

    def reg_placeholder(self):
        with tf.name_scope("input_output") as scope:
            self.X = tf.placeholder(dtype=tf.float32, shape=[None. self.p], name="design_matrix")
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="target_vector")

    def reg_main_flow(self):
        with tf.name_scope("linear_term") as scope:
            linear_terms = tf.add(self.w0,
                                  tf.reduce_sum(
                                      tf.multiply(self.W, self.X),
                                      axis=1, keep_dims=True))
        with tf.name_scope("second_order_term") as scope:
            pair_interact = tf.multiply(0.5,
                                        tf.reduce_sum(
                                            tf.subtract(
                                                tf.pow(
                                                    tf.matmul(self.X, tf.transpose(self.V)), 2),
                                                tf.matmul(
                                                    tf.pow(self.X, 2),
                                                    tf.transpose(tf.pow(self.V, 2)))),
                                            axis=1, keep_dims=True))
        with tf.name_scope("l2_regularization") as scope:
            y_hat = tf.add(linear_terms, pair_interact)
            mean_square_loss = tf.losses.mean_squared_error(labels=self.y, predictions=y_hat)
            regularization_1 = tf.nn.l2_loss(self.W, name="L2_regularization_linear_term")
            regularization_2 = tf.nn.l2_loss(self.V, name="L2_regularization_pair_term")
            self.target = tf.add_n([mean_square_loss, regularization_1, regularization_2])
            self.target = tf.verify_tensor_all_finite(self.target,
                                                      msg="Nan or Inf in target!",
                                                      name="target")

    def build_graph(self):
        """Build computational graph"""
        # Create computational graph
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        with self.graph.as_default():
            self.reg_learnable_params()  # Register learnable parameters
            self.reg_placeholder()  # Register placeholders: X, y
            self.reg_main_flow()  # Register main TensorFlow

            self.trainer = self.optimizer.minimize(self.target)

    # ======= Model Training =======

