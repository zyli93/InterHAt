import tensorflow as tf


def single_attention(queries,
                     keys,
                     values,
                     scope,
                     reuse=None,
                     regularize=False,
                     regularize_scale=0.1
                     ):
    """Single attention

    :param queries: 1-D Tensor, shape=[C], query vector
    :param keys: 3-D Tensor, shape=[N, T, C], key tensor
    :param values: 3-D Tensor, shape=[N, T, C], value tensor
    :param scope:
    :param reuse:
    :param regularize: Boolean. Do regularization or not.
    :param regularize_scale: float
    :return:
    """
    _, T, C = keys.get_shape().as_list()

    if regularize:
        regularizer = tf.contrib.layers.l2_regularizer(scale=regularize_scale)
    else:
        regularizer = None

    initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope(scope, reuse=reuse):
        # W: T * C
        W = tf.get_variable(name="single_attn_weight",
                            dtype=tf.float32,
                            shape=(T * C),
                            initializer=initializer,
                            regularizer=regularizer)

        # b: C
        b = tf.get_variable(name="single_attn_bias",
                            dtype=tf.float32,
                            shape=C,
                            initializer=initializer,
                            regularizer=regularizer)

        # vector to make it scalar
        # h: C
        h = tf.get_variable(name="single_attn_h",
                            dtype=tf.float32,
                            shape=C,
                            initializer=initializer,
                            regularizer=regularizer)

        """
        Math equation of a_i
            a_i = h^T . RELU(W * outer(Q, K[i]) + b)
        """

        # outer(Q, K[i])

        kq_outer = tf.reshape(
            tf.multiply(keys, queries),
            shape=[-1, T * C]
        )  # (N, T * C)

        # relu(W * outer(Q, k[i]) + b)

        linear_activation = tf.nn.relu(
            tf.reshape(
                tf.multiply(kq_outer, W),
                shape=[-1, T, C])  # (N, T, C)
            + tf.reshape(
                b,
                shape=[1, 1, -1])  # (1, 1, C)
        )   # (N, T, C)

        # h^T relu (W * outer(Q, k[i]) + b)

        h_ = tf.reshape(h, shape=[1, 1, C])
        attention_factor = tf.reduce_sum(
            tf.multiply(linear_activation, h_),
            axis=-1
        )  # (N, T)

        attention_factor = tf.expand_dims(
            attention_factor,
            axis=1
        )  # (N, T) to (N, 1, T)

        weighted_value = tf.matmul(attention_factor, values)

    return weighted_value

