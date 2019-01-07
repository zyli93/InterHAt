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

    TODO: find initializer for W, b, and h.

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
                            shape=(T*C),
                            initializer=initializer,
                            regularizer=regularizer)  # TODO: good initializer

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

        # a_i = h^T RELU(W outer(Q, K[i]) + b)

        # compute attention factor: outer(Q, K[i])
        # _outer = tf.multiply(keys, tf.reshape(queries, shape=[1, 1, C]))  # (N, T, C)
        # _outer = tf.multiply(keys, queries)  # (N, T, C)
        _outer = tf.reshape(tf.multiply(keys, queries), shape=[-1, T * C])  # (N, T * C)

        # RELU(W outer(Q, k[i]) + b)
        # _activ = tf.nn.relu(tf.matmul(_outer, W)) + tf.reshape(b, (1, 1, -1))  # (N, T, C)
        _activ = tf.nn.relu(
            tf.reshape(tf.multiply(_outer, W), [-1, T, C])  # (N, T, C)
            + tf.reshape(b, (1, 1, -1))  # (1, 1, C)
        )   # (N, T, C)

        # h^T RELU(W outer(Q, k[i]) + b)
        h_ = tf.reshape(h, shape=[1, 1, C])
        attention_factor = tf.reduce_sum(tf.multiply(_activ, h_), axis=-1)  # (N, T)

        attention_factor = tf.expand_dims(attention_factor, axis=1)  # (N, T) to (N, 1, T)
        weighted_V = tf.matmul(attention_factor, values)

    return weighted_V







