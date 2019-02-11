import tensorflow as tf


def recur_attention(queries,
                    keys,
                    values,
                    scope,
                    reuse=None,
                    regularize_scale=None,
                    ):
    """Single attention

    :param queries: 2-D Tensor, shape=[N, C], query vector
    :param keys: 3-D Tensor, shape=[N, T, C], key tensor
    :param values: 3-D Tensor, shape=[N, T, C], value tensor
    :param scope:
    :param reuse:
    :param regularize: Boolean. Do regularization or not.
    :param regularize_scale: float
    :return:
    """
    _, T, C = keys.get_shape().as_list()

    if regularize_scale:
        regularizer = tf.contrib.layers.l2_regularizer(scale=regularize_scale)
    else:
        regularizer = None

    initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope(scope, reuse=reuse):
        # W: T * C
        W = tf.get_variable(name="prev_order_cross",
                            dtype=tf.float32,
                            shape=(T * C),
                            initializer=initializer,
                            regularizer=regularizer)

        # b: C
        b = tf.get_variable(name="single_attn_bias",
                            dtype=tf.float32,
                            shape=(C),
                            initializer=initializer,
                            regularizer=regularizer)

        # vector to make it scalar
        # h: C
        h = tf.get_variable(name="single_attn_h",
                            dtype=tf.float32,
                            shape=(C),
                            initializer=initializer,
                            regularizer=regularizer)

        """
        Math equation of a_i
            a_i = h^T . RELU(W * outer(Q, K[i]) + b)
        """

        # outer(Q, K[i])

        kq_outer = tf.reshape(
            tf.multiply(keys,  # [N, T, C]
                        tf.expand_dims(queries, axis=1)  # [N, 1, C]
                        ),  # [N, T, C]
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


def agg_attention(query,
                  keys,
                  values,
                  attention_size,
                  regularize_scale=None):
    """

    :param query: [a_s]
    :param keys: [N, T, dim]
    :param values: [N, T, dim]
    :param attention_size: [attn_size]
    :param regularize_scale:
    :return:
    """
    if regularize_scale:
        regularizer = tf.contrib.layers.l2_regularizer(scale=regularize_scale)
    else:
        regularizer=None

    # project keys to attention space
    projected_keys = tf.layers.dense(keys, attention_size,
                                     activation=tf.nn.relu,
                                     kernel_regularizer=regularizer,
                                     bias_regularizer=regularizer)  # [N, T, a_s]

    # reshape query
    query_ = tf.reshape(query, [1, 1, -1])  # [1, 1, a_s]

    # multiply query_, keys (broadcast)
    attention_energy = tf.reduce_sum(
        tf.multiply(projected_keys, query_),  # [N, T, a_s]
        axis=2
    )  # [N, T]

    # generate attention weights
    attentions = tf.nn.softmax(logits=attention_energy,
                               name="attention")  # [N, T]

    results = tf.reduce_sum(
        tf.multiply(values,
                    tf.expand_dims(attentions, axis=1)),  # [N, T, dim]
        axis=1
    )  # [N, dim]

    return results, attentions













