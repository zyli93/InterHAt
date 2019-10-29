# This file contains all the updates since IJCAI submissions.

1. Removed the save model section. Apparently, we don't need to do so.

2. Removed run dev set code:
```python
            # # run validation set
            # epoch_msg, attn1, attn2, attn3, attnk \
            #     = run_evaluation(sess=sess,
            #                      data_loader=data_loader,
            #                      epoch=epoch,
            #                      model=model)

            # attn = [attn1, attn2, attn3, attnk]
            # print(epoch_msg)
            # print(epoch_msg, file=performance_writer)
            # if epoch > 5:
            #     for x, ar in enumerate(attn):
            #         np.savetxt("./performance/vis/{}.val.{}.{}.csv".format(
            #             FLAGS.trial_id, epoch, x+1),
            #             ar, delimiter=",", fmt="%f")
```

3. Removed tensorflow recording
```python
    train_writer.add_summary(
        merged_summary,
        global_step=sess.run(model.global_step)
    )
    
    train_writer = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)
```

4. Removed Version 1 comments
```python
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
```

5. Removed `key mask` from
```python
    key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

    paddings = tf.ones_like(outputs)*(-2**32+1)
    outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
```
and `query mask` from transformer
```python
    query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    outputs *= query_masks  # broadcasting. (N, T_q, C)
```

6. Removed the long long cross feature
```python
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
```


# Action Items:
1. Figure out Dropout and BatchNormalization, see what is what. 
Try replace Dropout with BatchNormalization.

2. Remove all `scale_embedding commands`

3. Remove all figure out what the masks are

4. Play with the TODO (num_unit/emb_size) in modules
