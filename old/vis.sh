#!/bin/bash

python interprecsys.cross/main.py \
    --trial_id 8606\
    --epoch 10 \
    --batch_size 256 \
    --dataset "vis" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.002 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --num_head 4 \
    --attention_size 16

# 8601: 3 orders
# 8602: 5 orders
# 8603: 3 orders