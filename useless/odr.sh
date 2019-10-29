#!/bin/bash

# default settings
CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.abla/main.py \
    --trial_id 5019 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 12 \
    --dropout_rate 0 \
    --num_head 12 \
    --attention_size 30

# CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.abla/main.py \
#     --trial_id 5025 \
#     --epoch 5 \
#     --batch_size 2048 \
#     --dataset "avazu" \
#     --use_graph=False \
#     --num_iter_per_save 10000 \
#     --scale_embedding=False \
#     --regularization_weight 0.0002 \
#     --embedding_size 8 \
#     --dropout_rate 0 \
#     --num_head 8 \
#     --attention_size 20
