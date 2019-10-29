#!/bin/bash

# default setting
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 101 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 20000 \
    --scale_embedding=False \
    --regularization_weight 0.01 \
    --embedding_size 4

# embedding size=64
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 102 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 20000 \
    --scale_embedding=False \
    --regularization_weight 0.01 \
    --embedding_size 8

# embedding size=256
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 103 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 20000 \
    --scale_embedding=False \
    --regularization_weight 0.01 \
    --embedding_size 16

# embedding size=512
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id n104 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 20000 \
    --scale_embedding=False \
    --regularization_weight 0.01 \
    --embedding_size 32
