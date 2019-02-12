#!/bin/bash

# default setting
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 001 \
    --epoch 20 \
    --batch_size 256 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 20000 \
    --scale_embedding=False \
    --regularization_weight 0.01 \
    --embedding_size 128

# embedding size=64
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 002 \
    --epoch 20 \
    --batch_size 256 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 20000 \
    --scale_embedding=False \
    --regularization_weight 0.01 \
    --embedding_size 64

# embedding size=256
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 003 \
    --epoch 20 \
    --batch_size 256 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 20000 \
    --scale_embedding=False \
    --regularization_weight 0.01 \
    --embedding_size 256

# embedding size=512
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 004 \
    --epoch 20 \
    --batch_size 256 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 20000 \
    --scale_embedding=False \
    --regularization_weight 0.01 \
    --embedding_size 512
