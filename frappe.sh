#!/bin/bash

# embedding size=128
CUDA_VISIBLE_DEVICES=1,2 python3 interprecsys/main.py \
    --trial_id 102 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 8

# default settings
CUDA_VISIBLE_DEVICES=1,2 python3 interprecsys/main.py \
    --trial_id 101 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 4


# embedding size=256
CUDA_VISIBLE_DEVICES=1,2 python3 interprecsys/main.py \
    --trial_id 103 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 16

# embedding size=512
CUDA_VISIBLE_DEVICES=1,2 python3 interprecsys/main.py \
    --trial_id 104 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 32
