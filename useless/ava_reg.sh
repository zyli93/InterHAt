#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 621 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0001 \
    --embedding_size 8

# embedding size=128
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 622 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8

# default settings
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 623 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8


CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 624 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 8

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 625 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.002 \
    --embedding_size 8

# embedding size=256
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 626 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.005 \
    --embedding_size 8

# embedding size=512
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 627 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.01 \
    --embedding_size 8
