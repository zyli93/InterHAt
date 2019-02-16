#!/bin/bash

# default setting
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 421 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 8 \
    --dropout_rate 0.1
   

# embedding size=64
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 422 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 8 \
    --dropout_rate 0.2

# embedding size=256
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 423 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 8 \
    --dropout_rate 0.3

# embedding size=512
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 424 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 8 \
    --dropout_rate 0.4

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 425 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 8 \
    --dropout_rate 0.5

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 426 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 8 \
    --dropout_rate 0.6

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 427 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 8 \
    --dropout_rate 0.7

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 428 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 8 \
    --dropout_rate 0.8

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 429 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 8 \
    --dropout_rate 0.9
