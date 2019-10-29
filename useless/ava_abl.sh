#!/bin/bash

# default settings
CUDA_VISIBLE_DEVICES=0,1 python3 interprecsys.cross/main.py \
    --trial_id 7021 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --num_head 1 \
    --attention_size 30

CUDA_VISIBLE_DEVICES=0,1 python3 interprecsys.cross/main.py \
    --trial_id 7022 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --num_head 2 \
    --attention_size 30

CUDA_VISIBLE_DEVICES=0,1 python3 interprecsys.cross/main.py \
    --trial_id 7023 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --num_head 4 \
    --attention_size 30


CUDA_VISIBLE_DEVICES=0,1 python3 interprecsys.cross/main.py \
    --trial_id 7024 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --num_head 8 \
    --attention_size 30

CUDA_VISIBLE_DEVICES=0,1 python3 interprecsys.cross/main.py \
    --trial_id 7025 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --num_head 12 \
    --attention_size 30

CUDA_VISIBLE_DEVICES=0,1 python3 interprecsys.cross/main.py \
    --trial_id 7026 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --num_head 16 \
    --attention_size 30
