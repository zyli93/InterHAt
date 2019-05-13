#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 611 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0001 \
    --embedding_size 8

# embedding size=128
CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 612 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8

# default settings
CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 613 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8


CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 614 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.001 \
    --embedding_size 8

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 615 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.002 \
    --embedding_size 8

# embedding size=256
CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 616 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.005 \
    --embedding_size 8

# embedding size=512
CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 617 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.01 \
    --embedding_size 8
