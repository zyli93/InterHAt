#!/bin/bash

# default settings
CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 711 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 712 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0.05

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 713 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0.1

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 714 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 16 \
    --dropout_rate 0

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 715 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 16 \
    --dropout_rate 0.05

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 716 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 16  \
    --dropout_rate 0.1

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 717 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --num_head 16

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys/main.py \
    --trial_id 718 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0.1 \
    --num_head 16
