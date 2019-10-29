#!/bin/bash

# default settings
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.aug/main.py \
    --trial_id 811 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.aug/main.py \
    --trial_id 812 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0.05

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.aug/main.py \
    --trial_id 813 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0.1

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.aug/main.py \
    --trial_id 814 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 16 \
    --dropout_rate 0

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.aug/main.py \
    --trial_id 815 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 16 \
    --dropout_rate 0.05

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.aug/main.py \
    --trial_id 816 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 16  \
    --dropout_rate 0.1

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.aug/main.py \
    --trial_id 817 \
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

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.aug/main.py \
    --trial_id 818 \
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
