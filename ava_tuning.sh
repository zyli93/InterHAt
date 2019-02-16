#!/bin/bash

# default settings
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 721 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --learning_rate 0.001

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 722 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --learning_rate 0.002

CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 723 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --learning_rate 0.005


CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 724 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0.05 \
    --learning_rate 0.001


CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 725 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0.05 \
    --learning_rate 0.002


CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 726 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0.1 \
    --learning_rate 0.001


CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys/main.py \
    --trial_id 727 \
    --epoch 20 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0.1 \
    --learning_rate 0.002
