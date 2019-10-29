#!/bin/bash

# default settings
#CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
#    --trial_id 1001 \
#    --epoch 10 \
#    --batch_size 2048 \
#    --dataset "criteoDAC" \
#    --use_graph=False \
#    --num_iter_per_save 10000 \
#    --scale_embedding=False \
#    --regularization_weight 0.0005 \
#    --embedding_size 8 \
#    --dropout_rate 0

# CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
#     --trial_id 1002 \
#     --epoch 10 \
#     --batch_size 2048 \
#     --dataset "criteoDAC" \
#     --use_graph=False \
#     --num_iter_per_save 50000 \
#     --scale_embedding=False \
#     --regularization_weight 0.0002 \
#     --embedding_size 8 \
#     --dropout_rate 0 
# # 
CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 1003 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8 \
    --dropout_rate 0

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 1004 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 16 \
    --dropout_rate 0

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 1005 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0.05

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 1006 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8  \
    --dropout_rate 0.1

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 1007 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --num_head 6

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 1008 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8 \
    --dropout_rate 0.05 \
    --num_head 6

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 1009 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0.1 \
    --num_head 6

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 1010 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --num_head 10


CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 1011 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8 \
    --dropout_rate 0.05 \
    --num_head 10

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 1012 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.0005 \
    --embedding_size 8 \
    --dropout_rate 0.1 \
    --num_head 10

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 1013 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --num_head 8 \
    --attention_size 64
