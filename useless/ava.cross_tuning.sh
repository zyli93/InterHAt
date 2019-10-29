#!/bin/bash

# default settings
CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.cross/main.py \
    --trial_id 2001 \
    --epoch 10 \
    --batch_size 2048 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 8 \
    --dropout_rate 0 \
    --num_head 8 \
    --attention_size 20 

# CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.cross/main.py \
#     --trial_id 2002 \
#     --epoch 20 \
#     --batch_size 2048 \
#     --dataset "avazu" \
#     --use_graph=False \
#     --num_iter_per_save 10000 \
#     --scale_embedding=False \
#     --regularization_weight 0.0005 \
#     --embedding_size 8 \
#     --dropout_rate 0 \
#     --learning_rate 0.002
# 
# CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.cross/main.py \
#     --trial_id 723 \
#     --epoch 20 \
#     --batch_size 2048 \
#     --dataset "avazu" \
#     --use_graph=False \
#     --num_iter_per_save 10000 \
#     --scale_embedding=False \
#     --regularization_weight 0.0005 \
#     --embedding_size 8 \
#     --dropout_rate 0 \
#     --learning_rate 0.005
# 
# 
# CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.cross/main.py \
#     --trial_id 724 \
#     --epoch 20 \
#     --batch_size 2048 \
#     --dataset "avazu" \
#     --use_graph=False \
#     --num_iter_per_save 10000 \
#     --scale_embedding=False \
#     --regularization_weight 0.0005 \
#     --embedding_size 8 \
#     --dropout_rate 0.05 \
#     --learning_rate 0.001
# 
# 
# CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.cross/main.py \
#     --trial_id 725 \
#     --epoch 20 \
#     --batch_size 2048 \
#     --dataset "avazu" \
#     --use_graph=False \
#     --num_iter_per_save 10000 \
#     --scale_embedding=False \
#     --regularization_weight 0.0005 \
#     --embedding_size 8 \
#     --dropout_rate 0.05 \
#     --learning_rate 0.002
# 
# 
# CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.cross/main.py \
#     --trial_id 726 \
#     --epoch 20 \
#     --batch_size 2048 \
#     --dataset "avazu" \
#     --use_graph=False \
#     --num_iter_per_save 10000 \
#     --scale_embedding=False \
#     --regularization_weight 0.0005 \
#     --embedding_size 8 \
#     --dropout_rate 0.1 \
#     --learning_rate 0.001
# 
# 
# CUDA_VISIBLE_DEVICES=4,5 python3 interprecsys.cross/main.py \
#     --trial_id 727 \
#     --epoch 20 \
#     --batch_size 2048 \
#     --dataset "avazu" \
#     --use_graph=False \
#     --num_iter_per_save 10000 \
#     --scale_embedding=False \
#     --regularization_weight 0.0005 \
#     --embedding_size 8 \
#     --dropout_rate 0.1 \
#     --learning_rate 0.002
