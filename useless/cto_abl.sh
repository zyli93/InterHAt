#!/bin/bash

# default settings
CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 8011 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 12 \
    --dropout_rate 0 \
    --num_head 1 \
    --attention_size 30

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 8012 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 12 \
    --dropout_rate 0 \
    --num_head 2 \
    --attention_size 30

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 8013 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 12 \
    --dropout_rate 0 \
    --num_head 4 \
    --attention_size 30


CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 8014 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 12 \
    --dropout_rate 0 \
    --num_head 8 \
    --attention_size 30

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 8015 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 12 \
    --dropout_rate 0 \
    --num_head 12 \
    --attention_size 30

CUDA_VISIBLE_DEVICES=2,3 python3 interprecsys.cross/main.py \
    --trial_id 8016 \
    --epoch 5 \
    --batch_size 2048 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 10000 \
    --scale_embedding=False \
    --regularization_weight 0.0002 \
    --embedding_size 12 \
    --dropout_rate 0 \
    --num_head 16 \
    --attention_size 30
