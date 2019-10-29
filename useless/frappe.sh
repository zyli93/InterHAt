#!/bin/bash

# embedding size=128
python interhat.new/main.py \
    --trial_id 10933 \
    --epoch 10 \
    --batch_size 128 \
    --dataset "frappe" \
    --num_iter_per_save 50000 \
    --learning_rate 0.001 \
    --regularization_weight 0.1 \
    --embedding_size 12 \
    --dropout_rate 0 \
    --highest_order 3 \
    --temperature 0.5 \
    --num_head 4 \
    --num_block 1 \
    --attenion_size 8 \
    --attention_size_last 12 \
    --pred_layers "32,2"

#python interprecsys/main.py \
#    --trial_id 8812 \
#    --epoch 20 \
#    --batch_size 256 \
#    --dataset "frappe" \
#    --use_graph=False \
#    --num_iter_per_save 50000 \
#    --scale_embedding=False \
#    --regularization_weight 0.002 \
#    --embedding_size 8 \
#    --dropout_rate 0 \
#    --num_head 2 \
#    --attention_size 16
#
#python interprecsys/main.py \
#    --trial_id 8813 \
#    --epoch 20 \
#    --batch_size 256 \
#    --dataset "frappe" \
#    --use_graph=False \
#    --num_iter_per_save 50000 \
#    --scale_embedding=False \
#    --regularization_weight 0.002 \
#    --embedding_size 8 \
#    --dropout_rate 0 \
#    --num_head 4 \
#    --attention_size 16
#
#python interprecsys/main.py \
#    --trial_id 8814 \
#    --epoch 20 \
#    --batch_size 256 \
#    --dataset "frappe" \
#    --use_graph=False \
#    --num_iter_per_save 50000 \
#    --scale_embedding=False \
#    --regularization_weight 0.002 \
#    --embedding_size 8 \
#    --dropout_rate 0 \
#    --num_head 8 \
#    --attention_size 16
#
#python interprecsys/main.py \
#    --trial_id 8815 \
#    --epoch 20 \
#    --batch_size 256 \
#    --dataset "frappe" \
#    --use_graph=False \
#    --num_iter_per_save 50000 \
#    --scale_embedding=False \
#    --regularization_weight 0.002 \
#    --embedding_size 8 \
#    --dropout_rate 0 \
#    --num_head 12 \
#    --attention_size 16

#python interprecsys/main.py \
#    --trial_id 8806 \
#    --epoch 20 \
#    --batch_size 256 \
#    --dataset "frappe" \
#    --use_graph=False \
#    --num_iter_per_save 50000 \
#    --scale_embedding=False \
#    --regularization_weight 0.002 \
#    --embedding_size 8 \
#    --dropout_rate 0 \
#    --num_head 16 \
#    --attention_size 16
