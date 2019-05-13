#!/bin/bash

# embedding size=128
python interprecsys.cross/main.py \
    --trial_id 9428 \
    --epoch 20 \
    --batch_size 256 \
    --dataset "frappe" \
    --use_graph=False \
    --num_iter_per_save 50000 \
    --scale_embedding=False \
    --regularization_weight 0.002 \
    --embedding_size 12 \
    --dropout_rate 0 \
    --num_head 4 \
    --attention_size 16

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