#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python3 interhat/main.py \
   --trial_id 1 \
   --epoch 10 \
   --batch_size 2048 \
   --dataset "criteoDAC" \
   --num_iter_per_save 50000 \
   --scale_embedding=False \
   --regularization_weight 0.0002 \
   --embedding_size 12 \
   --dropout_rate 0 \
   --num_head 12 \
   --attention_size 30
