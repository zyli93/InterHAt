#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1
python interprecsys/main.py \
    --trial_id $2 \
    --epoch 20 \
    --batch_size 256 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 2000 \
    --scale_embedding=False \
    --regularization_weight 0.0 \
    --embedding_size 128
