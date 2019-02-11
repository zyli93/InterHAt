#!/usr/bin/env bash
python3 interprecsys/main.py \
    --trial_id $1 \
    --epoch 20 \
    --batch_size 256 \
    --dataset "criteoDAC" \
    --use_graph=False \
    --num_iter_per_save 5000 \
    --scale_embedding=False \
    --regularization_weight 0.01 \
    --embedding_size 128
