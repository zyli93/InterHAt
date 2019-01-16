#!/usr/bin/env bash
python interprecsys/main.py \
    --trial_id $1 \
    --epoch 2 \
    --batch_size 10 \
    --dataset "safedriver" \
    --use_graph=False \
    --num_iter_per_save 100 \
    --scale_embedding=False \
    --regularization_weight 0.0
