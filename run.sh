#!/usr/bin/env bash
python interprecsys/main.py \
    --trial_id $1 \
    --epoch 2 \
    --batch_size 20 \
    --dataset "safedriver" \
    --use_graph=False \
    --num_iter_per_save 100
