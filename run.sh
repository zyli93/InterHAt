#!/usr/bin/env bash
python interprecsys/main.py \
    --trial_id $1 \
    --epoch 2 \
    --batch_size 32 \
    --dataset "toy" \
    --use_graph=False \
    --num_iter_per_save 100