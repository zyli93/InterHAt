#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICE=4,5
python3 interprecsys/main.py \
    --trial_id $1 \
    --epoch 20 \
    --batch_size 256 \
    --dataset "avazu" \
    --use_graph=False \
    --num_iter_per_save 20000 \
    --scale_embedding=False \
    --regularization_weight 0.01 \
    --embedding_size 128
