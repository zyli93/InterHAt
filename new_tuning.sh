CUDA_VISIBLE_DEVICES=2 python3 interhat/main.py \
   --trial_id 90024 \
   --epoch 10 \
   --batch_size 2048 \
   --dataset "criteoDAC" \
   --use_graph=False \
   --num_iter_per_save 50000 \
   --scale_embedding=False \
   --regularization_weight 0.0002 \
   --embedding_size 12 \
   --dropout_rate 0 \
   --num_head 12 \
   --attention_size 30

CUDA_VISIBLE_DEVICES=2 python3 interhat/main.py \
   --trial_id 90025 \
   --epoch 10 \
   --batch_size 2048 \
   --dataset "criteoDAC" \
   --use_graph=False \
   --num_iter_per_save 50000 \
   --scale_embedding=False \
   --regularization_weight 0.0002 \
   --embedding_size 16\
   --dropout_rate 0 \
   --num_head 12 \
   --attention_size 32

CUDA_VISIBLE_DEVICES=2 python3 interhat/main.py \
   --trial_id 90026 \
   --epoch 10 \
   --batch_size 2048 \
   --dataset "criteoDAC" \
   --use_graph=False \
   --num_iter_per_save 50000 \
   --scale_embedding=False \
   --regularization_weight 0.0002 \
   --embedding_size 32\
   --dropout_rate 0 \
   --num_head 12 \
   --attention_size 32


CUDA_VISIBLE_DEVICES=2 python3 interhat/main.py \
   --trial_id 90027 \
   --epoch 10 \
   --batch_size 2048 \
   --dataset "criteoDAC" \
   --use_graph=False \
   --num_iter_per_save 50000 \
   --scale_embedding=False \
   --regularization_weight 0.0002 \
   --embedding_size 32\
   --dropout_rate 0 \
   --num_head 10 \
   --attention_size 32
