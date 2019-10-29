CUDA_VISIBLE_DEVICES=5 python3 interhat/main.py \
   --trial_id 91094 \
   --epoch 15 \
   --batch_size 512 \
   --learning_rate 0.0001 \
   --dataset "ml" \
   --use_graph=False \
   --num_iter_per_save 50000 \
   --scale_embedding=False \
   --regularization_weight 0.0002 \
   --embedding_size 12 \
   --dropout_rate 0 \
   --num_head 8 \
   --attention_size 16 

