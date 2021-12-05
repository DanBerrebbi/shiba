#!/usr/bin/env bash

model="exp3"
python3 train_xnli.py \
--output_dir "XNLI_MODELS/${model}" \
--num_train_epochs 10
--logging_steps 50 \
--max_steps 5000 \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 300 \
--save_steps 5000 \
--learning_rate  0.00001 \
--adam_beta2 0.98 \
--adam_epsilon 1e-06 \
--dropout 0.1 \
--overwrite_output_dir False \
--masking_type rand_span \
--gradient_accumulation_steps 24 \
--masking_type $masking \
--deep_transformer_stack_layers 12 \
--per_device_eval_batch_size 2 \
--per_device_train_batch_size 4 \
--resume_from_checkpoint "/projects/tir5/users/dberrebb/CANINE/forked_shiba/shiba/training/models/${model}/checkpoint-10000/pytorch_model.bin"

