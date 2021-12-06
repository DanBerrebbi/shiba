#!/usr/bin/env bash

masking=rand_span  # masking is rand_span or bpe_span or rand_char
echo $masking

python training/train.py \
--data "training/all_examples.jsonl" \
--logging_steps 100 \
--max_steps 3000 \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 500 \
--save_steps 500 \
--learning_rate 0.000001 \
--adam_beta2 0.98 \
--adam_epsilon 1e-06 \
--dropout 0.1 \
--weight_decay 0.01  \
--output_dir "training/models/exp_contrast_over_jimin" \
--overwrite_output_dir False \
--masking_type rand_span \
--gradient_accumulation_steps 50 \
--masking_type $masking \
--deep_transformer_stack_layers 6 \
--per_device_eval_batch_size 2 \
--per_device_train_batch_size 4 \
--resume_from_checkpoint "/projects/tir5/users/dberrebb/CANINE/forked_shiba/shiba/training/models/jimin_checkpoint-20000/checkpoint-20000" \

