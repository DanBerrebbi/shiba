#!/usr/bin/env bash

masking=bpe_span  # masking is rand_span or bpe_span or rand_char
echo $masking

python training/train.py \
--data "training/all_examples.jsonl" \
--logging_steps 50 \
--max_steps 20000 \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 500 \
--save_steps 500 \
--learning_rate 0.0004 \
--adam_beta2 0.98 \
--adam_epsilon 1e-06 \
--dropout 0.1 \
--weight_decay 0.01  \
--output_dir "training/models/first_${masking}" \
--overwrite_output_dir True \
--masking_type rand_span \
--gradient_accumulation_steps 6 \
--masking_type $masking \
--deep_transformer_stack_layers 12 \
--per_device_eval_batch_size 12 \
--per_device_train_batch_size 16
