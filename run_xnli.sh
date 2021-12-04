#!/usr/bin/env bash

model="exp3"
python3 train_xnli.py \
--output_dir "XNLI_MODELS/${model}" \
--logging_steps 50 \
--max_steps 5000 \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 25 \
--save_steps 5000 \
--learning_rate  0.1 \
--adam_beta2 0.98 \
--adam_epsilon 1e-06 \
--dropout 0.1 \
#--resume_from_checkpoint "/projects/tir5/users/dberrebb/CANINE/forked_shiba/shiba/training/models/${model}/checkpoint-10000/pytorch_model.bin"
