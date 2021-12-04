#!/usr/bin/env bash

model="exp_simclr_alpha0.8_bs4_acc8"
python3 train_xnli.py \
--output_dir "XNLI_MODELS/${model}" \
--logging_steps 1000 \
--max_steps 100000 \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 2000 \
--save_steps 5000 \
--learning_rate  0.0004 \
--adam_beta2 0.98 \
--adam_epsilon 1e-06 \
--dropout 0.1 \
--resume_from_checkpoint "/projects/tir5/users/dberrebb/CANINE/forked_shiba/shiba/training/models/${model}/checkpoint-10000/pytorch_model.bin"
