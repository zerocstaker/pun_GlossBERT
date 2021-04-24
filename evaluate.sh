#!/bin/bash

DATA_DIR=$1
MODEL_DIR=${2:-"./model"}
OUTPUT_DIR=${3:-"./output"}

python GlossBERT/run_classifier_WSD_sent.py \
--task_name WSD \
--eval_data_dir "$DATA_DIR" \
--output_dir "$OUTPUT_DIR" \
--bert_model "$MODEL_DIR" \
--do_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 128 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--num_train_epochs 6.0 \
--seed 1314