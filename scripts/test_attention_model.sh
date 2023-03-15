#!/bin/bash

DATA_DIR='datasets'
LOGS_DIR='logs'
export PYTHONPATH=$PYTHONPATH:src

python3 src/attention_model/test.py --log_dir $LOGS_DIR --log_name test_attention.txt --write_errors \
  --data_dir $DATA_DIR --label_files gt_hkr.csv --saved_model attention_hkr/best_cer.pth --eval_stage test1
