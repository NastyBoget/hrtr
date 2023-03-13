#!/bin/bash

DATA_DIR='datasets'
LOGS_DIR='logs'
export PYTHONPATH=$PYTHONPATH:src

python3 src/attention_model/train.py --log_dir $LOGS_DIR --log_name train_attention.txt --out_dir attention \
  --data_dir $DATA_DIR --label_files gt_hkr.csv
