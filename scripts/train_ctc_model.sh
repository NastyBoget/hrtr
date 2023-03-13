#!/bin/bash

DATA_DIR='datasets'
LOGS_DIR='logs'
export PYTHONPATH=$PYTHONPATH:src

python3 src/ctc_model/train.py --log_dir $LOGS_DIR --log_name train_ctc.txt --out_dir ctc \
  --data_dir $DATA_DIR --label_files gt_hkr.csv
