#!/bin/bash

DATA_DIR='datasets'
LOGS_DIR='logs'
export PYTHONPATH=$PYTHONPATH:src

python3 src/transformer_model/train.py --log_dir $LOGS_DIR --log_name train_transformer.txt --out_dir transformer \
  --data_dir $DATA_DIR --label_file gt_hkr.csv
