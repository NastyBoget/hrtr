#!/bin/bash

DATA_DIR='datasets'
LOGS_DIR='logs'
export PYTHONPATH=$PYTHONPATH:src

python3 src/transformer_model/train.py --log_dir $LOGS_DIR --log_name test_transformer.txt --write_errors \
  --data_dir $DATA_DIR --label_files gt_hkr.csv --saved_model transformer_hkr/model.ckpt --eval_mode --eval_stage test1
