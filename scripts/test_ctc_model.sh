#!/bin/bash

DATA_DIR='datasets'
LOGS_DIR='logs'
export PYTHONPATH=$PYTHONPATH:src

python3 src/ctc_model/test.py --log_dir $LOGS_DIR --log_name test_ctc.txt --write_errors \
  --data_dir $DATA_DIR --label_files gt_hkr.csv --saved_model ctc_hkr/best_cer.pt --eval_stage test1
