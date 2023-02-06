#!/bin/bash

DATA_DIR='datasets'
LOGS_DIR='logs'
export PYTHONPATH=$PYTHONPATH:src

echo "Try to create dataset"
python3 src/process_datasets/create_lmdb_dataset.py --out_dir $DATA_DIR --datasets_list hkr cyrillic synthetic \
  --log_dir $LOGS_DIR --log_name datasets_log.txt
