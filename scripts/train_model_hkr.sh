#!/bin/bash

ROOT=$(readlink -f .)
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/src

DATASET='hkr'
DATA_DIR='datasets'
OUT_DIR='saved_models'
LOGS_DIR='logs'

CUDA_VISIBLE_DEVICES=0 python3 src/train.py --train_data $DATA_DIR/train --valid_data $DATA_DIR/val/$DATASET \
  --select_data $DATASET --log_dir $LOGS_DIR --log_name train\_$DATASET.txt --out_dir $OUT_DIR/$DATASET \
  --batch_ratio 1 --manual_seed 1 --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM \
  --prediction Attn --sensitive --lang rus
