#!/bin/bash

ROOT=$(readlink -f .)
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/src

DATASET='hkr'
DATA_DIR='datasets'
OUT_DIR='saved_models'
LOGS_DIR='logs'

CUDA_VISIBLE_DEVICES=1 python3 src/train.py --train_data $DATA_DIR/train --valid_data $DATA_DIR/val \
  --select_data $DATASET-generate --log_dir $LOGS_DIR --log_name train\_$DATASET\_generate\_preprocessing.txt --out_dir $OUT_DIR/$DATASET\_generate\_preprocessing \
  --batch_ratio 0.5-0.5 --manual_seed 1 --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM \
  --prediction Attn --sensitive --lang rus --preprocessing
