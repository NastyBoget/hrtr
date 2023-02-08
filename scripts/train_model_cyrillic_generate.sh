#!/bin/bash

ROOT=$(readlink -f .)
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/src

DATASET='cyrillic'
DATA_DIR='datasets'
OUT_DIR='saved_models'
LOGS_DIR='logs'

python3 src/train.py --train_data $DATA_DIR/train --valid_data $DATA_DIR/val/$DATASET \
  --select_data $DATASET-generate --log_dir $LOGS_DIR --log_name train\_$DATASET\_generate.txt --out_dir $OUT_DIR/$DATASET\_generate \
  --batch_ratio 0.5-0.5 --manual_seed 1 --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM \
  --prediction Attn --sensitive --lang rus
