#!/bin/bash

ROOT=$(readlink -f .)
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/src

DATASET='cyrillic'
DATA_DIR='datasets'
OUT_DIR='saved_models'
LOGS_DIR='logs'

CUDA_VISIBLE_DEVICES=3 python3 src/test.py --test_data $DATA_DIR/test/$DATASET --select_data $DATASET-synthetic \
  --log_dir $LOGS_DIR --log_name test_$DATASET\_synthetic.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_synthetic/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive

CUDA_VISIBLE_DEVICES=3 python3 src/test.py --test_data $DATA_DIR/test/$DATASET --select_data $DATASET-synthetic \
  --log_dir $LOGS_DIR --log_name test_$DATASET\_synthetic\_preprocessing.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_synthetic\_preprocessing/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive --preprocessing

CUDA_VISIBLE_DEVICES=3 python3 src/test.py --test_data $DATA_DIR/test/$DATASET --select_data $DATASET-synthetic \
  --log_dir $LOGS_DIR --log_name test_$DATASET\_synthetic\_preprocessing\_augmentation.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_synthetic\_preprocessing\_augmentation/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive --preprocessing
