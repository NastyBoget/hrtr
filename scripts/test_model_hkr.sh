#!/bin/bash

ROOT=$(readlink -f .)
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/src

DATASET='hkr'
DATA_DIR='datasets'
OUT_DIR='saved_models'
LOGS_DIR='logs'

CUDA_VISIBLE_DEVICES=0 python3 src/test.py --test_data $DATA_DIR/test1/$DATASET --select_data $DATASET \
  --log_dir $LOGS_DIR --log_name test1_$DATASET.txt --write_errors --saved_model $ROOT/saved_models/$DATASET/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive

CUDA_VISIBLE_DEVICES=0 python3 src/test.py --test_data $DATA_DIR/test2/$DATASET --select_data $DATASET \
  --log_dir $LOGS_DIR --log_name test2_$DATASET.txt --write_errors --saved_model $ROOT/saved_models/$DATASET/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive

CUDA_VISIBLE_DEVICES=0 python3 src/test.py --test_data $DATA_DIR/test1/$DATASET --select_data $DATASET \
  --log_dir $LOGS_DIR --log_name test1_$DATASET\_preprocessing.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_preprocessing/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive --preprocessing

CUDA_VISIBLE_DEVICES=0 python3 src/test.py --test_data $DATA_DIR/test2/$DATASET --select_data $DATASET \
  --log_dir $LOGS_DIR --log_name test2_$DATASET\_preprocessing.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_preprocessing/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive --preprocessing

CUDA_VISIBLE_DEVICES=0 python3 src/test.py --test_data $DATA_DIR/test1/$DATASET --select_data $DATASET \
  --log_dir $LOGS_DIR --log_name test1_$DATASET\_preprocessing\_augmentation.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_preprocessing\_augmentation/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive --preprocessing

CUDA_VISIBLE_DEVICES=0 python3 src/test.py --test_data $DATA_DIR/test2/$DATASET --select_data $DATASET \
  --log_dir $LOGS_DIR --log_name test2_$DATASET\_preprocessing\_augmentation.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_preprocessing\_augmentation/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive --preprocessing
