#!/bin/bash

ROOT=$(readlink -f .)
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/src

DATASET='hkr'
DATA_DIR='datasets'
OUT_DIR='saved_models'
LOGS_DIR='logs'

CUDA_VISIBLE_DEVICES=3 python3 src/test.py --test_data $DATA_DIR/test1/$DATASET --select_data $DATASET-generate \
  --log_dir $LOGS_DIR --log_name test1_$DATASET\_generate.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_generate/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive

CUDA_VISIBLE_DEVICES=3 python3 src/test.py --test_data $DATA_DIR/test1/$DATASET --select_data $DATASET-generate \
  --log_dir $LOGS_DIR --log_name test1_$DATASET\_generate\_preprocessing.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_generate\_preprocessing/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive --preprocessing

CUDA_VISIBLE_DEVICES=3 python3 src/test.py --test_data $DATA_DIR/test1/$DATASET --select_data $DATASET-generate \
  --log_dir $LOGS_DIR --log_name test1_$DATASET\_generate\_augmentation.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_generate\_augmentation/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive

CUDA_VISIBLE_DEVICES=3 python3 src/test.py --test_data $DATA_DIR/test1/$DATASET --select_data $DATASET-generate \
  --log_dir $LOGS_DIR --log_name test1_$DATASET\_generate\_preprocessing\_augmentation.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_generate\_preprocessing\_augmentation/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive --preprocessing


CUDA_VISIBLE_DEVICES=3 python3 src/test.py --test_data $DATA_DIR/test2/$DATASET --select_data $DATASET-generate \
  --log_dir $LOGS_DIR --log_name test2_$DATASET\_generate.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_generate/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive

CUDA_VISIBLE_DEVICES=3 python3 src/test.py --test_data $DATA_DIR/test2/$DATASET --select_data $DATASET-generate \
  --log_dir $LOGS_DIR --log_name test2_$DATASET\_generate\_preprocessing.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_generate\_preprocessing/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive --preprocessing

CUDA_VISIBLE_DEVICES=3 python3 src/test.py --test_data $DATA_DIR/test2/$DATASET --select_data $DATASET-generate \
  --log_dir $LOGS_DIR --log_name test2_$DATASET\_generate\_augmentation.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_generate\_augmentation/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive

CUDA_VISIBLE_DEVICES=3 python3 src/test.py --test_data $DATA_DIR/test2/$DATASET --select_data $DATASET-generate \
  --log_dir $LOGS_DIR --log_name test2_$DATASET\_generate\_preprocessing\_augmentation.txt --write_errors --saved_model $ROOT/saved_models/$DATASET\_generate\_preprocessing\_augmentation/best_loss.pth \
  --transformation TPS --feature_extraction ResNet --sequence_modeling BiLSTM --prediction Attn --sensitive --preprocessing
