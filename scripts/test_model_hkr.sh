#!/bin/bash

ROOT=$(readlink -f .)
DATA_DIR=$ROOT/datasets
LOGS_DIR=$ROOT/logs
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/src


CUDA_VISIBLE_DEVICES=0 python3 src/test.py --test_data $DATA_DIR/test1/hkr --select_data "hkr" \
  --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM \
  --Prediction Attn --sensitive --log_dir $LOGS_DIR --log_name train_hkr.txt --write_errors \
  --saved_model $ROOT/saved_models/hkr/best_cer.pth
