#!/bin/bash

ROOT=$(readlink -f .)
DATA_DIR=$ROOT/datasets
LOGS_DIR=$ROOT/logs
export PYTHONPATH=$PYTHONPATH:$ROOT:$ROOT/src


CUDA_VISIBLE_DEVICES=0 python3 src/train.py --train_data $DATA_DIR/train --valid_data $DATA_DIR/val/hkr --select_data "hkr" \
  --batch_ratio 1 --manualSeed 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM \
  --Prediction Attn --sensitive --lang rus --log_dir $LOGS_DIR --log_name train_hkr.txt --out_dir saved_models/hkr1
