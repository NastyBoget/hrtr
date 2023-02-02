#!/bin/bash

DATA_DIR="datasets"
LOGS_DIR="logs"
export PYTHONPATH=$PYTHONPATH:src


CUDA_VISIBLE_DEVICES=0 python3 src/train.py --train_data $DATA_DIR/train --valid_data $DATA_DIR/val --select_data "/" \
  --batch_ratio 1 --FT --manualSeed 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM \
  --Prediction Attn --sensitive --lang rus --datasets_list hkr
