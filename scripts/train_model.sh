#!/bin/bash

BASE_DIR="datasets"
DATA_DIR="$BASE_DIR/lmdb"
export PYTHONPATH=$PYTHONPATH:src

if [ -d "$DATA_DIR" ]; then
  echo "Skip dataset creation"
else
  echo "Try to create dataset"
  python3 src/process_datasets/create_lmdb_dataset.py --out_dir $BASE_DIR --datasets_list hkr
fi


CUDA_VISIBLE_DEVICES=0 python3 src/train.py --train_data $DATA_DIR/train --valid_data $DATA_DIR/val --select_data "/" \
  --batch_ratio 1 --FT --manualSeed 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM \
  --Prediction Attn --sensitive --lang rus --datasets_list hkr
