#!/bin/bash

# sudo mount -o resvport -t nfs 10.10.10.224:/data ~/work/htr_copy

DATA_DIR="datasets/lmdb"

if [ -d "$DATA_DIR" ]; then
  echo "Skip dataset creation"
else
  echo "Try to create dataset"
  python3 create_lmdb_dataset_rus.py
fi



python3 train.py --train_data $DATA_DIR/train --valid_data $DATA_DIR/val --select_data "/" \
  --batch_ratio 1 --FT --manualSeed 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM \
  --Prediction CTC --saved_model saved_models/AttentionHTR-General-sensitive.pth --sensitive --lang en
