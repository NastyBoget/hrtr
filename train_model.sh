#!/bin/bash

# sudo mount -o resvport -t nfs 10.10.10.224:/data ~/work/htr_copy

python3 train.py --train_data datasets/lmdb/train --valid_data datasets/lmdb/val --select_data "/" \
  --batch_ratio 1 --FT --manualSeed 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM \
  --Prediction CTC --saved_model saved_models/AttentionHTR-General-sensitive.pth --sensitive --lang en
