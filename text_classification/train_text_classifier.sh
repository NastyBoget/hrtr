#!/bin/bash

# sudo mount -o resvport -t nfs 10.10.10.224:/data ~/work/htr_copy

python3 train_text_classifier.py --data_path ../datasets --save_model_path ../saved_models
