# !/bin/bash

EXP_NAME="network_congestion_independent"

cp ./args/$EXP_NAME.py arguments.py
CUDA_VISIBLE_DEVICES=0 python -u train.py > $EXP_NAME.log
# mv $EXP_NAME.log ./model_save/$EXP_NAME/train.log
