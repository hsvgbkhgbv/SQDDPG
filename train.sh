# !/bin/bash

EXP_NAME="traffic_junction_independent"
ALIAS="_medium"

cp ./args/$EXP_NAME.py arguments.py
CUDA_VISIBLE_DEVICES=0 python -u train.py > $EXP_NAME$ALIAS.log
# mv $EXP_NAME.log ./model_save/$EXP_NAME/train.log
