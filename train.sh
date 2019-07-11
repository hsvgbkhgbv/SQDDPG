# !/bin/bash

EXP_NAME="spread_sqddpg"
ALIAS="_6_agents_5"

cp ./args/$EXP_NAME.py arguments.py
CUDA_VISIBLE_DEVICES=0 python -u train.py > $EXP_NAME$ALIAS.out &
# mv $EXP_NAME.log ./model_save/$EXP_NAME/train.log
