# !/bin/bash
# sh train.sh

EXP_NAME="simple_tag_maddpg"
ALIAS="_fix"
CUDA_VISIBLE_DEVICES=0

mkdir ./model_save/$EXP_NAME$ALIAS
cp ./args/$EXP_NAME.py arguments.py
python -u train.py > ./model_save/$EXP_NAME$ALIAS/exp.out &
