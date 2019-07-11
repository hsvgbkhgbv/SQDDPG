# !/bin/bash
# sh train.sh

EXP_NAME="simple_tag_sqddpg"
ALIAS=""

cp ./args/$EXP_NAME.py arguments.py
CUDA_VISIBLE_DEVICES=0 python -u train.py > $EXP_NAME$ALIAS.out &
echo $! > $EXP_NAME$ALIAS.pid
