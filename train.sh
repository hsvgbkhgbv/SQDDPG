# !/bin/bash
# sh train.sh



EXP_NAME="simple_tag_sqddpg"
ALIAS=""

cp ./args/$EXP_NAME.py arguments.py
CUDA_VISIBLE_DEVICES=1 python -u train.py >  $EXP_NAME$ALIAS.log &
echo $! > $EXP_NAME$ALIAS.pid
# mv $EXP_NAME.log ./model_save/$EXP_NAME/train.log
