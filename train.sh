# !/bin/bash
# sh train.sh

EXP_NAME="simple_tag_independent_ac"
ALIAS="_fix"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

mkdir ./model_save/$EXP_NAME$ALIAS
cp ./args/$EXP_NAME.py arguments.py
python -u train.py > ./model_save/$EXP_NAME$ALIAS/exp.out &
echo $! > ./model_save/$EXP_NAME$ALIAS/exp.pid
