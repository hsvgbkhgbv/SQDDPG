# !/bin/bash

EXP_NAME="predator_prey_commnet"

cp ./args/$EXP_NAME.py arguments.py
python -u train.py > train.log
cp train.log ./model_save/$EXP_NAME/train.log
