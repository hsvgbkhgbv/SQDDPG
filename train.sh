# !/bin/bash

EXP_NAME="network_congestion_independent_ddpg"

cp ./args/$EXP_NAME.py arguments.py
python -u train.py > $EXP_NAME.py
mv $EXP_NAME.log ./model_save/$EXP_NAME/train.log
