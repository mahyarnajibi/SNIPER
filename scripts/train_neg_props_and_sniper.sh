#!/bin/bash 

help_str="Script for extracting proposals required for negative chip mining and training SNIPER with negative chip mining
[options]
--cfg: Path to the SNIPER config file
--prop_save_path: Path to save the extracted proposals for negative chip mining (default=data/computed_proposals)
--prop_train_path: Path for saving the proposal network weights used for negative chip mining (default=output/sniper_neg_props)"

# Parse the arguments
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --cfg)
    cfg_path="$2"
    shift 
    shift
    ;;
    --prop_save_path)
    prop_save_path="$2"
    shift 
    shift 
    ;;
    --prop_train_path)
    prop_train_path="$2"
    shift # past argument
    shift # past value
    ;;
    --help)
    echo "$help_str"
    shift
    exit
    ;;
esac
done

if [ -z ${cfg_path} ]; then 
echo "Please provide a path to the config file as '--cfg [PATH TO CONFIG FILE]'"
printf "\n$help_str\n"
exit; fi
if [ -z ${prop_save_path} ]; then prop_save_path="data/computed_proposals"; fi
if [ -z ${prop_train_path} ]; then prop_train_path="output/sniper_neg_proposals"; fi


# Train a proposal network just for 2 epochs (for negative chip mining)
echo 'Training a proposal network for 2 epochs'
python main_train.py --cfg ${cfg_path} --set TRAIN.USE_NEG_CHIPS False TRAIN.ONLY_PROPOSAL True \
TRAIN.end_epoch 2 output_path ${prop_train_path}

# Extract proposals for negative mining on training set
SET_NAME=$(python -c "
import init
from configs.faster.default_configs import config,update_config
update_config('$cfg_path')
print(config.dataset.image_set)
")
IFS='+' read -ra SETS <<< $SET_NAME
for cset in "${SETS[@]}"; do
	echo Extracting proposals on "$cset for negative chip mining..."
	python main_test.py --cfg ${cfg_path} --set TEST.EXTRACT_PROPOSALS True TEST.PROPOSAL_SAVE_PATH ${prop_save_path} \
	TEST.TEST_EPOCH 2 output_path ${prop_train_path} dataset.test_image_set ${cset}
done


# TRAIN SNIPER with negative chip mining
# Use the output path in the config for saving the final SNIPER models
echo 'Training SNIPER with negative chip mining'
python main_train.py --cfg ${cfg_path} --set proposal_path ${prop_save_path}
