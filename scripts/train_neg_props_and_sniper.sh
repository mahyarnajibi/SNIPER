#!/usr/bin/env bash
# Defining paths for saving trained proposal network and extracted proposals
proposal_training_path='./output/sniper_res101_neg_props'
save_proposal_path='./data/computed_proposals'

# Train a proposal network just for 2 epochs (for negative chip mining)
echo 'Training a proposal network for 2 epochs'
python main_train.py --set TRAIN.USE_NEG_CHIPS False TRAIN.ONLY_PROPOSAL True \
TRAIN.end_epoch 2 output_path ${proposal_training_path}

# Extract proposals for negative mining on training set
echo 'Extracting proposals for negative chip mining'
# Extract proposals for train2014
python main_test.py --set TEST.EXTRACT_PROPOSALS True TEST.PROPOSAL_SAVE_PATH ${save_proposal_path} \
TEST.TEST_EPOCH 2 output_path ${proposal_training_path} dataset.test_image_set train2014
# Extract proposal for val2014
python main_test.py --set TEST.EXTRACT_PROPOSALS True TEST.PROPOSAL_SAVE_PATH ${save_proposal_path} \
TEST.TEST_EPOCH 2 output_path ${proposal_training_path} dataset.test_image_set val2014

# TRAIN SNIPER with negative chip mining
# Use the output path in the config for saving the final SNIPER models
echo 'Training SNIPER with negative chip mining'
python main_train.py --set proposal_path ${save_proposal_path}
