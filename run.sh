#!/bin/bash
gpu_id=1

experiment_name="mit-b0-new-changhua-reproduce"
# experiment_name="eff-b7-new-changhua"

exp_dir=exp/$experiment_name
source_data_root="data/"

echo "Training model"
CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
	--exp_dir $exp_dir \
	--name $experiment_name
