#!/bin/bash

export WANDB_MODE=disabled
export n=20
export model=$1

export len=100
export token_num=2
steps=500
system_message=False
method=gjo
data_path=../data/harmbench_${method}.csv

if [ ! -d "./results" ]; then
    mkdir "./results"
    echo "Folder './results' created."
else
    echo "Folder './results' already exists."
fi

if [ ! -d "./logs" ]; then
    mkdir "./logs"
    echo "Folder './logs' created."
else
    echo "Folder './logs' already exists."
fi

control_init=""
for ((i=0; i<len; i++)); do
    control_init+="! "  # Append "! " to the control_init variable
done

python -u ./main.py \
    --config="./configs/transfer_${model}.py" \
    --config.attack=gcg \
    --config.train_data=${data_path}\
    --config.result_prefix="./results/transfer_${model}_gcg_${n}_progressive_w_sys_len_${len}_suffix" \
    --config.progressive_goals=True \
    --config.stop_on_success=False \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_test_data=$n \
    --config.n_steps=${steps} \
    --config.test_steps=5 \
    --config.batch_size=128 \
    --config.topk=256 \
    --config.ce_prefix=True \
    --config.system_message=${system_message} \
    --config.control_init="$control_init" \
    --config.token_num=${token_num} | tee logs/data_${n}_len_${len}_${model}_sys_${system_message}_${token_num}_token.txt