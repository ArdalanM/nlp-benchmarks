#!/usr/bin/env bash
cd ../../
 
dataset="db_pedia"

data_folder="datasets/${dataset}/han"
model_folder="models/han/${dataset}"
solver_type="adam"
batch_size="32"
epochs=100
lr=0.0005
max_grad_norm=1
lr_halve_interval=10
gamma=0.9
snapshot_interval=10
gpuid=0
nthreads=4


python -m src.han.train --dataset ${dataset} \
                        --data_folder ${data_folder} \
                        --model_folder ${model_folder} \
                        --solver_type ${solver_type} \
                        --batch_size ${batch_size} \
                        --epochs ${epochs} \
                        --lr ${lr} \
                        --lr_halve_interval ${lr_halve_interval} \
                        --max_grad_norm ${max_grad_norm} \
                        --gamma ${gamma} \
                        --snapshot_interval ${snapshot_interval} \
                        --gpuid ${gpuid} \
                        --nthreads ${nthreads} \
                        --curriculum \
