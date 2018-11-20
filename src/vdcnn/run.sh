#!/usr/bin/env bash
cd ../../
 
dataset="ag_news"

data_folder="datasets/${dataset}/vdcnn"
model_folder="models/vdcnn/${dataset}"
depth=9
maxlen=1024
batch_size=128
epochs=30
lr=0.01
lr_halve_interval=3
gamma=0.9
snapshot_interval=3
gpuid=0
nthreads=4

python -m src.vdcnn.main --dataset ${dataset} \
                         --model_folder ${model_folder} \
                         --data_folder ${data_folder} \
                         --depth ${depth} \
                         --maxlen ${maxlen} \
                         --batch_size ${batch_size} \
                         --epochs ${epochs} \
                         --lr ${lr} \
                         --lr_halve_interval ${lr_halve_interval} \
                         --snapshot_interval ${snapshot_interval} \
                         --gamma ${gamma} \
                         --gpuid ${gpuid} \
                         --nthreads ${nthreads} \
                         --shortcut \
