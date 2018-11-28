#!/usr/bin/env bash
cd ../../
 
dataset="ag_news"

data_folder="datasets/${dataset}/vdcnn"
model_folder="models/vdcnn/${dataset}"
depth=9
solver='sgd'
momentum=0.9
gamma=0.9
lr_halve_interval=15
maxlen=1024
batch_size=128
epochs=100
lr=0.01
snapshot_interval=3
gpuid=0
nthreads=4

python -m src.vdcnn.main --dataset ${dataset} \
                         --model_folder ${model_folder} \
                         --data_folder ${data_folder} \
                         --depth ${depth} \
                         --solver ${solver} \
                         --maxlen ${maxlen} \
                         --batch_size ${batch_size} \
                         --epochs ${epochs} \
                         --lr ${lr} \
                         --lr_halve_interval ${lr_halve_interval} \
                         --momentum ${momentum} \
                         --gamma ${gamma} \
                         --snapshot_interval ${snapshot_interval} \
                         --gpuid ${gpuid} \
                         --nthreads ${nthreads} \
                         --shortcut \
