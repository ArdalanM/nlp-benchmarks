#!/usr/bin/env bash
cd ../../
 
# dataset="ag_news"
# dataset="imdb"
dataset="db_pedia"
# dataset="yelp_review"
# dataset="yelp_polarity"

data_folder="datasets/${dataset}/han"
model_folder="models/han/${dataset}"
solver_type="adam"
batch_size="32"
epochs=100
lr=0.0001
lr_halve_interval=10
gamma=0.9
snapshot_interval=10
gpuid=1
nthreads=4


python -m src.han.train --dataset ${dataset} \
                        --data_folder ${data_folder} \
                        --model_folder ${model_folder} \
                        --solver_type ${solver_type} \
                        --batch_size ${batch_size} \
                        --epochs ${epochs} \
                        --lr ${lr} \
                        --lr_halve_interval ${lr_halve_interval} \
                        --gamma ${gamma} \
                        --snapshot_interval ${snapshot_interval} \
                        --gpuid ${gpuid} \
                        --nthreads ${nthreads} \
