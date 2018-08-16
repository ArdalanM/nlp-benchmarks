#!/usr/bin/env bash
cd ../../
 
# dataset="ag_news"
dataset="imdb"
max_feats=100000
max_sents=8
max_words=32
# dataset="db_pedia"
# dataset="yelp_review"
# dataset="yelp_polarity"


data_folder="datasets/${dataset}/han"
model_folder="models/han/${dataset}"
solver_type="adam"
batch_size="32"
epochs=100
lr=0.0001
lr_halve_interval=-1
gamma=0.9
snapshot_interval=10
gpuid=0

python -m src.han.main  --dataset ${dataset} \
                        --data_folder ${data_folder} \
                        --model_folder ${model_folder} \
                        --max_feats ${max_feats} \
                        --max_sents ${max_sents} \
                        --max_words ${max_words} \
                        --solver_type ${solver_type} \
                        --batch_size ${batch_size} \
                        --epochs ${epochs} \
                        --lr ${lr} \
                        --lr_halve_interval ${lr_halve_interval} \
                        --gamma ${gamma} \
                        --snapshot_interval ${snapshot_interval} \
                        --gpuid ${gpuid} \
                        --pretrain
