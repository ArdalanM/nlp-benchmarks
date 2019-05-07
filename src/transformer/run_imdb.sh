#!/usr/bin/env bash
cd ../../

dataset="imdb"

data_folder="datasets/${dataset}/transformer"
model_folder="models/transformer/${dataset}"
attention_dim=16
n_heads=4
n_layers=4
maxlen=500 # longest sequence
dropout=0.1
n_warmup_step=1000
batch_size=16
epochs=2
snapshot_interval=5
gpuid=0
nthreads=6

python -m src.transformer.train --dataset ${dataset} \
                                 --data_folder ${data_folder} \
                                 --model_folder ${model_folder} \
                                 --attention_dim ${attention_dim} \
                                 --n_heads ${n_heads} \
                                 --n_layers ${n_layers} \
                                 --maxlen ${maxlen} \
                                 --dropout ${dropout} \
                                 --n_warmup_step ${n_warmup_step} \
                                 --batch_size ${batch_size} \
                                 --epochs ${epochs} \
                                 --snapshot_interval ${snapshot_interval} \
                                 --gpuid ${gpuid} \
                                 --nthreads ${nthreads} \

