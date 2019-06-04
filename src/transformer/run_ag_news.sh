#!/usr/bin/env bash
cd ../../

# date=`date '+%Y%m%d_%H%M%S'`

dataset="ag_news"
data_folder="datasets/${dataset}/transformer"
model_folder="models/transformer/${dataset}"
# log_folder="${model_folder}/logs/$date"

# preprocessing
word_min_count=5

# model
attention_dim=16
n_heads=4
n_layers=4
maxlen=102 
dropout=0.1
ff_hidden_size=16

# optimizer
opt_name="adam_warmup_linear"
weight_decay=0.
n_warmup_step=1500
max_grad_norm=10

# training    
batch_size=64
epochs=100
snapshot_interval=10
gpuid=1
nthreads=4
seed=1337

python -m src.transformer.train \
                                 --dataset ${dataset} \
                                 --data_folder ${data_folder} \
                                 --model_folder ${model_folder} \
                                 --word_min_count ${word_min_count} \
                                 --attention_dim ${attention_dim} \
                                 --n_heads ${n_heads} \
                                 --n_layers ${n_layers} \
                                 --maxlen ${maxlen} \
                                 --dropout ${dropout} \
                                 --ff_hidden_size ${ff_hidden_size} \
                                 --opt_name ${opt_name} \
                                 --weight_decay ${weight_decay} \
                                 --n_warmup_step ${n_warmup_step} \
                                 --max_grad_norm ${max_grad_norm} \
                                 --batch_size ${batch_size} \
                                 --epochs ${epochs} \
                                 --snapshot_interval ${snapshot_interval} \
                                 --gpuid ${gpuid} \
                                 --nthreads ${nthreads} \
                                 --seed ${seed} \
                                 --curriculum

