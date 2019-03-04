#!/usr/bin/env bash
cd ../../

# base model (GPU ram > 8GB): embedding_dim=512, attention_dim=64, n_heads=8, n_layers=6, dropout=0.1, n_warmup_step=4000, batch_size=64
# big model (GPU ram > ?): embedding_dim=1024, attention_dim=64, n_heads=16, n_layers=6, dropout=0.1, n_warmup_step=4000, batch_size=64
# beware when max_sequence_length=-1, it will pad to the longest sequence which can be very long and cause GPU memory error

dataset="imdb"

data_folder="datasets/${dataset}/transformer"
model_folder="models/transformer/${dataset}"
embedding_dim=32
attention_dim=64
n_heads=3
n_layers=2
maxlen=200 # longest sequence will be calculated on training set
dropout=0.1
n_warmup_step=4000
batch_size=64
epochs=100
snapshot_interval=5
gpuid=1
nthreads=6

python -m src.transformer.train --dataset ${dataset} \
                                 --data_folder ${data_folder} \
                                 --model_folder ${model_folder} \
                                 --embedding_dim ${embedding_dim} \
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

