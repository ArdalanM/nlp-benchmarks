#!/usr/bin/env bash
cd ../../
dataset="sogu_news"
model_folder="models/CNN/CNN_${dataset}_small"
epoch_size=5000
batch_size=128
iterations=$(($epoch_size*50))
halving=$((3*$epoch_size))

python -m src.CNN   --dataset "${dataset}" \
                    --model_folder "${model_folder}" \
                    --maxlen 1014 \
                    --batch_size ${batch_size} \
                    --test_batch_size ${batch_size} \
                    --test_interval ${epoch_size} \
                    --iterations ${iterations} \
                    --lr 0.01 \
                    --lr_halve_interval ${halving} \
                    --seed 1337 \
                    --small_config \
                    --gpu