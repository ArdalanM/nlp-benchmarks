#!/usr/bin/env bash
cd ../../
dataset="sogu_news"
depth=17
model_folder="models/VDCNN/VDCNN_${dataset}_depth@${depth}"
epoch_size=5000
batch_size=128
iterations=$(($epoch_size*50))
halving=$((3*$epoch_size))

python -m src.VDCNN --dataset "${dataset}" \
                    --model_folder "${model_folder}" \
                    --depth ${depth} \
                    --maxlen 1024 \
                    --chunk_size 2048 \
                    --batch_size ${batch_size} \
                    --test_batch_size ${batch_size} \
                    --test_interval ${epoch_size} \
                    --iterations ${iterations} \
                    --lr 0.01 \
                    --lr_halve_interval ${halving} \
                    --seed 1337 \
