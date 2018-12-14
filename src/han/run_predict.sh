#!/usr/bin/env bash
 
model_path="/home/ardalan.mehrani/projects/nlp-benchmarks-github/models/han/ag_news/model_epoch_30"
classes="World Sports Business Sci/Tech"
sentences="""
Will Google researchers create AGI. or will it be Facebook :) ?<sep>
President Trump agreed to invest 2 billions in artificial intelligence. Sorry, was joking :p
"""


python predict.py  --model_path ${model_path} \
                   --classes ${classes} \
                   --sentences "${sentences}"

