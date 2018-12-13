#!/usr/bin/env bash
cd ../../
 
model_path="/home/ardalan.mehrani/projects/nlp-benchmarks-github/models/han/ag_news/model_epoch_30"
classes="World Sports Business Sci/Tech"
sentences="""
Will Google researchers create AGI. or will it be Facebook :) ?<sep>
Trump president agreed to invest 2 billion in artificial intelligence. Sorry, was joking :p
"""


python -m src.han.predict --model_path ${model_path} \
                          --classes ${classes} \
                          --sentences "${sentences}"

