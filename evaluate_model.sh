#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
python evaluate_model.py \
    --model_paths "GAIR/LIMO" \
    --dataset "GAIR/LIMO" \
    --scale 32B \
    --tok_limit 15872

# run it with nohup: 
# nohup ./evaluate_model.sh > evaluate_model_gpu0.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu1.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu2.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu3.log 2>&1 &
