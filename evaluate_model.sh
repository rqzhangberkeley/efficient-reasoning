#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python evaluate_model.py \
    --model_paths "GAIR/LIMO" \
    --dataset "datasets/converted_aime_dataset" \
    --scale 32B \
    --n_gpus 8
# run it with nohup: 
# nohup ./evaluate_model.sh > evaluate_model_gpu0.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu1.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu2.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu3.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu4.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu5.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu6.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu7.log 2>&1 &
