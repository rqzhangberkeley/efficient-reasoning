#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# evaluate the model from HF
python evaluate_model.py \
    --model_paths "Qwen/Qwen2.5-Math-1.5B" \
    --dataset "di-zhang-fdu/MATH500" \
    --scale 1.5B \
    --n_gpus 4

# run it with nohup: 
# nohup ./evaluate_model.sh > evaluate_model_MATH500_0_Qwen2.5-Math-1.5B.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gsm8k_0.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_aime_trained_lr5e-7_60.log 2>&1 &
