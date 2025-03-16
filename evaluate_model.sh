#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# evaluate the model from HF
python evaluate_model.py \
    --model_paths "Qwen/Qwen2.5-Math-1.5B" \
    --dataset "di-zhang-fdu/MATH500" \
    --scale 1.5B \
    --n_gpus 4

# run it with nohup: 
# nohup ./evaluate_model.sh > evaluate_model_MATH500_0_additional_prompt.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gsm8k_0.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu2.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu3.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu4.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu5.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu6.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu7.log 2>&1 &

# Not sure why I can only set n_gpus to 4 when I evaluate the checkpoint.
