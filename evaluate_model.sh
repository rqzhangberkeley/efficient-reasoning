#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

# evaluate the model from HF
python evaluate_model.py \
    --model_paths "./results-grpo/Qwen2.5-Math-1.5B-GRPO-limo-lr5e-6-wd0.1-G7-beta0.005-train_bsz8-gradacc8-seed1001-20250316-005346/checkpoint-36" \
    --dataset "datasets/converted_aime_dataset" \
    --scale 1.5B \
    --n_gpus 4

# run it with nohup: 
# nohup ./evaluate_model.sh > evaluate_model_gpu0.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu1.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu2.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu3.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu4.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu5.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu6.log 2>&1 &
# nohup ./evaluate_model.sh > evaluate_model_gpu7.log 2>&1 &

# Not sure why I can only set n_gpus to 4 when I evaluate the checkpoint.
