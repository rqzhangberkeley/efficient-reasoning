#!/bin/bash

wandb login 363018e9dc8339fae726d3b48a839f262c457194

# Function to send email notification
send_email() {
    local subject="$1"
    local body="$2"
    echo "$body" | mail -s "$subject" rqzhang@berkeley.edu
}

# Function to run a single experiment
run_experiment() {
    local lr="$1"
    local wd="$2"
    local G="$3"
    local seed="$4"
    local per_device_train_batch_size="$5"
    local gradient_accumulation_steps="$6"
    local beta="$7"
    local max_grad_norm="$8"
    local max_samples="$9" # -1 means use all data

    # Set environment variables for this run
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # Use all 8 GPUs
    export dataset_name="limo" # A 7.5k dataset.
    export num_processes="7"  # Using 7 GPUs for training.
    export per_device_eval_batch_size="32"
    
    # Generate unique ID for this run
    export uid="$(date +%Y%m%d-%H%M%S)"
    export DISABLE_WANDB=1
    
    # Create output directory name
    export output_dir="./results/QWen-2.5-1.5B-Open-R1-GRPO-${dataset_name}-lr${lr}-wd${wd}-G${G}-beta${beta}-train_bsz${per_device_train_batch_size}-gradacc${gradient_accumulation_steps}-seed${seed}-${uid}"
    
    echo "Starting experiment with:"
    echo "Learning rate: $lr"
    echo "Weight decay: $wd"
    echo "G: $G"
    echo "Beta: $beta"
    echo "Seed: $seed"
    echo "Batch size: $per_device_train_batch_size"
    echo "Gradient accumulation: $gradient_accumulation_steps"
    echo "Using all 8 GPUs"
    
    # Run the experiment and capture both stdout and stderr
    {
        ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
            --num_processes=${num_processes} src/open_r1/grpo.py \
            --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml \
            --model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct \
            --dataset_name=openai/gsm8k \
            --output_dir=${output_dir} \
            --per_device_train_batch_size=${per_device_train_batch_size} \
            --per_device_eval_batch_size=${per_device_eval_batch_size} \
            --gradient_accumulation_steps=${gradient_accumulation_steps} \
            --num_generations=${G} \
            --weight_decay=${wd} \
            --max_grad_norm=${max_grad_norm} \
            --beta=${beta} \
            --learning_rate=${lr} \
            --lr_scheduler_type=cosine \
            --max_samples=${max_samples} \
            --seed=${seed} \
            --report_to=wandb 2>&1
    } # > logs/grpo_Qwen1.5B_${dataset_name}_${uid}.txt
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        # Get the last few lines of the log for error context
        local error_context=$(tail -n 20 logs/grpo_Qwen1.5B_${dataset_name}_${uid}.txt)
        send_email "Experiment Failed" "Experiment with lr=${lr}, wd=${wd}, G=${G}, seed=${seed} failed with exit code ${exit_code}. Error context:\n\n${error_context}"
    else
        send_email "Experiment Completed" "Successfully completed experiment with lr=${lr}, wd=${wd}, G=${G}, seed=${seed}"
    fi
}

# Define hyperparameter configurations
# Format: "lr wd G seed train_batch_size_per_device grad_accum beta max_grad_norm max_samples"
declare -a configs=(
    "1e-6 0.1 4 1001 32 1 0.005 0.1 -1"       # baseline
)

# Create a directory for job status
mkdir -p job_status

# Submit each job sequentially
for i in "${!configs[@]}"; do
    config=(${configs[$i]})
    
    # Run the experiment and wait for it to complete before starting next one
    run_experiment "${config[0]}" "${config[1]}" "${config[2]}" "${config[3]}" "${config[4]}" "${config[5]}" "${config[6]}" "${config[7]}" "${config[8]}"
    
    echo "Completed job $i"
done

echo "All jobs completed"