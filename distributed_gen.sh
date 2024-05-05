#!/bin/bash

# Function to get the number of available GPUs
get_gpu_count() {
    nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
}

# Function to launch a job on a specific GPU
launch_job() {
    local gpu_idx=$1
    python scripts/n_inference.py --decoder_name_or_path ibm/merlinite-7b --port 800${gpu_idx} --output_path ./shard-splits-4_chunk_${gpu_idx}.jsonl --dataset_path="/dccstor/gxamr/linux-386/llm-alignment/noised_ppo_merlinite_train.jsonl" --chunk_idx $gpu_idx --max_instances 100 > logs/gpu_${gpu_idx}.log 2>&1 &
}

# Get the number of available GPUs
num_gpus=$(get_gpu_count)
echo "Detected $num_gpus GPUs."

# Launch one job per GPU
for ((gpu_idx=0; gpu_idx<num_gpus; gpu_idx++)); do
    echo "Launching job on GPU $gpu_idx"
    launch_job $gpu_idx
done
