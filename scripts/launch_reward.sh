#!/bin/bash

# Define GPU allocation for each model
MODEL1_GPUS="7"


MODEL1_PATH=${1:-"Qwen/Qwen2.5-Math-PRM-7B"}

mkdir -p logs

# Launch Model 1
echo "Launching Model 1 on GPUs $MODEL1_GPUS..."
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=$MODEL1_GPUS python -O -u -m vllm.entrypoints.openai.api_server \
    --port=8305 \
    --dtype=bfloat16 \
    --model=$MODEL1_PATH \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.85 \
    --task=reward \
    --max-model-len=10000 \
    --trust-remote-code > logs/model1.log 2>&1 &
MODEL1_PID=$!