#!/bin/bash

# Define GPU allocation for each model
MODEL1_GPUS="5"

# Paths to the models
MODEL1_PATH=${1:-"microsoft/phi-4"}

mkdir -p logs

# Launch Model 1
echo "Launching Model 1 on GPUs $MODEL1_GPUS..."
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=$MODEL1_GPUS python -O -u -m vllm.entrypoints.openai.api_server \
    --port=8000 \
    --dtype=bfloat16 \
    --model=$MODEL1_PATH \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.85 \
    --max-model-len=20000 \
    --trust-remote-code \
    --enable_chunked_prefill > logs/model0.log 2>&1 &
MODEL1_PID=$!d

# Wait for servers to be ready
wait_for_server() {
    for i in {1..1000}; do
        if grep -q "Started server process" "logs/$1.log"; then
            echo "$1 server started successfully!"
            # Return 0 (success) to indicate the server started successfully
            # This is standard shell function behavior - 0 means success, non-zero means error
            return 0
        fi
        echo "Waiting for $1... (${i}/1000 attempts)"
        sleep 5
    done
    echo "ERROR: $1 failed to start"
    exit 1
}

wait_for_server "model0" && echo "Both models ready!"