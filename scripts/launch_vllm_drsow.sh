#!/bin/bash

# Define GPU allocation for each model
MODEL1_GPUS="1,2"
MODEL2_GPUS="3,4"


MODEL1_PATH=${1:-"Qwen/Qwen2.5-32B-instruct"}
MODEL2_PATH=${2:-"Qwen/Qwen2.5-32B"}

mkdir -p logs

# Launch Model 1
echo "Launching Model 1 on GPUs $MODEL1_GPUS..."
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=$MODEL1_GPUS python -O -u -m vllm.entrypoints.openai.api_server \
    --port=8305 \
    --dtype=bfloat16 \
    --model=$MODEL1_PATH \
    --tensor-parallel-size=2 \
    --gpu-memory-utilization=0.85 \
    --max-model-len=10000 \
    --trust-remote-code \
    --enable_chunked_prefill > logs/model1.log 2>&1 &
MODEL1_PID=$!


# Launch Model 2
echo "Launching Model 2 on GPUs $MODEL2_GPUS..."
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 CUDA_VISIBLE_DEVICES=$MODEL2_GPUS python -O -u -m vllm.entrypoints.openai.api_server \
    --port=8306 \
    --dtype=bfloat16 \
    --model=$MODEL2_PATH \
    --tensor-parallel-size=2 \
    --gpu-memory-utilization=0.85 \
    --max-model-len=10000 \
    --trust-remote-code \
    --enable_chunked_prefill > logs/model2.log 2>&1 &
MODEL2_PID=$!

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

wait_for_server "model1" && wait_for_server "model2" && echo "Both models ready!"