# Check if the directory does not exist
if [ ! -d "logs" ]; then
    # Create the directory
    mkdir "logs"
    echo "Directory 'logs' created."
fi

CUDA_VISIBLE_DEVICES=0 python -u -m vllm.entrypoints.openai.api_server \
       --host 0.0.0.0 \
       --model NousResearch/Nous-Hermes-2-Mistral-7B-DPO \
       --port 8000 \
       --tensor-parallel-size 1 \
       --load-format auto \
       --download-dir ./download_dir > logs/server_0.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -u -m vllm.entrypoints.openai.api_server \
       --host 0.0.0.0 \
       --model teknium/OpenHermes-2.5-Mistral-7B \
       --port 8001 \
       --tensor-parallel-size 1 \
       --load-format auto \
       --download-dir ./download_dir > logs/server_1.log 2>&1 &
