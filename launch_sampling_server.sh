# Check if the directory does not exist
if [ ! -d "logs" ]; then
    # Create the directory
    mkdir "logs"
    echo "Directory 'logs' created."
fi


# Accessing the first
server_engine="$1"

echo "Server Engine is : $server_engine"

get_gpu_count() {
    echo $(nvidia-smi -L | wc -l)
}
# Get the number of available GPUs
num_gpus=$(get_gpu_count)

echo "Detected $num_gpus GPUs."

for ((gpu_idx=0; gpu_idx<num_gpus; gpu_idx++))
do
    CUDA_VISIBLE_DEVICES=$gpu_idx python -u -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --model $server_engine \
        --port 800${gpu_idx} \
        --tensor-parallel-size 1 \
        --load-format auto \
        --dtype float16 \
        --download-dir ./download_dir > logs/gen_server_${gpu_idx}.log 2>&1 &
done
