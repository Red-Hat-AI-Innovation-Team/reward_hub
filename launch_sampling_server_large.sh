# Check if the directory does not exist
if [ ! -d "logs" ]; then
    # Create the directory
    mkdir "logs"
    echo "Directory 'logs' created."
fi

ray disable-usage-stats
export OPENBLAS_NUM_THREADS=18
export OMP_NUM_THREADS=18
ray start --head --num-cpus=32 --num-gpus=8

check_success() {
    local file=$1
    local message="Started server process"
    # Tail the log file and grep for success message, exit when found
    tail -f "$file" | grep -q "$message"
    echo "Server at $file has started successfully."
}

# Accessing the first
server_engine="$1"

echo "Server Engine is : $server_engine"

get_gpu_count() {
    echo $(nvidia-smi -L | wc -l)
}
# Get the number of threads to launch
num_threads=2

echo "Number of vllm copies to launch: $num_threads"

for thread_idx in $(seq 0 $((num_threads - 1))); do
    # cuda visible devices is idx*4, idx*4+1, idx*4+2, idx*4+3
    CUDA_VISIBLE_DEVICES=$((thread_idx * 4)),$((thread_idx * 4 + 1)),$((thread_idx * 4 + 2)),$((thread_idx * 4 + 3)) python -u -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --model $server_engine \
        --port 802${thread_idx} \
        --tensor-parallel-size 4 \
        --load-format auto \
        --dtype float16 \
        --download-dir ./download_dir > logs/sampling_server_${thread_idx}.log 2>&1 &
    sleep 1
    # Start monitoring each server log
    check_success "logs/sampling_server_${thread_idx}.log" &
    pid_array+=($!)  # Save the PID of the check_success process
done


# Wait only for the check_success processes
for pid in "${pid_array[@]}"; do
    wait $pid
done
