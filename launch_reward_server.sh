# Check if the directory does not exist
if [ ! -d "logs" ]; then
    # Create the directory
    mkdir "logs"
    echo "Directory 'logs' created."
fi

check_success() {
    local file=$1
    local message="Started server process"
    # Tail the log file and grep for success message, exit when found
    tail -f "$file" | grep -q "$message"
    echo "Server at $file has started successfully."
}

ray disable-usage-stats
export OPENBLAS_NUM_THREADS=18
export OMP_NUM_THREADS=18
ray start --head --num-cpus=32 --num-gpus=8


pref_model="mistralai/Mixtral-8x7B-Instruct-v0.1"
ref_model="mistralai/Mixtral-8x7B-v0.1"

echo "pref_model: $pref_model"
echo "ref_model: $ref_model"


get_gpu_count() {
    echo $(nvidia-smi -L | wc -l)
}
# Get the number of available GPUs
num_gpus=$(get_gpu_count)

echo "Detected $num_gpus GPUs."


CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m vllm.entrypoints.openai.api_server \
       --host 0.0.0.0 \
       --model $pref_model \
       --port 8020 \
       --tensor-parallel-size 4 \
       --load-format auto \
       --download-dir /new_data/hf_cache > logs/server_0.log 2>&1 &

sleep 1
# Start monitoring each server log
check_success "logs/server_0.log" &
pid_array+=($!)  # Save the PID of the check_success process



CUDA_VISIBLE_DEVICES=4,5,6,7 python -u -m vllm.entrypoints.openai.api_server \
       --host 0.0.0.0 \
       --model $ref_model \
       --port 8021 \
       --tensor-parallel-size 4 \
       --load-format auto \
       --download-dir /new_data/hf_cache > logs/server_1.log 2>&1 &

sleep 1
# Start monitoring each server log
check_success "logs/server_1.log" &
pid_array+=($!)  # Save the PID of the check_success process

# Wait only for the check_success processes
for pid in "${pid_array[@]}"; do
    wait $pid
done

