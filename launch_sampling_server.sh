# Check if the directory does not exist
if [ ! -d "logs" ]; then
    # Create the directory
    mkdir "logs"
    echo "Directory 'logs' created."
fi

for gpu_idx in {0..7}
do
    CUDA_VISIBLE_DEVICES=$gpu_idx python -u -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --model ibm/merlinite-7b\
        --port 800${gpu_idx} \
        --tensor-parallel-size 1 \
        --load-format auto \
        --download-dir ./download_dir > logs/gen_server_${gpu_idx}.log 2>&1 &
done
