# Check if the directory does not exist
if [ ! -d "logs" ]; then
    # Create the directory
    mkdir "logs"
    echo "Directory 'logs' created."
fi


# Accessing the first and second argument
pref_model="$1"
ref_model="$2"

echo "pref_model: $pref_model"
echo "ref_model: $ref_model"

CUDA_VISIBLE_DEVICES=5 python -u -m vllm.entrypoints.openai.api_server \
       --host 0.0.0.0 \
       --model $pref_model \
       --port 8010 \
       --tensor-parallel-size 1 \
       --load-format auto \
       --download-dir ./download_dir > logs/server_0.log 2>&1 &


CUDA_VISIBLE_DEVICES=7 python -u -m vllm.entrypoints.openai.api_server \
       --host 0.0.0.0 \
       --model $ref_model \
       --port 8011 \
       --tensor-parallel-size 1 \
       --load-format auto \
       --download-dir ./download_dir > logs/server_1.log 2>&1 &
