# Check if the directory does not exist
if [ ! -d "logs" ]; then
    # Create the directory
    mkdir "logs"
    echo "Directory 'logs' created."
fi

ray disable-usage-stats
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
ray start --head --num-cpus=32 --num-gpus=8


# Accessing the first and second argument
pref_model="$1"
ref_model="$2"

echo "pref_model: $pref_model"
echo "ref_model: $ref_model"

CUDA_VISIBLE_DEVICES=0,1 python -u -m vllm.entrypoints.openai.api_server \
       --host 0.0.0.0 \
       --model $pref_model \
       --port 8010 \
       --tensor-parallel-size 2 \
       --load-format auto \
       --download-dir /new_data/hf_cache > logs/server_0.log 2>&1 &


CUDA_VISIBLE_DEVICES=2,3 python -u -m vllm.entrypoints.openai.api_server \
       --host 0.0.0.0 \
       --model $ref_model \
       --port 8011 \
       --tensor-parallel-size 2 \
       --load-format auto \
       --download-dir /new_data/hf_cache > logs/server_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5 python -u -m vllm.entrypoints.openai.api_server \
       --host 0.0.0.0 \
       --model $pref_model \
       --port 8012 \
       --tensor-parallel-size 2 \
       --load-format auto \
       --download-dir /new_data/hf_cache > logs/server_2.log 2>&1 &


CUDA_VISIBLE_DEVICES=6,7 python -u -m vllm.entrypoints.openai.api_server \
       --host 0.0.0.0 \
       --model $ref_model \
       --port 8013 \
       --tensor-parallel-size 2 \
       --load-format auto \
       --download-dir /new_data/hf_cache > logs/server_3.log 2>&1 &
