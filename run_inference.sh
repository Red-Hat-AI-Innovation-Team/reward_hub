
# Assign the first argument to input_data; raise an error if not provided
input_data=${1:?"Please provide the input data path as the third argument"}

# Assign the second argument to model_engine; raise an error if not provided
model_engine=${2:?"Please provide the model engine as the fourth argument"}

# Assign the third argument to SHARD_NUMS with a default of 1 if not provided
SHARD_NUMS=${3:-1}

# Assign the fourth argument to SHARD_IDX with a default of 0 if not provided
SHARD_IDX=${4:-0}

echo "Input data is: $input_data"
echo "Model engine is: $model_engine"
echo "Shard nums is: $SHARD_NUMS"
echo "Shard index is: $SHARD_IDX"

bash launch_sampling_server.sh $model_engine

for bestn in 32; do
    filename_with_extension=$(basename "$input_data")
    filename="${filename_with_extension%.jsonl}"

    output_dir=$(dirname "$input_data")/$filename-distribute
    mkdir -p "$output_dir"  # Ensures the directory exists

    # Launch repeat_n_sampling with shard arguments
    python scripts/repeat_n_sampling.py \
    --decoder_name_or_path "$model_engine" \
    --base_port 8020 \
    --output_path "$output_dir/best_of_${bestn}_distribute_shard_${SHARD_IDX}.jsonl" \
    --dataset_path "$input_data" \
    --num_return_sequences "$bestn" \
    --vllm_batch_size 20 \
    --num_threads 8 \
    --max_prompt_length 2048 \
    --max_new_tokens 1024 \
    --shard_nums "$SHARD_NUMS" \
    --shard_idx "$SHARD_IDX"
    # Echo the process ID of the last background process
    echo $!

done
