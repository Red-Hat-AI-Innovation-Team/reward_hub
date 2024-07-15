
# granite-sampling
model_engine="instructlab/granite-7b-lab"

# Define the reward comparison models
pref_model="mistralai/Mixtral-8x7B-Instruct-v0.1"
ref_model="mistralai/Mixtral-8x7B-v0.1"

# granite round1
input_data="/new_data/gx/synthetic_preference/granite_RL_batches_new/uniform_sample_batch0.jsonl"


# Assign the first argument to SHARD_NUMS with a default of 1 if not provided
SHARD_NUMS=${1:-1}

# Assign the second argument to SHARD_IDX with a default of 0 if not provided
SHARD_IDX=${2:-0}

# Assign the third argument to input_data; raise an error if not provided
input_data=${3:?"Please provide the input data path as the third argument"}

echo "Shard nums is: $SHARD_NUMS"
echo "Shard index is: $SHARD_IDX"

echo "Input data is: $input_data"

bash launch_sampling_server.sh $model_engine


for bestn in 64; do
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
    --max_prompt_length 2048 \
    --max_new_tokens 1024 \
    --shard_nums "$SHARD_NUMS" \
    --shard_idx "$SHARD_IDX"
    # Echo the process ID of the last background process
    echo $!

done
