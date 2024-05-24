# Define the reward comparison models
pref_model="mistralai/Mixtral-8x7B-Instruct-v0.1"
ref_model="mistralai/Mixtral-8x7B-v0.1"

# Define the input dataset path
# input_data="/new_data/gx/synthetic_preference/uniform_sample_dataset_30k_best_of_64/bon_sampling_data_split_0.jsonl"
input_data="/dccstor/gxamr/linux-386/llm-alignment/merlinite_RL_batches/uniform_sample_batch0.jsonl"
input_data="/new_data/gx/synthetic_preference/merlinite_RL_batches/uniform_sample_batch0/best_of_64_shard_0.jsonl"


# Assign the second argument to SHARD_IDX with a default of 0 if not provided
SHARD_IDX=${1:-0}

echo "Shard index is: $SHARD_IDX"

bestn=64
mkdir -p logs

filename_with_extension=$(basename "$input_data")
# Remove the .jsonl extension
filename="${filename_with_extension%.jsonl}"

# "$output_dir/best_of_${bestn}_shard_${SHARD_IDX}.jsonl"

# Run bon scoring
python scripts/run_bon_scoring.py \
    --model="$pref_model" \
    --ref_model="$ref_model" \
    --num_threads 1 \
    --base_port 8020 \ # where to expect server port to start from
    --batch_size=4 \
    --debug True \
    --pref_sets $input_data

