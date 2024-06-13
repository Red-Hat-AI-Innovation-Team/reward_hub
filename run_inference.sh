
# Define the sampling model
# round1
# model_engine="/new_data/gx/rejection_sampling_checkpoints/merlinite-deep-serenity-19/hf_format/samples_187264"
# round2-sampling
# model_engine="/new_data/gx/iterative_rejection_sampling_checkpoints/round1/merlinite-summer-armadillo-24/hf_format/samples_157440"
# round3-sampling
# model_engine="/new_data/gx/iterative_rejection_sampling_checkpoints/round2/merlinite-helpful-capybara-25/hf_format/samples_68880"

# granite-sampling
model_engine="instructlab/granite-7b-lab"

# Define the reward comparison models
pref_model="mistralai/Mixtral-8x7B-Instruct-v0.1"
ref_model="mistralai/Mixtral-8x7B-v0.1"

# Define the input dataset path
# input_data="/new_data/gx/synthetic_preference/uniform_sample_dataset_30k_best_of_64/bon_sampling_data_split_0.jsonl"
# input_data="/dccstor/gxamr/linux-386/llm-alignment/merlinite_RL_batches/uniform_sample_batch0.jsonl"

# round-1, batch0
# input_data="/new_data/gx/synthetic_preference/merlinite_RL_batches/uniform_sample_batch1.jsonl"

# round-2, batch1
# input_data="/new_data/gx/synthetic_preference/merlinite_RL_batches/uniform_sample_batch2.jsonl"

# granite round1
input_data="/new_data/gx/synthetic_preference/granite_RL_batches/uniform_sample_batch_pre.jsonl"


# Assign the first argument to SHARD_NUMS with a default of 1 if not provided
SHARD_NUMS=${1:-1}

# Assign the second argument to SHARD_IDX with a default of 0 if not provided
SHARD_IDX=${2:-0}

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

    # Run bon scoring
    # python scripts/run_bon_scoring.py --model="$pref_model" --ref_model="$ref_model" --num_threads 2 --batch_size=4 --debug True --pref_sets $input_data > "logs/$filename.log" 2>&1 &
    # echo $!
done
