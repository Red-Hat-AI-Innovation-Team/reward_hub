# Define the reward comparison models
pref_model="mistralai/Mixtral-8x7B-Instruct-v0.1"
ref_model="mistralai/Mixtral-8x7B-v0.1"

input_data="/new_data/gx/synthetic_preference/merlinite_RL_batches/uniform_sample_batch0/best_of_64_shard_0.jsonl"

echo "Input data is: $input_data"

bash launch_reward_server.sh

bestn=64
mkdir -p logs

filename_with_extension=$(basename "$input_data")
# Remove the .jsonl extension
filename="${filename_with_extension%.jsonl}"

# Run bon scoring
python scripts/run_bon_scoring.py \
    --model="$pref_model" \
    --ref_model="$ref_model" \
    --num_threads 1 \
    --base_port 8020 \
    --batch_size=8 \
    --debug True \
    --pref_sets $input_data

