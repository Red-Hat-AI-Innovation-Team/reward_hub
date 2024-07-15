# Define the reward comparison models
pref_model="RLHFlow/ArmoRM-Llama3-8B-v0.1"
input_data="/new_data/gx/synthetic_preference/granite_RL_batches_new/uniform_sample_batch0-distribute/best_of_64_distribute_shard_0.jsonl"

mkdir -p logs

python scripts/run_bon_scoring.py \
--model="$pref_model" \
--num_threads 8 \
--model_type "classifier" \
--batch_size=8 \
--input_path $input_data \
--debug True
echo $!
