# Define the reward comparison models
pref_model="RLHFlow/ArmoRM-Llama3-8B-v0.1"
input_data="/new_data/gx/synthetic_preference/merlinite_RL_batches/uniform_sample_batch0.jsonl"

mkdir -p logs

python scripts/run_bon_scoring.py \
--model="$pref_model" \
--num_threads 8 \
--model_type "classifier" \
--batch_size=8 \
--input_path $input_data \
--save_dir test_output \
--debug True
echo $!
