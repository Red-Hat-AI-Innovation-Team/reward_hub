# Define the reward comparison models
pref_model="mistralai/Mixtral-8x7B-Instruct-v0.1"
ref_model="mistralai/Mixtral-8x7B-v0.1"

input_data=$1 # get input data

mkdir -p logs

echo "Input Source is: $input_data"

bash launch_reward_server.sh

# Run bon scoring
python scripts/run_bon_scoring.py \
--model="$pref_model" \
--ref_model="$ref_model" \
--num_threads 1 \
--base_port 8020 \
--batch_size=4 \
--input_path $input_data
echo $!

