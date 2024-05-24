
# Define the reward comparison models
pref_model="mistralai/Mixtral-8x7B-Instruct-v0.1"
ref_model="mistralai/Mixtral-8x7B-v0.1"

input_data=$1 # get input data

mkdir -p logs

echo "Input Source is: $input_data"

bash launch_reward_server.sh

for bestn in 64; do
    filename_with_extension=$(basename "$input_data")
    # Remove the .jsonl extension
    filename="${filename_with_extension%.jsonl}"

    # Run bon scoring
    python scripts/run_bon_scoring.py \
    --model="$pref_model" \
    --ref_model="$ref_model" \
    --num_threads 1 \
    --base_port 8020 \
    --batch_size=4 \
    --pref_sets $input_data > "logs/$filename.log" 2>&1
    echo $!
done
