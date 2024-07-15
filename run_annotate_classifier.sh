# Define the reward comparison models
pref_model="RLHFlow/ArmoRM-Llama3-8B-v0.1"

input_data=$1 # get input data

mkdir -p logs

echo "Input Source is: $input_data"


# Run bon scoring
python scripts/run_bon_scoring.py \
--model="$pref_model" \
--model_type "classifier" \
--num_threads 8 \
--batch_size 8 \
--input_path $input_data
echo $!

