
# Define the reward comparison models
pref_model="RLHFlow/ArmoRM-Llama3-8B-v0.1"

mkdir -p logs


for bestn in 32 64 128; do
    input_data="mt_bench-distribute/best_of_${bestn}_distribute_shard_0.jsonl"
    # Run bon scoring
    python scripts/run_bon_scoring_rm.py \
    --model="$pref_model" \
    --num_threads 1 \
    --base_port 8020 \
    --batch_size=8 \
    --pref_sets $input_data
    echo $!
done
