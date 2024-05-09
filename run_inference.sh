# Define the sampling model
model_engine="ibm/merlinite-7b"


# Define the reward comparison models
pref_model="mistralai/Mixtral-8x7B-Instruct-v0.1"
ref_model="mistralai/Mixtral-8x7B-v0.1"

# Define the input dataset path
input_data="mt_bench"


for bestn in 32 64 128; do
# for bestn in 32; do
    output_dir="mt_bench_results/best_of_${bestn}"
    mkdir -p "$output_dir"  # Ensures the directory exists
    # launch repeat_n_sampling
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/repeat_n_sampling.py --decoder_name_or_path $model_engine --base_port 8000 --output_path $output_dir/bon_sampling_data.jsonl --dataset_path=$input_data --num_return_sequences $bestn > logs/repeat_n_sampling.log 2>&1
    python scripts/run_bon_scoring.py --model="$pref_model" --ref_model="$ref_model" --batch_size=4 --pref_sets "$output_dir/bon_sampling_data.jsonl"
done

    # python scripts/run_bon_scoring.py --model="$pref_model" --ref_model="$ref_model" --batch_size=4 --pref_sets "$output_dir/bon_sampling_data.jsonl"






