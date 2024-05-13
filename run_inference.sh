
# Define the sampling model
model_engine="ibm/merlinite-7b"

# Define the reward comparison models
pref_model="mistralai/Mixtral-8x7B-Instruct-v0.1"
ref_model="mistralai/Mixtral-8x7B-v0.1"

# Define the input dataset path
input_data="/new_data/gx/synthetic_preference/uniform_sample_dataset_30k_best_of_64/bon_sampling_data_split_0.jsonl"

for bestn in 64; do
    # output_dir=$(dirname "$input_data")
    # mkdir -p "$output_dir"  # Ensures the directory exists
    
    Get the filename with extension
    filename_with_extension=$(basename "$input_data")

    # Remove the .jsonl extension
    filename="${filename_with_extension%.jsonl}"
    
    # Launch repeat_n_sampling
    # python scripts/repeat_n_sampling.py --decoder_name_or_path "$model_engine" --base_port 8000 --output_path "$output_dir/bon_sampling_data_split_1.jsonl" --dataset_path="$input_data" --num_return_sequences "$bestn" > "logs/pref_30k_64_sampling_split1.log" 2>&1 &
    # echo $!
    
    # Run bon scoring
    python scripts/run_bon_scoring.py --model="$pref_model" --ref_model="$ref_model" --num_threads 2 --batch_size=4 --debug True --pref_sets $input_data > "logs/$filename.log" 2>&1 &
    echo $!
done
