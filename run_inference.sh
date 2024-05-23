
# Define the sampling model
model_engine="/dccstor/creme_brulee/aimodel_factory/instructlab/merlinite-deep-serenity-19_187264"

# Define the reward comparison models
pref_model="mistralai/Mixtral-8x7B-Instruct-v0.1"
ref_model="mistralai/Mixtral-8x7B-v0.1"

# Define the input dataset path
# input_data="/new_data/gx/synthetic_preference/uniform_sample_dataset_30k_best_of_64/bon_sampling_data_split_0.jsonl"
input_data="/dccstor/gxamr/linux-386/llm-alignment/merlinite_RL_batches/uniform_sample_batch0.jsonl"


bash launch_sampling_server.sh $model_engine


for bestn in 64; do
    output_dir=$(dirname "$input_data")
    mkdir -p "$output_dir"  # Ensures the directory exists
    
    # Get the filename with extension
    filename_with_extension=$(basename "$input_data")

    # Remove the .jsonl extension
    filename="${filename_with_extension%.jsonl}"

    # Launch repeat_n_sampling
    python scripts/repeat_n_sampling.py --decoder_name_or_path "$model_engine" --base_port 8020 --output_path "$output_dir/bon_sampling_data_split_0.jsonl" --dataset_path="$input_data" --num_return_sequences "$bestn" > "logs/pref_30k_64_sampling_batch0.log" 2>&1 &
    echo $!
    
    # Run bon scoring
    # python scripts/run_bon_scoring.py --model="$pref_model" --ref_model="$ref_model" --num_threads 2 --batch_size=4 --debug True --pref_sets $input_data > "logs/$filename.log" 2>&1 &
    # echo $!
done
