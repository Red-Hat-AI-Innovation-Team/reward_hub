import json
import os

def read_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for row in file:
            data.append(json.loads(row))
    return data


def read_jsonl_input(path):
    data = []
    with open(path, 'r') as file:

        data = json.load(file)
    return data


def save_as_jsonl(data, filename):
    """
    Saves a list of dictionaries to a JSONL file.

    Args:
    data (list of dict): Data to save, where each dictionary in the list represents a separate JSON object.
    filename (str): Name of the file to save the data to.

    Returns:
    None
    """
    # Extract the directory from the full file path
    directory = os.path.dirname(filename)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'w', encoding='utf-8') as file:
        for entry in data:
            json_string = json.dumps(entry)
            file.write(json_string + '\n')



import random

template = "./merlinite-model-answer3/merlinite-7b-0.jsonl"

def post_process_mt_file(best_of_n, model_name, random_flag=False):
    input_file = f"./mt_bench-distribute/{model_name}/best_of_{best_of_n}_distribute_shard_0.jsonl-rewards.jsonl"

    input_data = read_jsonl(input_file)
    output_data = read_jsonl(template)
    
    new_output = []
    for idx,ex in enumerate(output_data):
        new_ex = ex.copy()
        sec_idx = idx + len(output_data)
        if random_flag:

            t1 = random.choice(input_data[idx]["output"])
            t2 = random.choice(input_data[sec_idx]["output"]),
            new_turns = [t1, t2]
        else:
            new_turns = [
                input_data[idx]["best_of_n_sample"],
                input_data[sec_idx]["best_of_n_sample"],
            ]
        new_ex["choices"] = [{
            "turns": new_turns,
        }]
        new_output.append(new_ex)
    
    model_concise = model_name.split("/")[-1]
    if random_flag:
        save_as_jsonl(new_output, f"..mt_results/random_of_{best_of_n}_granite.jsonl")
    else:
        save_as_jsonl(new_output, f"./mt_results/best_of_{best_of_n}_granite.jsonl")
    print("Successfully saved prepared output to ", f"./mt_results/best_of_{best_of_n}_granite.jsonl")
        


model_name = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

for best_of_n in [32, 64, 128]:
    
    if best_of_n == 0:
        random_flag=True 
        best_of_n = 128
    else:
        random_flag=False
    post_process_mt_file(best_of_n, model_name, random_flag=random_flag)
    breakpoint()

