import json 
import os 


def read_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for row in file:
            
            data.append(json.loads(row))
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



qids = [ex['question_id'] for ex in read_jsonl("questions.jsonl") if ex['category'] in ["roleplay", "writing", "reasoning"]]
for i in range(5):
    filename = f"granite_model_answer/merlinite-granite-7b-lab-{i}.jsonl"
    data = read_jsonl(filename)
    data = [ex for ex in data if ex['question_id'] in qids]
    save_as_jsonl(data, f"granite_model_answer3/merlinite-granite-7b-lab-{i}.jsonl")
