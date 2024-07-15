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


question_file = "./questions.jsonl"


def format_single_turn(turns, chat_template="merlinite"):
    if chat_template == "merlinite":
        sys_prompt = "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
        turn_1 = turns[0]
        prompt = f'<|system|>\n{sys_prompt}\n<|user|>\n{turn_1}\n<|assistant|>\n'
        return prompt

    raise Exception(f"{chat_template} is not implemented")



def format_two_turns(turns, turn1_answer, chat_template="merlinite"):
    if chat_template == "merlinite":
        sys_prompt = "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
        turn_1, turn_2 = turns[0], turns[1]
        prompt = f'<|system|>\n{sys_prompt}\n<|user|>\n{turn_1}\n<|assistant|>\n{turn1_answer}\n<|user|>\n{turn_2}\n<|assistant|>\n'
        return prompt

    raise Exception(f"{chat_template} is not implemented")


def load_question_file_turn1(question_file):
    question_data = read_jsonl(question_file)
    input_data = [
        {
            "targets": "",
            "dataset":  "mt-bench",
            "group": ["single-turn"],
            "prompt": format_single_turn(ex["turns"]),
            "formatted_input": format_single_turn(ex["turns"]),
        }
        for ex in question_data
    ]
    return input_data


def load_question_file_turn2(question_file, response_file):
    
    question_data = read_jsonl(question_file)
    qids = [ex['question_id'] for ex in question_data]
    response_data = read_jsonl(response_file)
    response_data = [ex for ex in response_data if ex["question_id"] in qids]

    turn1_responses = [ex["choices"][0]["turns"][0] for ex in response_data]
    input_data = [
        {
            "targets": "none",
            "dataset":  "mt-bench",
            "group": ["second-turn"],
            "prompt": format_two_turns(ex["turns"], turn1_responses[idx]),
            "formatted_input": format_two_turns(ex["turns"], turn1_responses[idx]),
        }
        for idx, ex in enumerate(question_data)
    ]
    return input_data

def load_best_of_n_mt_bench(question_file, response_file):
    t1 = load_question_file_turn1(question_file)
    t2 = load_question_file_turn2(question_file, response_file)
    
    return t1+t2


if __name__ == "__main__":
    list_dict_data = load_best_of_n_mt_bench("questions.jsonl", "granite_model_answer/merlinite-granite-7b-lab-4.jsonl")
    breakpoint()