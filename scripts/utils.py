import json


MERLINITE_SYSTEM= "<|system|>\nYou are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
ASSISTANT = "\n<|assistant|>\n"
USER = "\n<|user|>\n"

MSG_MERLINITE_SYSTEM= "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."

def convert_to_json_format(input_string, append_system=False):

    segments = []
    # Remove the system prompt at the beginning if it exists
    if input_string.startswith(MERLINITE_SYSTEM):
        input_string = input_string[len(MERLINITE_SYSTEM):]
        if append_system:
            segments.append({"content": MSG_MERLINITE_SYSTEM, "role": "system"})
    else:
        raise Exception("no system prompt found, error")

    # Split the remaining string by the user and assistant tags
    temp = input_string
    assert temp.startswith(USER)
    role = None
    while temp:
        if temp.startswith(ASSISTANT):
            role = "assistant"
            temp = temp[len(ASSISTANT):]
        elif temp.startswith(USER):
            role = "user"
            temp = temp[len(USER):]
        else:
            content_end = min(temp.find(ASSISTANT) if temp.find(ASSISTANT) != -1 else len(temp),
                              temp.find(USER) if temp.find(USER) != -1 else len(temp))
            content = temp[:content_end]
            if content.strip():  # Prevent empty content due to consecutive tags
                segments.append({"content": content.strip(), "role": role})
            temp = temp[content_end:]
    
    return segments



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
            

from datasets import Dataset, load_dataset
import random
import os

def save_as_hf_dataset(data, directory):
    """
    Splits data into a training and test dataset, then saves both using the Hugging Face datasets library.

    Parameters:
        data (list): A list of dictionaries representing the data.
        directory (str): Base directory path where the datasets will be saved.
    """
    # Transform list of dictionaries to a dictionary of lists
    transformed_data = {key: [dic[key] for dic in data if key in dic] for key in set().union(*data)}

    # Convert dictionary of lists into a Hugging Face Dataset
    dataset = Dataset.from_dict(transformed_data)
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=random.randint(0, 10000))
    
    # Split the dataset into training and test datasets
    test_size = 1000  # fixed size for the test set
    if len(dataset) > test_size:
        train_size = len(dataset) - test_size
    else:
        raise ValueError("Data size must be greater than 1000 for splitting")
    
    # Using Dataset.train_test_split to handle the split
    split_datasets = dataset.train_test_split(test_size=test_size, train_size=train_size)
    
    # Prepare directories for saving datasets
    train_dir = os.path.join(directory, "train")
    test_dir = os.path.join(directory, "test")
    
    # Saving the datasets to disk
    split_datasets['train'].save_to_disk(train_dir)
    split_datasets['test'].save_to_disk(test_dir)
    
    print(f"Datasets saved: Train dataset at {train_dir}, Test dataset at {test_dir}")



test_case1 = f"{MERLINITE_SYSTEM}{USER}user-question{USER}user-question"

test_case2 = f"{MERLINITE_SYSTEM}{ASSISTANT}assistant-question{ASSISTANT}assistant-question"

test_case3 = f"{MERLINITE_SYSTEM}{ASSISTANT}assistant-question{ASSISTANT}assistant-question{ASSISTANT}assistant-question{ASSISTANT}assistant-question{USER}user-question"

test_case4 = f"{MERLINITE_SYSTEM}{USER}user-question{USER}user-question{USER}user-question{ASSISTANT}assistant-question{ASSISTANT}assistant-question{USER}user-question{USER}user-question{USER}user-question"

if __name__ == "__main__":
    # for case in [test_case1,test_case2,test_case3,test_case4]:
    #     print(convert_to_json_format(case))
    
    test_data = [
        {"key": 100}
    ] * 100000
    
    save_as_hf_dataset(test_data, "./")
    
    
    
    
    
    
    
    
    def sample_n(rdata):
        focus_data = ["chatbot_arena", "reasoning -> math_and_logic", "poetry", 'mixtral-cot', "stem_synth_textbooks", 'comparative_analysis_qa', 'data_interpretation_qa', 'STEM -> math -> arithmetic_w_grammar', 'ibm_100_pr_training_2', 'writing', 'api_single_sgd', 'rag_cot', 'chatbot_arena', 'api_multi_snips', 'oasst2', 'ELUR_experts', 'linguistics -> instruction_following -> keyword', 'linguistics -> instruction_following -> length', 'main_point_qa', 'meeting_insights', 'STEM -> science -> geography', 'formatit', 'sequence_of_events_type_2', 'ibm_100_pr', 'rag_textbooks_qa', 'open_plat_science', 'mixtbench', 'linguistics -> instruction_following -> punctuation', 'ELIH', 'api_single_multiwoz', 'ELUR_characters', 'STEM -> math -> reasoning', 'musique', 'helpsteer', 'hhrlhf-preference', 'long_summary', 'puns', 'inference_qa', 'model-identity', 'sap_knowledge_qa', 'bias', 'domain_professional_qa', 'comparison_contrast_qa', 'short_summary', '2wikimultihop_train', 'hotpot_qa', 'mmlu_longform', 'editing', 'codeparrot_conala', 'mixtral-niv', 'pups_new', 'linguistics -> classification -> agent_classification', 'ibm_100_pr_training', 'slot_filling', 'ELIP', 'reasoning -> financial_reasoning', 'rag_chat', 'STEM -> math -> area', 'api_single_topv2', 'linguistics -> summarization -> list_of_sentences', 'specific_risk', 'linguistics -> summarization -> ignore_pii', 'api_multi_sgd', 'squad_v2', 'api_single_atis', 'api_multi_multiwoz', 'textbooks_qa', 'details_qa', 'math', 'prosocial', 'STEM -> math -> distance_conversion', 'wiki_insights', 'rag_table_gen', 'mixtbench-extraction', 'complex_qa']
        for category in focus_data:
            filtered_data = rdata.filter(lambda example: example['dataset'] == category)[:350]
        


