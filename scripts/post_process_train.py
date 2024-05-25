import json
from utils import convert_to_json_format, save_as_jsonl, save_as_hf_dataset
import glob
import os
import random
import statistics


raw_data = "/dccstor/gxamr/linux-386/llm-alignment/preference-generator/uniform_sample_dataset_10k_best_of_64/bon_sampling_data_split_0.jsonl-rewards.jsonl"
raw_data = "/new_data/gx/synthetic_preference/uniform_sample_dataset_30k_best_of_64/mistralai/Mixtral-8x7B-Instruct-v0.1/bon_sampling_data_split_0.jsonl-rewards.jsonl"
raw_data1 = "/new_data/gx/synthetic_preference/uniform_sample_dataset_30k_best_of_64/mistralai/Mixtral-8x7B-Instruct-v0.1/bon_sampling_data_split_1.jsonl-rewards.jsonl"

raw_data_dir = "/new_data/gx/synthetic_preference/merlinite_RL_batches/uniform_sample_batch0-distribute/"

def read_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for row in file:
            data.append(json.loads(row))
    return data

def read_jsonl_dir(data_dir):
    data = []
    for filename in os.listdir(data_dir):
        if filename.startswith('best_of_64_distribute_shard') and filename.endswith('.jsonl-rewards.jsonl'):
            path = os.path.join(data_dir, filename)
            data.extend(read_jsonl(path))
    return data


def load_and_format_rs(data_dir, model_name="/mistralai/Mixtral-8x7B-Instruct-v0.1"):
    """
    Raw data formats:
    dict_keys(['target_output', 'dataset', 'group', 'output', 'truncated_prompt', 'prompt', 'decoder_name_or_path', 'sample_mode', 'best_of_n_sample', 'target_is_bestn', 'output_reward_scores'])

    Args:
        data_path (_type_): _description_
    """
    annotations = read_jsonl_dir(os.join(data_dir,model_name) )

    overlap = sum([ex["target_is_bestn"] for ex in annotations])
    # I should filter one words solutions/or is it math that I'm more concerned with. 
    print("overlap rate between best-of-64 and Mixtral output: ", overlap/len(annotations))

    
    best_sample_ls, best_merlinite_sample_ls = [], []
    for instance in annotations:
        msg = convert_to_json_format(instance["prompt"], append_system=True)
        msg_prompt = msg[-1]["content"]
        assert "<|system|>" not in msg_prompt
        assert "<|user|>" not in msg_prompt
        assert "<|assistant|>" not in msg_prompt

        best_text_msg = msg + [{
            "content": instance["best_of_n_sample"],
            "role": "assistant"
        }]


        # get non-teacher max-key
        if instance["target_is_bestn"]:
            import heapq
            lst = instance["output_reward_scores"]
            second_max = heapq.nlargest(2, range(len(lst)), key=lst.__getitem__)[-1]
            # The second highest value
            non_teacher_max = instance["output"][second_max]

        else:
            non_teacher_max = instance["best_of_n_sample"]

        # dataset_teacher_vs_sample
        best_sampled_text_msg =  msg + [{
            "content": non_teacher_max,
            "role": "assistant"
        }]

        best_sample = {
            "messages": best_text_msg,
            "group":"rejection-sampling",
            "dataset":"reward:Mixtral log-ratio",
            "metadata":"{\"num_turns\": 1}"
        }
        
        best_merlinite_sample = {
            "messages": best_sampled_text_msg,
            "group":"rejection-sampling",
            "dataset":"reward:Mixtral log-ratio",
            "metadata":"{\"num_turns\": 1}"
        }
        best_sample_ls.append(best_sample)
        best_merlinite_sample_ls.append(best_merlinite_sample)
    
    print_RS_stats(best_sample_ls, "best_abs_ls")
    
    print_RS_stats(best_merlinite_sample_ls, "best_model_ls")
    
    return best_sample_ls, best_merlinite_sample_ls


def sample_random(out_list, best_text, best_text2=None):
    if len(out_list) ==1: # prevent infinite looping
        return out_list[0]
    sampled = random.choice(out_list)
    if sampled == best_text or sampled==best_text2:
        # throw anotehr sample if duplicate
        sampled = random.choice(out_list)
    return sampled
        

def get_statistics(str_ls):
    """Print the average, median, and standard deviation of word counts in the list of strings.

    Args:
        str_ls (list of str): List of strings from which to calculate statistics.
    
    Prints:
        Average word count, median word count, and standard deviation of word counts.
    """
    # Calculate word counts for each string in the list
    word_counts = [len(string.split()) for string in str_ls]
    
    # Calculate average word count
    avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0
    
    # Calculate median word count
    median_word_count = statistics.median(word_counts) if word_counts else 0
    
    # Calculate standard deviation of word counts
    std_dev_word_count = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
    
    # Print the results
    print(f"Total Number of Instancest: ", len(str_ls))
    print(f"Average word count: {avg_word_count}")
    print(f"Median word count: {median_word_count}")
    print(f"Standard deviation of word counts: {std_dev_word_count}")


def print_DPO_stats(sample_ls, ds_name):
    
    print("#"*10 + " "*3 + ds_name +  " "*3 + "#"*10)
    print("#"*10 + " "*3 + "chosen split" +  " "*3 + "#"*10)
    
    get_statistics([ex['chosen'][-1]['content'] for ex in sample_ls])
    
    
    print("#"*10 + " "*3 + "rejected split" +  " "*3 + "#"*10)
    get_statistics([ex['rejected'][-1]['content'] for ex in sample_ls])


def print_RS_stats(sample_ls, ds_name):
    
    print("#"*10 + " "*3 + ds_name +  " "*3 + "#"*10)
    
    get_statistics([ex['messages'][-1]['content'] for ex in sample_ls])



def load_and_format_dpo(data_dir, model_name="/mistralai/Mixtral-8x7B-Instruct-v0.1"):
    
    """
    Raw data formats:
    dict_keys(['target_output', 'dataset', 'group', 'output', 'truncated_prompt', 'prompt', 'decoder_name_or_path', 'sample_mode', 'best_of_n_sample', 'target_is_bestn', 'output_reward_scores'])

    Args:
        data_path (_type_): _description_
        
    Return:
    DPO needs a mix of different combination pairs, for each prompt, and its corresponding samplings. Here are the combinations we sample for each prompt:
    1. following the ultrafeedback-binarized method: => I think this will do better in general, since it rewards the optimal (within mistral/mixtral) behavior. 
    Best vs a random sample: "To create UltraFeedback Binarized, we picked the highest overall_score as the "chosen" completion, and one of the remaining 3 at random as the "rejected" one."
    
    2. meta mentions that it includes gold-targets into the RS data, which improves performance for them.  => I think this will do better in squeezing variance. 
    But this model has seen the SFT data, and it may not be ideal to further sample from it, since model should assign high likelihood for it.
    Creates a DPO data version that don't include any SFT seen response, purely model sampling best/worst. 
    
    If the goal is to reduce variance, this will do!. 
        
    """
    
    annotations = read_jsonl_dir(os.join(data_dir,model_name) )
    
    
    overlap = sum([ex["target_is_bestn"] for ex in annotations])
    # I should filter one words solutions/or is it math that I'm more concerned with. 
    print("overlap rate between best-of-64 and Mixtral output: ", overlap/len(annotations))

    
    best_sample_ls, best_merlinite_sample_ls = [], []
    for instance in annotations:
        msg = convert_to_json_format(instance["prompt"])
        msg_prompt = msg[-1]["content"]
        assert "<|system|>" not in msg_prompt
        assert "<|user|>" not in msg_prompt
        assert "<|assistant|>" not in msg_prompt

        best_text_msg = msg + [{
            "content": instance["best_of_n_sample"],
            "role": "assistant"
        }]

        # get non-teacher max-key
        if instance["target_is_bestn"]:
            import heapq
            lst = instance["output_reward_scores"]
            second_max = heapq.nlargest(2, range(len(lst)), key=lst.__getitem__)[-1]
            # The second highest value
            non_teacher_max = instance["output"][second_max]

        else:
            non_teacher_max = instance["best_of_n_sample"]

        # dataset_teacher_vs_sample
        best_sampled_text_msg =  msg + [{
            "content": non_teacher_max,
            "role": "assistant"
        }]

        # curate DPO to punish length
        random_text1 = sample_random(instance["output"], instance['best_of_n_sample']) # don't sample the max
        random_text2 = sample_random(instance["output"], instance['best_of_n_sample'], non_teacher_max) # don't sample the max or second max
        
        best_sample = {
            "prompt": msg_prompt,
            "messages": msg,
            "chosen": best_text_msg,
            "rejected": msg + [{
                "content": random_text1,
                "role": "assistant"
            }]
        }

        best_merlinite_sample = {
            "prompt": msg_prompt,
            "messages": msg,
            "chosen": best_sampled_text_msg,
            "rejected": msg + [{
                "content": random_text2,
                "role": "assistant"
            }]
        }
        if best_sample['chosen'][-1]['content'] != best_sample['rejected'][-1]['content']:
            best_sample_ls.append(best_sample)
        
        if best_merlinite_sample['chosen'][-1]['content'] != best_merlinite_sample['rejected'][-1]['content']:
            best_merlinite_sample_ls.append(best_merlinite_sample)
    
    return best_sample_ls, best_merlinite_sample_ls


"""
{"group":"chat-multiturn","dataset":"chatbot_arena","messages":[{"content":"You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.","role":"system"},{"content":"What is the history of the Cavoodle?","role":"user"},{"content":"The Cavoodle, also known as the Cavapoo, is a crossbreed that originated in the 1990s, primarily in Australia, by combining the desirable traits of two popular breeds: the Poodle and the Cavalier King Charles Spaniel. The primary objective was to create a small, friendly, and intelligent companion dog with low-shedding and hypoallergenic qualities.\n\nThe Poodle is known for its intelligence and low-shedding coat, while the Cavalier King Charles Spaniel is appreciated for its friendly and affectionate nature. The Cavoodle has inherited these traits, making it a popular choice for families and individuals seeking a companion dog.\n\nThe Cavoodle's popularity has grown beyond Australia,and it is now found in various countries around the world. As a crossbreed, the Cavoodle does not have a standardized appearance or temperament, and individual dogs may vary in size, coat type, and personality. However, the overall trend is a small, friendly, and intelligent dog that is well-suited for various living situations and family dynamics.\n\nTo learn more about the Cavoodle and its parent breeds, you can consult reputable breeders, dog clubs, or animal welfare organizations. They can provide you with more detailed information on the breed's history, care requirements, and temperament, as well as help you make an informed decision if you are considering adding a Cavoodle to your family.\n\nPlease note that the information I provided is based on my knowledge up to the time of my last update, and the world of crossbreeds is continuously evolving. It's always a good idea to check the latest information from reliable sources for the most current details.","role":"assistant"}],"metadata":"{\"num_turns\": 1}"}
"""


if __name__ == "__main__":
    
    # ## Here it saves data for Rejection Sampling
    best_sample_ls, best_merlinite_sample_ls = load_and_format_rs(raw_data_dir)
    breakpoint()
    
    prefix = raw_data_dir
    # prefix = "./may15_30k_batch/"
    save_as_hf_dataset(best_sample_ls, f"{prefix}/abs_best_rs")
    save_as_jsonl(best_sample_ls, f"{prefix}/abs_best_rs.jsonl")
    breakpoint()
    save_as_hf_dataset(best_merlinite_sample_ls, f"{prefix}/model_best_rs")
    save_as_jsonl(best_merlinite_sample_ls, f"{prefix}/model_best_rs.jsonl")


    ## Here it saves data for DPO training
    
    # best_sample_ls, best_merlinite_sample_ls = load_and_format_dpo(raw_data,raw_data1)
    # breakpoint()
    # print_DPO_stats(best_sample_ls, "best_sample_ls")
    
    
    # print_DPO_stats(best_merlinite_sample_ls, "best_merlinite_sample_ls")
    
    # breakpoint()
    # prefix = "/new_data/gx/synthetic_preference/uniform_sample_dataset_30k_best_of_64/may15_30k_batch/"
    # # prefix = "./may15_30k_batch/"
    # save_as_hf_dataset(best_sample_ls, f"{prefix}/abs_best_dpo")
    # save_as_jsonl(best_sample_ls, f"{prefix}/abs_best_dpo.jsonl")
    # save_as_hf_dataset(best_merlinite_sample_ls, f"{prefix}/model_best_dpo")
    # save_as_jsonl(best_merlinite_sample_ls, f"{prefix}/model_best_dpo.jsonl")
