import json
from utils import convert_to_json_format, save_as_jsonl, save_as_hf_dataset
import glob
import os

MERLINITE_SYSTEM= "<|system|>\nYou are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
simular_prompt = "<|system|> You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."


ASSSTANT = "\n<|assistant|>\n"
USER = "\n<|user|>\n"
data_path = "./Mixtral-8x7B-Instruct-v0.1.json"
def strip_prompt(text):
    text = text.replace(MERLINITE_SYSTEM, "")
    text = text.replace(simular_prompt, "")
    text = text.replace(ASSSTANT, "").replace(ASSSTANT.strip(), "")
    text = text.replace(USER, "").replace(USER.strip(), "")
    return text.strip()



def read_jsonl(path):
    data = []
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def read_jsonl1(path):
    with open(path, 'r') as file:
        data = json.load(file)
    ds = []
    for idx in range(len(data['text_chosen'])):
        instance = {
            "prompt": data["prompt"][idx],
            "text_chosen": data["text_chosen"][idx],
            "scores_chosen": data["scores_chosen"][idx],
            "text_rejected": data["text_rejected"][idx],
            "scores_rejected": data["scores_rejected"][idx],
        }
        ds.append(instance)
    
    return ds


REWARD_WRONG_DT = []

def load_and_format_pref(data_path):
    annotations = read_jsonl(data_path)

    distances = []

    dataset_teacher_vs_sample = []

    dataset_combination_pairs = []
    
    dataset_best_worst_pairs = []

    discrepency = 0
    
    teacher_prefer_counts = 0

    
    # 1. confirm every 6 samples can be parsed into 1.
    # Optional
    for i in range(len(annotations["prompt"])//3):
        start_idx = i*3
        end_idx = start_idx + 3
        prompts = annotations["prompt"][start_idx:end_idx]
        
        for t in range(1, 3):
            if t< len(prompts):
                assert prompts[t]== prompts[t-1]

        score_map = {}
        teacher_response = None
        for idx in range(start_idx, end_idx):
            text1 = annotations["text_chosen"][idx]
            text1_s = annotations["scores_chosen"][idx]
            
            text2 = annotations["text_rejected"][idx]
            text2_s = annotations["scores_rejected"][idx]
            
            if text1 not in score_map:
                score_map[text1] = text1_s
            else:
                try:
                    assert score_map[text1]-10 < text1_s < score_map[text1]+10
                    score_map[text1] = min(text1_s, score_map[text1])
                except:
                    discrepency+=1
                    score_map[text1] = min(text1_s, score_map[text1])

            if text2 not in score_map:
                score_map[text2] = text2_s
            else:
                try:
                    assert score_map[text2]-10 < text2_s < score_map[text2]+10
                except:
                    discrepency+=1
            
            if idx == end_idx-1:
                # this is basically the Mixtral-teacher response
                teacher_response, teacher_score = text1, text1_s
        if len(score_map) < 4:
            continue
        max_key = max(score_map, key=score_map.get)
        min_key = min(score_map, key=score_map.get)
        distance = score_map[max_key] - score_map[min_key]
        distances.append(distance)

        msg = [{"content": strip_prompt(prompts[0]), "role": "user"}]
        msg_prompt = msg[-1]["content"]
        assert "<|system|>" not in msg_prompt
        assert "<|user|>" not in msg_prompt
        assert "<|assistant|>" not in msg_prompt


        
        # get non-teacher max-key
        if teacher_response == max_key:
            teacher_prefer_counts += 1
            sorted_items = sorted(score_map.items(), key=lambda item: item[1])
            # The second highest value
            non_teacher_max = sorted_items[-2][0]

        else:
            non_teacher_max = max_key

        teacher_text_msg = msg + [{
            "content": teacher_response,
            "role": "assistant"
        }]
        
        # dataset_teacher_vs_sample
        best_sampled_text_msg =  msg + [{
            "content": non_teacher_max,
            "role": "assistant"
        }]

        teacher_vs_sample = {
            "prompt": msg_prompt,
            "messages": msg,
            "chosen": teacher_text_msg,
            "rejected": best_sampled_text_msg
        }
        dataset_teacher_vs_sample.append(teacher_vs_sample)

        if teacher_response != max_key:
            REWARD_WRONG_DT.append(teacher_vs_sample)

        max_score_msg = msg + [{
            "content": max_key,
            "role": "assistant"
        }]
        least_score_msg = msg + [{
            "content": min_key,
            "role": "assistant"
        }]
        
        # best-worse-pairs
        best_worst_pair = {
            "prompt": msg_prompt,
            "messages": msg,
            "chosen": max_score_msg,
            "rejected": least_score_msg
        }
        dataset_best_worst_pairs.append(best_worst_pair)


        # combination_pairs;
        # I would say include all combinatorial pairs that have difference greater than 5
        try:
            assert len(score_map) <= 5
        except:
            breakpoint()
        from itertools import combinations
        combos = list(combinations(list(score_map.keys()), 2))
        for combo in combos:
            key1, key2 = combo
            if abs(score_map[key1] - score_map[key2]) < 5:
                # skip it if difference is within margins of error
                continue 
            
            if score_map[key1] > score_map[key2]:
                max_key, min_key = key1, key2
            else:
                max_key, min_key = key2, key1

            chosen_text_msg = msg + [{
                "content": max_key,
                "role": "assistant"
            }]
            
            rejected_text_msg = msg + [{
                "content": min_key,
                "role": "assistant"
            }]

            combo_pairs_instance = {
                "prompt": msg_prompt,
                "messages": msg,
                "chosen": chosen_text_msg,
                "rejected": rejected_text_msg,
                "score_chosen": score_map[max_key],
                "score_rejected": score_map[min_key]
            }

            dataset_combination_pairs.append(combo_pairs_instance)

    print("Accuracy of Mixtral reward to prefer Mixtral outputs: ", teacher_prefer_counts/len(dataset_teacher_vs_sample))

    return dataset_teacher_vs_sample, dataset_combination_pairs, dataset_best_worst_pairs


raw_dir = "./pref_annotations"

final_dataset_teacher_vs_sample, final_dataset_combination_pairs, final_dataset_best_worst_pairs = [], [], []
for raw_file in glob.glob(os.path.join(raw_dir, '*.jsonl')):
    dataset_teacher_vs_sample, dataset_combination_pairs, dataset_best_worst_pairs = load_and_format_pref(raw_file)
    final_dataset_teacher_vs_sample.extend(dataset_teacher_vs_sample)
    final_dataset_combination_pairs.extend(dataset_combination_pairs)
    final_dataset_best_worst_pairs.extend(dataset_best_worst_pairs)


breakpoint()
import random
random.shuffle(dataset_best_worst_pairs)

for i in range(10):
    save_as_jsonl(dataset_best_worst_pairs[i*10:i*10+10], f"eyeball_samples/teacher_merlinite_sample_dpo_pairs_dataset/samples_{i}.jsonl")

random.shuffle(REWARD_WRONG_DT)

for i in range(10):
    save_as_jsonl(REWARD_WRONG_DT[i*10:i*10+10], f"eyeball_samples/mixtral_reward_wrong_cases/samples_{i}.jsonl")


breakpoint()
breakpoint()
save_as_hf_dataset(final_dataset_best_worst_pairs, "./merlinite_dataset_best_worst_pairs")
save_as_jsonl(final_dataset_best_worst_pairs, "./merlinite_dataset_best_worst_pairs.jsonl")

breakpoint()
# it would be interesting to closely look at where Reward Model and Teacher response don't agree. 


# save_as_hf_dataset(final_dataset_combination_pairs, "./merlinite_dataset_combination_pairs")
# save_as_jsonl(final_dataset_combination_pairs, "./merlinite_dataset_combination_pairs.jsonl")

