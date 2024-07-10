import random
from typing import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

class ArmoRMPipeline:
    def __init__(self, model_name, device=0):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map=device, 
                                    trust_remote_code=True, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = model
        self.tokenizer = tokenizer
        random.seed(0)
        
    def __call__(self, candidates: List[str], **kwargs):
        """
        samples: List[str]
        """
        device = self.model.device
        scores, multi_obj_rewards = [], []
        with torch.no_grad():
            for candidate in tqdm(candidates):
                input_ids = self.tokenizer.apply_chat_template(candidate, return_tensors="pt").to(device)
                output = self.model(input_ids)
                score = output.score.cpu().float().item()
                scores.append(score)
                multi_obj_reward = output.rewards.cpu().float() 
                multi_obj_rewards.append(multi_obj_reward)

        return scores, multi_obj_rewards

if __name__ == "__main__":
    prompt = 'What are some synonyms for the word "beautiful"?'
    response = "Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant"
    messages = [[{"role": "user", "content": prompt},
            {"role": "assistant", "content": response}]]
    
    device = 0
    path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

    armopipeline = ArmoRMPipeline(path, device=device)
    scores, multi_obj_rewards = armopipeline(messages)
    breakpoint()
    