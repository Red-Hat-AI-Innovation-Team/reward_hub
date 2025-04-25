# Copyright 2025 GX Xu (gxxu@redhat.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel
)
from typing import Union, List
from reward_hub.base import AbstractOutcomeRewardModel, AbstractProcessRewardModel, PRMResult
import os
from vllm import LLM


class VLLMOutcomeRM(AbstractOutcomeRewardModel):
    def __init__(self, model_name: str, device: Union[str, int], **kwargs):
        raise NotImplementedError("VLLMOutcomeRM is not implemented")
    
    def score(self, question: str, responses: List[str], batch_size: int = 4, max_input_tokens: int = 8192) -> List[float]:
        raise NotImplementedError("VLLMOutcomeRM is not implemented")

class VLLMProcessRM(AbstractProcessRewardModel):
    def __init__(self, model_name: str, device: Union[str, int], **kwargs):
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        
        self.model = LLM(model=model_name, 
                    task="reward",
                    device=device,
                    gpu_memory_utilization=0.8,
                    tensor_parallel_size=1,
                    )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_name == "Qwen/Qwen2.5-Math-PRM-7B":
            self.tokenizer.truncation_side = "left"
        self.model_name = model_name

    def score(self, question: str, responses: List[str], step_separator: str = "\n\n", aggregate_method: str = "last", return_full_prm_result: bool = False, max_input_tokens: int = 8192) -> Union[List[PRMResult], List[float]]:
        
        if self.model_name == "Qwen/Qwen2.5-Math-PRM-7B":
            formatted_convs = []
            for ans in responses:
                if aggregate_method == "model_aggregate":
                    steps_list = [ans]    
                else:
                    steps_list = ans.split(step_separator)
                QWEN_PRM_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
                messages = [
                    {"role": "system", "content": QWEN_PRM_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "<extra_0>".join(steps_list) + "<extra_0>"},
                ]

                # Prepare conversation for scoring
                conversation = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                formatted_convs.append(conversation)

            all_input_ids = self.tokenizer(
                formatted_convs,
                return_tensors="pt", 
                truncation=True,
                padding=True,
                max_length=max_input_tokens
            ).input_ids
            batch_decoded = self.tokenizer.batch_decode(all_input_ids, skip_special_tokens=False)
            all_outputs = self.model.encode(batch_decoded)
            all_scores = [[d[-1].item() for d in ex.outputs.data] for ex in all_outputs]

        else:
            raise ValueError(f"Model {self.model_name} is not supported")
        
        if return_full_prm_result:
            return [PRMResult(scores=scores) for scores in all_scores]
        else:
            return [PRMResult(scores=scores, aggregate_method=aggregate_method).score for scores in all_scores]


if __name__ == "__main__":
    output = """To determine how much Janet makes from selling the duck eggs at the farmers' market, we need to follow these steps:\n\n1. Calculate the total number of eggs laid by the ducks each day.\n2. Determine how many eggs Janet eats and bakes for herself each day.\n3. Find out how many eggs are left to be sold.\n4. Calculate the revenue from selling the remaining eggs at $2 per egg.\n\nLet's start with the first step:\n\n1. Janet's ducks lay 16 eggs per day.\n\nNext, we calculate how many eggs Janet eats and bakes for herself each day:\n\n2. Janet eats 3 eggs for breakfast every morning.\n3. Janet bakes 4 eggs for her friends every day.\n\nSo, the total number of eggs Janet eats and bakes for herself each day is:\n\\[ 3 + 4 = 7 \\text{ eggs} \\]\n\nNow, we find out how many eggs are left to be sold:\n\\[ 16 - 7 = 9 \\text{ eggs} \\]\n\nFinally, we calculate the revenue from selling the remaining eggs at $2 per egg:\n\\[ 9 \\times 2 = 18 \\text{ dollars} \\]\n\nTherefore, Janet makes boxed18 dollars every day at the farmers' market.
    """
    model = VLLMOutcomeRM("Qwen/Qwen2.5-Math-RM-72B")
    out = model.score("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                 [output])
    breakpoint()
