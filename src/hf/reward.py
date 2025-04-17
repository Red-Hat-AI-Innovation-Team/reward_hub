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
from reward_hub.src.base import AbstractOutcomeRewardModel, AbstractProcessRewardModel, PRMResult


# Design Issues:
# Error Handling: The code has minimal error handling, especially for API calls to models that might fail.
# Hard-coded Model Behavior: Each model's specific behavior is hard-coded in conditional blocks, making it difficult to add new models.
# Inconsistent Parameter Handling: Some parameters like max_token_length are passed to some models but not others.
# Limited Abstraction: The base classes AbstractOutcomeRewardModel and AbstractProcessRewardModel aren't leveraged effectively to standardize interfaces.
# Batch Processing: Only implemented for one model (Qwen2.5-Math-PRM-7B) but not consistently across all models.


class HuggingFaceOutcomeRM(AbstractOutcomeRewardModel):
    def __init__(self, model_name: str, device_map: str, **kwargs):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        device_map=device_map, 
                        trust_remote_code=True
                    ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if model_name == "Qwen/Qwen2.5-Math-RM-72B":
            self.tokenizer.truncation_side = "left"

    def score(self, question: str, responses: List[str], max_token_length: int = 8192) -> List[float]:
        all_scores = []
        if self.model_name == "internlm/internlm2-7b-reward":
            for ans in responses:
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ans},
                ]
                reward_score = self.model.get_score(self.tokenizer, messages)
                all_scores.append(reward_score)

        elif self.model_name == "Qwen/Qwen2.5-Math-RM-72B":
            for ans in responses:
                QWEN_PRM_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
                message = [
                    {"role": "system", "content": QWEN_PRM_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ans},
                ]
                conversation_str = self.tokenizer.apply_chat_template(
                    message, 
                    tokenize=False, 
                    add_generation_prompt=False
                )

                input_ids = self.tokenizer(
                    conversation_str, 
                    return_tensors="pt",
                    add_special_tokens=False, 
                    truncation=True, 
                    max_length=max_token_length
                ).input_ids.to(self.model.device)
                raw_outputs = self.model(input_ids=input_ids)
                reward_score = raw_outputs[0].item()
                all_scores.append(reward_score)

        elif self.model_name == "RLHFlow/ArmoRM-Llama3-8B-v0.1":
            for ans in responses:
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ans},
                ]
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_token_length,
                ).to(self.model.device)
                with torch.no_grad():
                    output = self.model(input_ids)
                    reward_score = output.score.float().item()
                    all_scores.append(reward_score)
        else:
            raise ValueError(f"Model {self.model_name} is not supported")
        
        return all_scores


class HuggingFaceProcessRM(AbstractProcessRewardModel):
    def __init__(self, model_name: str, device_map: str, **kwargs):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        device_map=device_map, 
                        trust_remote_code=True
                    ).eval() 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if model_name == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            plus_tag_id = self.tokenizer.encode("+")[-1]
            minus_tag_id = self.tokenizer.encode("-")[-1]
            self.candidate_tokens = [plus_tag_id, minus_tag_id]
        elif model_name == "Qwen/Qwen2.5-Math-PRM-7B":
            self.tokenizer.truncation_side = "left"


    def score(self, question: str, responses: List[str], step_separator: str = "\n\n", aggregate_method: str = None, return_full_prm_result: bool = False, batch_size: int = 4, max_token_length: int = 8192) -> List[Union[PRMResult, float]]:
        """
        if return_full_prm_result is True, return the PRMResult object.     
        if return_full_prm_result is False, return the score.
        """
        if aggregate_method not in ["prod", "min", "last", "model_aggregate"]:
            raise ValueError(f"Invalid aggregate method: {aggregate_method}")
        all_scores = []
        if self.model_name == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":

            for ans in responses:
                step_scores = []
                conversation = []
                if aggregate_method == "model_aggregate":
                    ans_list = [ans]
                else:
                    ans_list = ans.split(step_separator)

                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})
                    input_ids = self.tokenizer.apply_chat_template(
                        conversation, return_tensors="pt"
                    ).to(self.model.device)
                    with torch.no_grad():
                        logits = self.model(input_ids).logits[
                            :, -3, self.candidate_tokens
                        ]  # simple version, the +/- is predicted by the '-3' position
                        single_score = logits.softmax(dim=-1)[
                            :, 0
                        ]  # 0 means the prob of + (1 mean -)
                        step_scores.append(
                            single_score[0]
                            .detach()
                            .to("cpu", dtype=torch.float32)
                            .item()
                        )
                all_scores.append(step_scores)
        
        elif self.model_name == "Qwen/Qwen2.5-Math-PRM-7B":
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
                ] # 0.88671875

                # Prepare conversation for scoring
                conversation = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                formatted_convs.append(conversation)

            # TODO: tokenize each batch independently so there is less padding and more memory efficient
            all_scores = []
            for i in range(0, len(formatted_convs), batch_size):
                batch = formatted_convs[i:i + batch_size]
                batch_input_ids = self.tokenizer(
                    batch,
                    return_tensors="pt", 
                    truncation=True,
                    padding=True,
                    max_length=max_token_length
                ).input_ids
                batch_decoded = self.tokenizer.batch_decode(batch_input_ids, skip_special_tokens=False)
                batch_outputs = self.model.encode(batch_decoded)
                batch_scores = [[d[-1].item() for d in ex.outputs.data] for ex in batch_outputs]
                all_scores.extend(batch_scores)

        else:
            raise ValueError(f"Model {self.model_name} is not supported")
        
        if return_full_prm_result:
            return [PRMResult(scores=scores, aggregate_method=aggregate_method) for scores in all_scores]
        else:
            return [PRMResult(scores=scores, aggregate_method=aggregate_method).score for scores in all_scores]
