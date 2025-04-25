import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from typing import Union, List
from reward_hub.base import AbstractOutcomeRewardModel, AbstractProcessRewardModel, PRMResult, AggregationMethod
import torch.nn.functional as F

class HuggingFaceOutcomeRewardModel(AbstractOutcomeRewardModel):
    def __init__(self, model_name: str, device: Union[str, int], **kwargs):
        self.model_name = model_name
        if model_name == "RLHFlow/ArmoRM-Llama3-8B-v0.1":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        device_map=device, 
                        trust_remote_code=True
                    ).eval()
        else:
            self.model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        device_map=device, 
                        trust_remote_code=True
                    ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    def score(self, messages: Union[List[List[dict]], List[dict]], max_input_tokens: int = 8192) -> List[float]:
        """
        Input messages uses the OpenAi chat completion format.
        If messages is a list of list of dicts, then each list of dicts is a conversation.
        If messages is a list of dicts, then it is a single conversation.
        """
        if isinstance(messages[0], dict):
            # ensure the input is a list of list of dicts   
            messages = [messages]

        all_scores = []
        if self.model_name == "internlm/internlm2-7b-reward":
            for conv_messages in messages:
                reward_score = self.model.get_score(self.tokenizer, conv_messages)
                all_scores.append(reward_score)

        elif self.model_name == "Qwen/Qwen2.5-Math-RM-72B":
            for conv_messages in messages:
                QWEN_PRM_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
                conv_messages = [
                    {"role": "system", "content": QWEN_PRM_SYSTEM_PROMPT},
                ] + conv_messages
                conversation_str = self.tokenizer.apply_chat_template(
                    conv_messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )

                input_ids = self.tokenizer(
                    conversation_str, 
                    return_tensors="pt",
                    add_special_tokens=False, 
                    truncation=True, 
                    max_length=max_input_tokens
                ).input_ids.to(self.model.device)
                raw_outputs = self.model(input_ids=input_ids)
                reward_score = raw_outputs[0].item()
                all_scores.append(reward_score)

        elif self.model_name == "RLHFlow/ArmoRM-Llama3-8B-v0.1":
            for conv_messages in messages:
                input_ids = self.tokenizer.apply_chat_template(
                    conv_messages,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_tokens,
                ).to(self.model.device)
                with torch.no_grad():
                    output = self.model(input_ids)
                    # The preference score for the response, aggregated from the 
                    # multi-objective rewards with the gating layer
                    reward_score = output.score.cpu().float().item()
                    all_scores.append(reward_score)
        else:
            raise ValueError(f"Model {self.model_name} is not supported")
        
        return all_scores






def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


class HuggingFaceProcessRewardModel(AbstractProcessRewardModel):
    def __init__(self, model_name: str, device: Union[str, int], **kwargs):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if model_name == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device, 
                trust_remote_code=True
            ).eval()
            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            plus_tag_id = self.tokenizer.encode("+")[-1]
            minus_tag_id = self.tokenizer.encode("-")[-1]
            self.candidate_tokens = [plus_tag_id, minus_tag_id]
        elif model_name == "Qwen/Qwen2.5-Math-PRM-7B":
            self.model = AutoModel.from_pretrained(
                            model_name,
                            torch_dtype=torch.bfloat16,
                            device_map=device, 
                            trust_remote_code=True
                        ).eval() 
            self.tokenizer.truncation_side = "left"


    def score(self, messages: Union[List[List[dict]], List[dict]], step_separator: str = "\n\n", 
              aggregation_method: Union[AggregationMethod, str] = AggregationMethod.LAST, 
              return_full_prm_result: bool = False, max_input_tokens: int = 8192) -> List[Union[PRMResult, float]]:
        """
        if return_full_prm_result is True, return the PRMResult object.     
        if return_full_prm_result is False, return the score.
        """
        # Convert string to enum if needed for backward compatibility
        if isinstance(aggregation_method, str):
            aggregation_method = AggregationMethod(aggregation_method)
        if isinstance(messages[0], dict):
            messages = [messages]
        all_scores = []
        if self.model_name == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
            for conv_messages in messages:
                step_scores = []
                if aggregation_method == AggregationMethod.MODEL:
                    steps_list = [conv_messages[-1]['content']]
                else:
                    steps_list = conv_messages[-1]['content'].split(step_separator)
                conversation = conv_messages[:-1]
                for k in range(len(steps_list)):
                    if k == 0:
                        text = " " + steps_list[0]
                    else:
                        text = steps_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})
                    input_ids = self.tokenizer.apply_chat_template(
                        conversation,
                        return_tensors="pt",
                        max_length=max_input_tokens
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
            for conv_messages in messages:
                if aggregation_method == AggregationMethod.MODEL:
                    steps_list = [conv_messages[-1]['content']]    
                else:
                    steps_list = conv_messages[-1]['content'].split(step_separator)
                QWEN_PRM_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
                system_turn = [
                    {"role": "system", "content": QWEN_PRM_SYSTEM_PROMPT},
                ] # 0.88671875
                conv_messages = system_turn + conv_messages[:-1] + [{"role": "assistant", "content": "<extra_0>".join(steps_list) + "<extra_0>"},]
                # Prepare conversation for scoring
                conversation = self.tokenizer.apply_chat_template(
                    conv_messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )

                # TODO: tokenize each batch independently so there is less padding and more memory efficient

                all_input_ids = self.tokenizer.encode(
                    conversation,
                    return_tensors="pt", 
                    truncation=True,
                    padding=True,
                    max_length=max_input_tokens
                ).to(self.model.device)
                step_sep_id = self.tokenizer.encode("<extra_0>")[0]
                token_masks = (all_input_ids == step_sep_id)
                all_outputs = self.model(input_ids=all_input_ids)
                all_scores.append(make_step_rewards(all_outputs[0], token_masks)[0])
        else:
            raise ValueError(f"Model {self.model_name} is not supported")

        if return_full_prm_result:
            return [PRMResult(scores=scores, aggregation_method=aggregation_method) for scores in all_scores]
        else:
            return [PRMResult(scores=scores, aggregation_method=aggregation_method).score for scores in all_scores]
