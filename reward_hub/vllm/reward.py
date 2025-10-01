import torch
from transformers import AutoTokenizer
from typing import Union, List
from reward_hub.base import AbstractOutcomeRewardModel, AbstractProcessRewardModel, PRMResult, AggregationMethod
import os
from vllm import LLM


class VllmOutcomeRewardModel(AbstractOutcomeRewardModel):
    def __init__(self, model_name: str, **kwargs):
        raise NotImplementedError("VLLMOutcomeRM is not implemented")
    
    def score(self, messages: Union[List[List[dict]], List[dict]], max_input_tokens: int = 8192) -> List[float]:
        raise NotImplementedError("VLLMOutcomeRM is not implemented")

class VllmProcessRewardModel(AbstractProcessRewardModel):
    def __init__(self, model_name: str, **kwargs):
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        
        self.model = LLM(model=model_name, 
                    task="reward",
                    gpu_memory_utilization=0.8,
                    tensor_parallel_size=1,
                    )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_name == "Qwen/Qwen2.5-Math-PRM-7B":
            self.tokenizer.truncation_side = "left"
        self.model_name = model_name

    def score(self, messages: Union[List[List[dict]], List[dict]], max_input_tokens: int = 8192, step_sep: str = "\n\n",
             aggregation_method: Union[AggregationMethod, str] = AggregationMethod.LAST, return_full_prm_result: bool = False, use_tqdm: bool = False) -> Union[List[PRMResult], List[float]]:
        """
        Score last turn assistant message using the OpenAI chat completion format.
        
        Args:
            messages: List of conversations in OpenAI chat completion format
            max_input_tokens: Maximum number of input tokens
            return_full_prm_result: Whether to return full PRM results
            use_tqdm: Whether to display a progress bar (using tqdm)
        """
        if isinstance(aggregation_method, str):
            aggregation_method = AggregationMethod(aggregation_method)
        if isinstance(messages[0], dict):
            # ensure the input is a list of list of dicts   
            messages = [messages]
        if self.model_name == "Qwen/Qwen2.5-Math-PRM-7B":
            formatted_convs = []
            QWEN_PRM_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
            system_turn = [{
                "role": "system",
                "content": QWEN_PRM_SYSTEM_PROMPT
            }]
            for conv_messages in messages:

                last_assistant_message = conv_messages[-1]['content']
                if aggregation_method == AggregationMethod.MODEL:
                    steps_list = [last_assistant_message]
                else:
                    steps_list = last_assistant_message.split(step_sep)
                formatted_last_assistant_turn = [
                    {"role": "assistant", "content": "<extra_0>".join(steps_list) + "<extra_0>"}
                ]
                prepared_messages = system_turn + conv_messages[:-1] + formatted_last_assistant_turn
                conversation = self.tokenizer.apply_chat_template(
                    prepared_messages,
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
            all_outputs = self.model.encode(batch_decoded, use_tqdm=use_tqdm)
            all_scores = [[d[-1].item() for d in ex.outputs.data] for ex in all_outputs]

        else:
            raise ValueError(f"Model {self.model_name} is not supported")
        
        if return_full_prm_result:
            return [PRMResult(scores=scores) for scores in all_scores]
        else:
            return [PRMResult(scores=scores, aggregation_method=aggregation_method).score for scores in all_scores]
