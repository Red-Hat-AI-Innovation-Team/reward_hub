from transformers import AutoTokenizer

from typing import Union, List
from reward_hub.base import AbstractOutcomeRewardModel, AbstractProcessRewardModel
from reward_hub.openai.vllm_client import vllmClient
from reward_hub.drsow import DrSow, DrSowConfig



class OpenAIOutcomeRewardModel(AbstractOutcomeRewardModel):
    def __init__(self, model_name: str = None, port: int = None, api_key: str = None, drsow_config: DrSowConfig = None, **kwargs):
        self.model_name = model_name
        self.drsow_config = drsow_config

        if model_name == "drsow":
            assert drsow_config is not None and isinstance(drsow_config, DrSowConfig)
            strong_model = vllmClient(drsow_config.strong_model_name, drsow_config.strong_port)
            weak_model = vllmClient(drsow_config.weak_model_name, drsow_config.weak_port)
            self.tokenizer = AutoTokenizer.from_pretrained(drsow_config.strong_model_name)
            self.weak_tokenizer = AutoTokenizer.from_pretrained(drsow_config.weak_model_name)
            self.model = DrSow(strong_model, weak_model, self.tokenizer, self.weak_tokenizer)
            self.tokenizer.truncation_side = "left"

        elif port is not None:
            self.model = vllmClient(model_name=model_name, port=port) # TODO: implement vllmClient
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        elif api_key is not None:
            raise NotImplementedError("OpenAI_OutcomeRM is not implemented")

        else:
            raise ValueError("Either port or api_key must be provided")

    def score(self, messages: Union[List[List[dict]], List[dict]], max_input_tokens: int = 8192, num_workers: int = 40, return_raw_scores: bool = False, system_prompt: str = None, mask_logprob_special_tokens: bool = False, **kwargs) -> List[float]:
        """
        Score responses using the OpenAI chat completion format.
        
        Args:
            messages: List of conversations in OpenAI chat completion format
            max_input_tokens: Maximum number of input tokens
            num_workers: Number of workers for parallel processing
            return_raw_scores: Whether to return raw score details
            **kwargs: Additional keyword arguments
        """
        if isinstance(messages[0], dict):
            # ensure the input is a list of list of dicts   
            messages = [messages]

        if self.model_name == "drsow":
            system_turn = [{
                "role": "system",
                "content": system_prompt
            }] if system_prompt is not None else []
            formatted_convs = [
                self.tokenizer.apply_chat_template(
                    system_turn + conv_messages,
                    add_generation_prompt=False,
                    tokenize=False,
                    max_length=max_input_tokens,
                    truncation=True,
                )
                for conv_messages in messages
            ]
            
            # Get prompts by excluding the last assistant message
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    system_turn + conv_messages[:-1],
                    add_generation_prompt=True,
                    tokenize=False
                )
                for conv_messages in messages
            ]
            
            prepared_batch = [
                {   
                    "messages": system_turn + conv_messages,
                    "formatted_conv": conv,
                    "prompt": prompt
                }
                for conv_messages, prompt, conv in zip(messages, formatted_prompts, formatted_convs)
            ]

            reward_results = self.model.get_batch_logprobs(prepared_batch, num_workers=num_workers, mask_logprob_special_tokens=mask_logprob_special_tokens)
            scores = [x["avg_drsow_reward"] for x in reward_results]

            if return_raw_scores:
                return reward_results
            else:
                return scores
        else:
            raise NotImplementedError("OpenAI_OutcomeRM is not implemented")

class OpenAIProcessRewardModel(AbstractProcessRewardModel):
    def __init__(self, model_name: str, **kwargs):
        raise NotImplementedError("OpenAI_ProcessRM is not implemented")

