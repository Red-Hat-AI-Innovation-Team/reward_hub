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
from openai import OpenAI
from reward_hub.openai.vllm_client import vllmClient
from reward_hub.drsow import DrSow, DrSowConfig



class OpenAIOutcomeRM(AbstractOutcomeRewardModel):
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

    def score(self, question: str, responses: List[str], system_prompt: str = None, num_workers: int = 40, return_raw_scores: bool = False, max_input_tokens: int = 8192, **kwargs) -> List[float]:
        """
        Score responses to a question using the reward model.

        Args:
            question (str): The input question/prompt
            responses (List[str]): List of response strings to score
            **kwargs: Additional keyword arguments passed to the underlying model

        Returns:
            List[float]: List of reward scores for each response, higher scores indicate better responses

        Raises:
            NotImplementedError: If using unsupported model type
            ValueError: If responses list is empty
        """
        if self.model_name == "drsow":
            system_turn = [{
                "role": "system",
                "content": system_prompt
            }] if system_prompt is not None else []
            # TODO: add chat templates
            formatted_convs = [
                self.tokenizer.apply_chat_template(
                    system_turn + 
                    [
                        {
                            "role": "user",
                            "content": question
                        },
                        {
                            "role": "assistant",
                            "content": response 
                        }
                    ],
                    add_generation_prompt=False,
                    tokenize=False,
                    max_length=max_input_tokens,
                    truncation=True,
                )
                for response in responses
            ]
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    system_turn + 
                    [
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    add_generation_prompt=True,
                    tokenize=False
                )
                for _ in responses
            ]
            prepared_batch = [
                {
                    "formatted_conv": conv,
                    "prompt": prompt
                }
                for prompt, conv in zip(formatted_prompts, formatted_convs)
            ]

            reward_results = self.model.get_batch_logprobs(prepared_batch, num_workers=num_workers)
            scores = [x["avg_drsow_reward"] for x in reward_results]

            if return_raw_scores:
                return reward_results
            else:
                return scores

        else:
            raise NotImplementedError("OpenAI_OutcomeRM is not implemented")

class OpenAIProcessRM(AbstractProcessRewardModel):
    def __init__(self, model_name: str, **kwargs):
        raise NotImplementedError("OpenAI_ProcessRM is not implemented")



if __name__ == "__main__":
    drsow_config = DrSowConfig(
        strong_model_name="Qwen/Qwen2.5-32B-instruct",
        strong_port=8305,
        weak_model_name="Qwen/Qwen2.5-32B",
        weak_port=8306
        )

    reward_model = OpenAIOutcomeRM(model_name="drsow", drsow_config=drsow_config)
    raw_results = reward_model.score(
        question="Who is Michael Jordan?",
        responses=["Michael Jordan is the greatest basketball player of all time", "Michael Jordan is a good friend of mine who is from Ohio."],
        system_prompt="You are a helpful assistant.",
        return_raw_scores=True
    )
    print(raw_results)