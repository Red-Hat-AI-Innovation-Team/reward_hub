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



from .base import AbstractAutoRewardModel, AbstractOutcomeRewardModel, AbstractProcessRewardModel
from .utils import SUPPORTED_MODELS
from reward_hub.src.hf.reward import HuggingFaceOutcomeRM, HuggingFaceProcessRM
from reward_hub.src.vllm.reward import VLLMOutcomeRM, VLLMProcessRM
from reward_hub.src.openai.reward import OpenAIOutcomeRM, OpenAIProcessRM




load_method_to_class = {
    "vllm": [VLLMOutcomeRM, VLLMProcessRM],
    "hf": [HuggingFaceOutcomeRM, HuggingFaceProcessRM],
    "openai": [OpenAIOutcomeRM, OpenAIProcessRM],
}


class AutoRM(AbstractAutoRewardModel):
    def load(self, model_name: str, load_method: str, **kwargs):
        """
        load_methods support the following choices:
            - "vllm": load from python vllm library
            - "hf": load from huggingface library
            - "openai": load from openai compatible api
            
        Args:
            model_name: name of the model to load
            load_method: method to use for loading the model
            **kwargs: additional keyword arguments passed to the model constructor
                     e.g. api_key for OpenAI models, device for HF models
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} is not supported. Supported models: {list(SUPPORTED_MODELS.keys())}")
            
        if load_method not in load_method_to_class:
            raise ValueError(f"Load method {load_method} is not supported. Supported methods: {list(load_method_to_class.keys())}")
            
        # Get the supported reward model classes for this model
        supported_rm_classes = SUPPORTED_MODELS[model_name]
        
        # Get the reward model classes for this load method
        load_method_classes = load_method_to_class[load_method]
        
        # Find the intersection of supported classes
        valid_classes = set(load_method_classes).intersection(supported_rm_classes)
        assert len(valid_classes) != 0, f"Model {model_name} does not support loading with method {load_method}"
        assert len(valid_classes) == 1, f"Model {model_name} method should give one-on-one mapping {load_method}"
        
        # Initialize the first valid reward model class with kwargs
        return valid_classes[0](model_name, **kwargs)
