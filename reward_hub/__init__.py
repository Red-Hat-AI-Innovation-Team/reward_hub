from .base import AbstractAutoRewardModel
from .utils import SUPPORTED_BACKENDS
from reward_hub.hf.reward import HuggingFaceOutcomeRewardModel, HuggingFaceProcessRewardModel
from reward_hub.vllm.reward import VllmOutcomeRewardModel, VllmProcessRewardModel
from reward_hub.openai.reward import OpenAIOutcomeRewardModel, OpenAIProcessRewardModel
from reward_hub.openai.judge import OpenAIJudge, JudgeType
from typing import Union, Optional
import os


load_method_to_class = {
    "vllm": [VllmOutcomeRewardModel, VllmProcessRewardModel],
    "hf": [HuggingFaceOutcomeRewardModel, HuggingFaceProcessRewardModel],
    "openai": [OpenAIOutcomeRewardModel, OpenAIProcessRewardModel]
}


os.environ["TOKENIZERS_PARALLELISM"] = "true"

class AutoRM(AbstractAutoRewardModel):
    def load(model_name: str, load_method: str, **kwargs):
        """
        load_methods support the following choices:
            - "vllm": load from python vllm backend
            - "hf": load from huggingface backend
            - "openai": load model that uses openai compatible api
            
        Args:
            model_name: name of the model to load
            load_method: method to use for loading the model
            **kwargs: additional keyword arguments passed to the model constructor
                     e.g. api_key for OpenAI models, device for HF models
        """
        if model_name not in SUPPORTED_BACKENDS:
            raise ValueError(f"Model {model_name} is not supported. Supported models: {list(SUPPORTED_BACKENDS.keys())}")
            
        if load_method not in load_method_to_class:
            raise ValueError(f"Load method {load_method} is not supported. Supported methods: {list(load_method_to_class.keys())}")
            
        # Get the supported reward model classes for this model
        supported_rm_classes = SUPPORTED_BACKENDS[model_name]
        
        # Get the reward model classes for this load method
        load_method_classes = load_method_to_class[load_method]
        
        # Find the intersection of supported classes
        valid_classes = set(load_method_classes).intersection(supported_rm_classes)
        assert len(valid_classes) != 0, f"Model {model_name} does not support loading with method {load_method}"
        assert len(valid_classes) == 1, f"Model {model_name} method should give one-on-one mapping {load_method}"
        
        # Initialize the first valid reward model class with kwargs
        return list(valid_classes)[0](model_name=model_name, **kwargs)


class AutoJudge:
    """
    Factory class for creating judge instances.
    """
    
    @staticmethod
    def from_openai(model: str,
                    judge_type: Union[JudgeType, str],
                    judge_prompt: str,
                    base_url: Optional[str] = None,
                    api_key: Optional[str] = None,
                    top_n: Optional[int] = None) -> OpenAIJudge:
        """
        Create an OpenAI-compatible judge
        
        Args:
            model: Model name (e.g., "gpt-4")
            judge_type: Type of judge - "pointwise" or "groupwise"
            judge_prompt: Judge prompt (built-in name or custom prompt text)
            base_url: Base URL for OpenAI-compatible API
            api_key: API key for authentication
            top_n: Number of top responses to select (required for groupwise)
            
        Returns:
            OpenAIJudge instance
            
        Raises:
            ValueError: If invalid judge_type or missing required parameters
        """
        return OpenAIJudge(
            model=model,
            judge_type=judge_type,
            judge_prompt=judge_prompt,
            base_url=base_url,
            api_key=api_key,
            top_n=top_n
        )
