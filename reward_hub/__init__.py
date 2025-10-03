from .base import AbstractAutoRewardModel
from .utils import SUPPORTED_BACKENDS
from reward_hub.hf.reward import HuggingFaceOutcomeRewardModel, HuggingFaceProcessRewardModel
from reward_hub.vllm.reward import VllmOutcomeRewardModel, VllmProcessRewardModel
from reward_hub.openai.reward import OpenAIOutcomeRewardModel, OpenAIProcessRewardModel
from reward_hub.llm_judge import create_pointwise_judge, create_groupwise_judge, CriterionRegistry
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
                     e.g. api_key for OpenAI models
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
    def from_litellm(model: str,
                     judge_type: str,
                     criterion: str,
                     api_key: Optional[str] = None,
                     base_url: Optional[str] = None,
                     top_n: Optional[int] = None,
                     **kwargs):
        """
        Create a judge using LiteLLM backend
        
        Args:
            model: LiteLLM model name (e.g., "gpt-4", "claude-3-sonnet")
            judge_type: Type of judge - "pointwise" or "groupwise"
            criterion: Name of registered criterion or custom prompt text
            api_key: API key for authentication
            base_url: Base URL for custom endpoints
            top_n: Number of top responses to select (required for groupwise)
            **kwargs: Additional arguments passed to judge constructor
            
        Returns:
            PointwiseJudgeModel or GroupwiseJudgeModel instance
            
        Raises:
            ValueError: If invalid judge_type or missing required parameters
        """
        if judge_type == "pointwise":
            return create_pointwise_judge(
                model=model,
                criterion=criterion,
                api_key=api_key,
                base_url=base_url,
                **kwargs
            )
        elif judge_type == "groupwise":
            if top_n is None:
                raise ValueError("top_n must be specified for groupwise judge")
            return create_groupwise_judge(
                model=model,
                criterion=criterion,
                api_key=api_key,
                base_url=base_url,
                **kwargs
            )
        else:
            raise ValueError(f"Invalid judge_type '{judge_type}'. Must be 'pointwise' or 'groupwise'")
    
    @staticmethod
    def list_criteria() -> list:
        """List all available evaluation criteria"""
        return CriterionRegistry.list_criteria()
    
    @staticmethod
    def register_criterion(name: str,
                          prompt_text: str,
                          description: str = "",
                          category: Optional[str] = None) -> None:
        """
        Register a custom evaluation criterion
        
        Args:
            name: Name for the criterion
            prompt_text: Evaluation prompt text
            description: Description of what this criterion evaluates
            category: Optional category for organization
        """
        from reward_hub.llm_judge.prompts import Criterion
        
        criterion = Criterion(
            name=name,
            prompt_text=prompt_text,
            description=description,
            category=category
        )
        CriterionRegistry.register(criterion)
