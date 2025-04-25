from typing import Union, List
from abc import ABC, abstractmethod
import math
from enum import Enum


class AggregationMethod(Enum):
    MODEL = "model"
    PRODUCT = "prod"
    MIN = "min"
    LAST = "last"

class PRMResult:
    """
    full result of process reward model
    """
    def __init__(self, scores: List[float], aggregation_method: Union[AggregationMethod, str] = AggregationMethod.LAST):
        self.step_scores = scores
        self.product = math.prod(scores)
        self.min = min(scores)
        self.last = scores[-1]

        # Handle both string and enum inputs for backward compatibility
        if isinstance(aggregation_method, str):
            aggregation_method = AggregationMethod(aggregation_method)
        
        # Now aggregation_method is guaranteed to be an enum
        if aggregation_method == AggregationMethod.PRODUCT:
            self.score = self.product
        elif aggregation_method == AggregationMethod.LAST:
            self.score = self.last
        elif aggregation_method == AggregationMethod.MIN:
            self.score = self.min
        else:
            # model aggregate method; it only has one step
            assert len(scores) == 1, "model aggregate method should only have one step"
            self.score = self.last
        

class AbstractOutcomeRewardModel(ABC):
    """abstract base class for outcome reward models"""

    @abstractmethod
    def score(self, messages: Union[List[List[dict]], List[dict]], max_input_tokens: int = 8196) -> List[float]:
        """
        Score responses using the OpenAI chat completion format.
        If messages is a list of list of dicts, then each list of dicts is a conversation.
        If messages is a list of dicts, then it is a single conversation.
        """
        pass

class AbstractProcessRewardModel(ABC):
    """abstract base class for process reward models"""

    @abstractmethod
    def score(self, question: str, responses: List[str], step_sep: str = "\n\n", aggregation_method: str = None, return_full_prm_result: bool = False, max_input_tokens: int = 8196) -> Union[List[PRMResult], List[float]]:
        """the reward for the given steps"""
        pass


class AbstractAutoRewardModel(ABC):
    """
    Wrapper class for reward models.    
    auto-detect the type of reward model and return the appropriate class
    """

    @abstractmethod
    def load(self, model_name: str, load_method: str, device: Union[int, str] = None):
        """load the reward model
        supported load methods:
            - "hf": load from huggingface
            - "vllm": load from vllm
            - "openai": load from openai api
        device:
            - 0: load the model on device 0 gpu
            - only single gpu is supported for now
        """
        pass


