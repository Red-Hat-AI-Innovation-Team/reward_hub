from typing import Union, List
from abc import ABC, abstractmethod
import math
from enum import Enum
from dataclasses import dataclass


class AggregationMethod(Enum):
    MODEL = "model"
    PRODUCT = "prod"
    MIN = "min"
    LAST = "last"


@dataclass
class JudgeResult:
    """
    Unified result class for both pointwise and groupwise judge evaluations.

    For pointwise judges:
        - scores: List[float] containing individual scores (one per conversation)
        - reasonings: List[str] containing reasoning for each score

    For groupwise judges:
        - scores: List[float] containing binary scores (1.0 for top-N, 0.0 for others)
        - reasonings: List[str] with single reasoning explaining the ranking decision
    """
    scores: List[float]
    reasonings: List[str]


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
    """
    Abstract base class for outcome reward models and judge models
    
    This class supports both traditional reward models and LLM-based judge models
    that evaluate conversation outcomes and quality.
    """

    @abstractmethod
    def score(self, messages: Union[List[List[dict]], List[dict]], **kwargs) -> Union[List[float], float, "JudgeResult"]:
        """
        Score responses/conversations using the OpenAI chat completion format.

        Args:
            messages: Either a single conversation (List[dict]) or multiple conversations (List[List[dict]])
            **kwargs: Additional parameters (e.g., max_input_tokens, top_n, return_judge_reasoning, etc.)

        Returns:
            For single conversation: float (single score)
            For multiple conversations: List[float] (list of scores)
            If return_judge_reasoning=True: JudgeResult with scores and reasonings
        """
        pass

    async def ascore(self, messages: Union[List[List[dict]], List[dict]], **kwargs) -> Union[List[float], float, "JudgeResult"]:
        """
        Async version of score method.

        Default implementation raises NotImplementedError. Subclasses that support async scoring
        (e.g., LLM judges) should override this method.

        Args:
            messages: Either a single conversation (List[dict]) or multiple conversations (List[List[dict]])
            **kwargs: Additional parameters (e.g., return_judge_reasoning, etc.)

        Returns:
            For single conversation: float (single score)
            For multiple conversations: List[float] (list of scores)
            If return_judge_reasoning=True: JudgeResult with scores and reasonings

        Raises:
            NotImplementedError: If the subclass does not implement async scoring
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async scoring. "
            f"Use the synchronous score() method instead."
        )
    

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
    def load(self, model_name: str, load_method: str):
        """load the reward model
        supported load methods:
            - "hf": load from huggingface
            - "vllm": load from vllm
            - "openai": load from openai api
        """
        pass


