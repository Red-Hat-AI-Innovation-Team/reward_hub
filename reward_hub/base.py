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


from typing import Union, List
from abc import ABC, abstractmethod
import math


class PRMResult:
    """
    full result of process reward model
    """
    def __init__(self, scores: List[float], aggregate_method: str = None):
        self.step_scores = scores
        self.product = math.prod(scores)
        self.min = min(scores)
        self.last = scores[-1]

        if aggregate_method == "prod":
            self.score = self.product
        elif aggregate_method == "last":
            self.score = self.last
        elif aggregate_method == "min":
            self.score = self.min
        elif aggregate_method == "model_aggregate":
            self.score = self.last
        else:
            self.score = self.last


class AbstractOutcomeRewardModel(ABC):
    """abstract base class for outcome reward models"""

    @abstractmethod
    def score(self, question: str, responses: List[str], max_input_tokens: int = 8196) -> List[float]:
        """the reward for the given response"""
        pass

class AbstractProcessRewardModel(ABC):
    """abstract base class for process reward models"""

    @abstractmethod
    def score(self, question: str, responses: List[str], step_sep: str = "\n\n", aggregate_method: str = None, return_full_prm_result: bool = False, max_input_tokens: int = 8196) -> Union[List[PRMResult], List[float]]:
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


