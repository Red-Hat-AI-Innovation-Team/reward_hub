"""Pointwise judge implementation using LiteLLM"""

import litellm
import asyncio
from typing import List, Optional, Union
from ..base import AbstractOutcomeRewardModel, JudgeResult
from .prompts import CriterionRegistry, POINTWISE_PROCEDURAL
from .utils import validate_api_configuration, parse_json_response, with_retry


class PointwiseJudgeModel(AbstractOutcomeRewardModel):
    """
    Pointwise judge that scores individual conversations on a 0-10 scale.
    Uses LiteLLM for model calls with built-in retry and provider support.
    """
    
    def __init__(self,
                 model: str,
                 criterion: str,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 512,
                 **litellm_kwargs):
        """
        Initialize pointwise judge

        Args:
            model: LiteLLM model name (e.g., "gpt-4", "claude-3-sonnet", etc.)
            criterion: Name of registered criterion to use for evaluation
            api_key: API key for authentication
            base_url: Base URL for API (if using custom endpoint)
            temperature: Temperature for generation (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            **litellm_kwargs: Additional arguments passed to LiteLLM
        """
        self.model = model
        self.criterion = criterion
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.litellm_kwargs = litellm_kwargs

        # Compose full prompt from criterion + procedural
        criterion_text = CriterionRegistry.get(criterion)
        self.full_prompt = f"{criterion_text}\n\n{POINTWISE_PROCEDURAL}"

        # Set up LiteLLM configuration
        if api_key:
            litellm.api_key = api_key
        if base_url:
            litellm.api_base = base_url

        # Validate API key works by making a test call
        validate_api_configuration(self.model, **self.litellm_kwargs)
    
    def score(self, messages: Union[List[List[dict]], List[dict]], return_judge_reasoning: bool = False, **kwargs) -> Union[List[float], float, JudgeResult]:
        """
        Score conversations using the OpenAI chat completion format

        Args:
            messages: Either a single conversation (List[dict]) or multiple conversations (List[List[dict]])
            return_judge_reasoning: If True, return JudgeResult with scores and reasonings. If False, return just scores (default).
            **kwargs: Additional arguments passed to LiteLLM

        Returns:
            If return_judge_reasoning=False:
                For single conversation: float (single score 0.0-10.0)
                For multiple conversations: List[float] (list of scores)
            If return_judge_reasoning=True:
                JudgeResult with scores and reasonings lists
        """
        # Handle single conversation vs multiple conversations
        if isinstance(messages[0], dict):
            # Single conversation: List[dict]
            score, reasoning = self._score_single(messages, **kwargs)
            if return_judge_reasoning:
                return JudgeResult(scores=[score], reasonings=[reasoning])
            return score
        else:
            # Multiple conversations: List[List[dict]]
            results = [self._score_single(conv, **kwargs) for conv in messages]
            scores = [r[0] for r in results]
            reasonings = [r[1] for r in results]
            if return_judge_reasoning:
                return JudgeResult(scores=scores, reasonings=reasonings)
            return scores
    
    @with_retry(max_attempts=3, min_wait=0.1, max_wait=10.0)
    def _score_single(self, messages: List[dict], **kwargs) -> tuple[float, str]:
        """
        Score a single conversation

        Args:
            messages: Single conversation in OpenAI chat format
            **kwargs: Additional arguments passed to LiteLLM

        Returns:
            Tuple of (score, reasoning)
        """
        judge_messages = [
            {"role": "system", "content": self.full_prompt},
            {"role": "user", "content": f"Evaluate this conversation: {messages}"}
        ]

        response = litellm.completion(
            model=self.model,
            messages=judge_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.litellm_kwargs,
            **kwargs
        )

        response_text = response.choices[0].message.content
        # Parse numeric score from JSON response
        result = parse_json_response(response_text)
        return float(result["score"]), result["reasoning"]
    
    async def ascore(self, messages: Union[List[List[dict]], List[dict]], return_judge_reasoning: bool = False, **kwargs) -> Union[List[float], float, JudgeResult]:
        """
        Async version of score

        Args:
            messages: Either a single conversation (List[dict]) or multiple conversations (List[List[dict]])
            return_judge_reasoning: If True, return JudgeResult with scores and reasonings. If False, return just scores (default).
            **kwargs: Additional arguments passed to LiteLLM

        Returns:
            If return_judge_reasoning=False:
                For single conversation: float (single score 0.0-10.0)
                For multiple conversations: List[float] (list of scores)
            If return_judge_reasoning=True:
                JudgeResult with scores and reasonings lists
        """
        # Handle single conversation vs multiple conversations
        if isinstance(messages[0], dict):
            # Single conversation: List[dict]
            score, reasoning = await self._ascore_single(messages, **kwargs)
            if return_judge_reasoning:
                return JudgeResult(scores=[score], reasonings=[reasoning])
            return score
        else:
            # Multiple conversations: List[List[dict]]
            results = await asyncio.gather(*[self._ascore_single(conv, **kwargs) for conv in messages])
            scores = [r[0] for r in results]
            reasonings = [r[1] for r in results]
            if return_judge_reasoning:
                return JudgeResult(scores=scores, reasonings=reasonings)
            return scores
    
    @with_retry(max_attempts=3, min_wait=0.1, max_wait=10.0)
    async def _ascore_single(self, messages: List[dict], **kwargs) -> tuple[float, str]:
        """
        Async score a single conversation

        Args:
            messages: Single conversation in OpenAI chat format
            **kwargs: Additional arguments passed to LiteLLM

        Returns:
            Tuple of (score, reasoning)
        """
        judge_messages = [
            {"role": "system", "content": self.full_prompt},
            {"role": "user", "content": f"Evaluate the last assistant message given the context: {messages}"}
        ]

        response = await litellm.acompletion(
            model=self.model,
            messages=judge_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.litellm_kwargs,
            **kwargs
        )

        response_text = response.choices[0].message.content
        # Parse numeric score from JSON response
        result = parse_json_response(response_text)
        return float(result["score"]), result["reasoning"]
    
    
