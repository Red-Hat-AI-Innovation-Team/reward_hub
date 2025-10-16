"""Groupwise judge implementation using LiteLLM"""

import litellm
from typing import List, Optional, Union
from ..base import AbstractOutcomeRewardModel, JudgeResult
from .prompts import CriterionRegistry, GROUPWISE_PROCEDURAL
from .utils import validate_api_configuration, parse_json_response, extract_message_content, with_retry


class GroupwiseJudgeModel(AbstractOutcomeRewardModel):
    """
    Groupwise judge that ranks multiple conversations and returns binary scores.
    Uses LiteLLM for model calls with built-in retry and provider support.
    """
    
    def __init__(self, 
                 model: str,
                 criterion: str,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 1024,
                 **litellm_kwargs):
        """
        Initialize groupwise judge
        
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
        
        # Store criterion text for runtime composition
        self.criterion_text = CriterionRegistry.get(criterion)
        
        # Set up LiteLLM configuration
        if api_key:
            litellm.api_key = api_key
        if base_url:
            litellm.api_base = base_url
        
        # Validate API key works by making a test call
        validate_api_configuration(self.model, **self.litellm_kwargs)
    
    def _validate_shared_context(self, conversations: List[List[dict]]) -> None:
        """Validate all conversations share the same context (all messages except last)"""
        first_context = [extract_message_content(msg) for msg in conversations[0][:-1]]
        for i, conv in enumerate(conversations[1:], 1):
            if [extract_message_content(msg) for msg in conv[:-1]] != first_context:
                raise ValueError(f"Conversation {i} has different context than conversation 0")
    
    def score(self, messages: Union[List[List[dict]], List[dict]], top_n: int = 1, return_judge_reasoning: bool = False, **kwargs) -> Union[List[float], JudgeResult]:
        """
        Score conversations using the OpenAI chat completion format

        Args:
            messages: Must be multiple conversations (List[List[dict]]) for groupwise ranking
            top_n: Number of top conversations to select. Default to top_n = 1, only choose best 1 out of the group. 
            return_judge_reasoning: If True, return JudgeResult with scores and reasonings. If False, return just scores (default).
            **kwargs: Additional arguments passed to LiteLLM

        Returns:
            If return_judge_reasoning=False:
                List[float]: Binary scores (1.0 for top-N selected, 0.0 for others)
            If return_judge_reasoning=True:
                JudgeResult with scores and reasonings lists
        """
        # Groupwise judges require multiple conversations
        if isinstance(messages[0], dict):
            raise ValueError("GroupwiseJudgeModel requires multiple conversations, got single conversation")

        conversations = messages  # List[List[dict]]
        scores, reasoning = self._score_groupwise(conversations, top_n, **kwargs)
        if return_judge_reasoning:
            return JudgeResult(scores=scores, reasonings=[reasoning])
        return scores
    
    @with_retry(max_attempts=3, min_wait=0.1, max_wait=10.0)
    def _score_groupwise(self, conversations: List[List[dict]], top_n: int, **kwargs) -> tuple[List[float], str]:
        """
        Score conversations with binary ranking

        Args:
            conversations: List of conversations in OpenAI chat format
            top_n: Number of top conversations to select
            **kwargs: Additional arguments passed to LiteLLM

        Returns:
            Tuple of (scores, reasoning) where scores is list of binary scores (1.0 for top-N selected, 0.0 for others)
        """
        # Validate all conversations share the same context
        self._validate_shared_context(conversations)

        # Compose full prompt at runtime with variables
        procedural = GROUPWISE_PROCEDURAL.format(num_responses=len(conversations), top_n=top_n)
        full_prompt = f"{self.criterion_text}\n\n{procedural}"

        # Format conversation context and candidate responses
        context_messages = conversations[0][:-1]
        context_text = "\n".join([f"{msg['role'].capitalize()}: {extract_message_content(msg)}" for msg in context_messages])
        responses_text = "\n".join([f"Response {i}: {extract_message_content(conv[-1])}" for i, conv in enumerate(conversations)])

        judge_messages = [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": f"Conversation Context:\n{context_text}\n\nCandidate Responses:\n{responses_text}"}
        ]

        response = litellm.completion(
            model=self.model,
            messages=judge_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.litellm_kwargs,
            **kwargs
        )

        # Parse selected indices from JSON response
        response_text = response.choices[0].message.content
        result = parse_json_response(response_text)
        selected_indices = result["selected_indices"]
        reasoning = result["reasoning"]

        # Convert to binary scores
        scores = [0.0] * len(conversations)
        for idx in selected_indices:
            if 0 <= idx < len(conversations):
                scores[idx] = 1.0
        return scores, reasoning
    
    async def ascore(self, messages: Union[List[List[dict]], List[dict]], top_n: int = 1, return_judge_reasoning: bool = False, **kwargs) -> Union[List[float], JudgeResult]:
        """
        Async version of score

        Args:
            messages: Must be multiple conversations (List[List[dict]]) for groupwise ranking
            top_n: Number of top conversations to select; default to top_n = 1.
            return_judge_reasoning: If True, return JudgeResult with scores and reasonings. If False, return just scores (default).
            **kwargs: Additional arguments passed to LiteLLM

        Returns:
            If return_judge_reasoning=False:
                List[float]: Binary scores (1.0 for top-N selected, 0.0 for others)
            If return_judge_reasoning=True:
                JudgeResult with scores and reasonings lists
        """
        # Groupwise judges require multiple conversations
        if isinstance(messages[0], dict):
            raise ValueError("GroupwiseJudgeModel requires multiple conversations, got single conversation")

        conversations = messages  # List[List[dict]]
        scores, reasoning = await self._ascore_groupwise(conversations, top_n, **kwargs)
        if return_judge_reasoning:
            return JudgeResult(scores=scores, reasonings=[reasoning])
        return scores
    
    @with_retry(max_attempts=3, min_wait=0.1, max_wait=10.0)
    async def _ascore_groupwise(self, conversations: List[List[dict]], top_n: int, **kwargs) -> tuple[List[float], str]:
        """
        Async score conversations with binary ranking

        Args:
            conversations: List of conversations in OpenAI chat format
            top_n: Number of top conversations to select
            **kwargs: Additional arguments passed to LiteLLM

        Returns:
            Tuple of (scores, reasoning) where scores is list of binary scores (1.0 for top-N selected, 0.0 for others)
        """
        # Validate all conversations share the same context
        self._validate_shared_context(conversations)

        # Compose full prompt at runtime with variables
        procedural = GROUPWISE_PROCEDURAL.format(num_responses=len(conversations), top_n=top_n)
        full_prompt = f"{self.criterion_text}\n\n{procedural}"

        # Format conversation context and candidate responses
        context_messages = conversations[0][:-1]
        context_text = "\n".join([f"{msg['role'].capitalize()}: {extract_message_content(msg)}" for msg in context_messages])
        responses_text = "\n".join([f"Response {i}: {extract_message_content(conv[-1])}" for i, conv in enumerate(conversations)])

        judge_messages = [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": f"Conversation Context:\n{context_text}\n\nCandidate Responses:\n{responses_text}"}
        ]

        response = await litellm.acompletion(
            model=self.model,
            messages=judge_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.litellm_kwargs,
            **kwargs
        )

        # Parse selected indices from JSON response
        response_text = response.choices[0].message.content
        result = parse_json_response(response_text)
        selected_indices = result["selected_indices"]
        reasoning = result["reasoning"]

        # Convert to binary scores
        scores = [0.0] * len(conversations)
        for idx in selected_indices:
            if 0 <= idx < len(conversations):
                scores[idx] = 1.0
        return scores, reasoning
