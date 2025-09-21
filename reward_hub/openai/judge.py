"""OpenAI-compatible judge implementations"""

from typing import List, Dict, Optional, Union
from enum import Enum
import json
import re
from ..base import AbstractOutcomeRewardModel


class JudgeType(Enum):
    """Enum for different judge types"""
    POINTWISE = "pointwise"
    GROUPWISE = "groupwise"


# Groupwise evaluation prompt template
GROUPWISE_TEMPLATE = """You are an expert judge evaluating responses based on the following criteria: 

{criteria}

Conversation Context: {conversation_context}

Responses to evaluate:
{responses}

Please evaluate each response based on the given criteria, then select the top {top_n} response(s).

Return your answer as a JSON object with this exact format:
{{
  "reasoning": "detailed analysis of each response and comparison",
  "selected_indices": [exactly {top_n} integer(s)]
}}

First provide your reasoning, then specify the {top_n} response(s) you selected."""


class OpenAIJudge(AbstractOutcomeRewardModel):
    """
    OpenAI-compatible LLM-as-a-Judge implementation.
    Supports both pointwise and groupwise evaluation modes.
    """
    
    def __init__(self,
                 model: str,
                 judge_type: Union[JudgeType, str],
                 judge_prompt: str,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 top_n: Optional[int] = None):
        """
        Initialize OpenAI judge
        
        Args:
            model: Model name (e.g., "gpt-4")
            judge_type: Type of judge - "pointwise" or "groupwise"
            judge_prompt: Judge prompt (built-in name or custom prompt text)
            base_url: Base URL for OpenAI-compatible API
            api_key: API key for authentication
            top_n: Number of top responses to select (for groupwise only)
        """
        self.model = model
        self.judge_type = JudgeType(judge_type) if isinstance(judge_type, str) else judge_type
        self.judge_prompt = judge_prompt
        self.base_url = base_url
        self.api_key = api_key
        self.top_n = top_n
        
        if self.judge_type == JudgeType.GROUPWISE and top_n is None:
            raise ValueError("top_n must be specified for groupwise judge")
    
    def score(self, messages: Union[List[List[dict]], List[dict]], max_input_tokens: int = 8196) -> List[float]:
        """
        Score responses using the judge
        
        For pointwise: Each conversation is scored independently (0-10)
        For groupwise: Conversations are ranked comparatively, top-N get score 1.0, others 0.0
        
        Args:
            messages: Conversations in OpenAI chat format
            max_input_tokens: Maximum input tokens (unused for now)
            
        Returns:
            List of scores
        """
        # Normalize input format
        if isinstance(messages[0], dict):
            # Single conversation
            conversations = [messages]
        else:
            # Multiple conversations
            conversations = messages
        
        if self.judge_type == JudgeType.POINTWISE:
            return self._evaluate_pointwise(conversations)
        elif self.judge_type == JudgeType.GROUPWISE:
            return self._evaluate_groupwise(conversations)
        else:
            raise ValueError(f"Unsupported judge type: {self.judge_type}")
    
    def _evaluate_pointwise(self, conversations: List[List[Dict[str, str]]]) -> List[float]:
        """
        Evaluate conversations independently using pointwise judging
        
        Args:
            conversations: List of conversations in OpenAI chat format
            
        Returns:
            List of scores from 0-10, one per conversation
        """
        # TODO: Implement actual evaluation logic
        # This should:
        # 1. Load built-in prompts if judge_prompt is a known template
        # 2. For each conversation, create a judge prompt
        # 3. Call OpenAI API to get score
        # 4. Parse and validate score (0-10)
        # 5. Return list of scores
        
        raise NotImplementedError("Pointwise evaluation not yet implemented")
    
    def _evaluate_groupwise(self, conversations: List[List[Dict[str, str]]]) -> List[float]:
        """
        Evaluate conversations comparatively using groupwise judging
        
        Args:
            conversations: List of conversations in OpenAI chat format
            
        Returns:
            List of binary scores (1.0 for top-N selected, 0.0 for others)
        """
        # TODO: Implement actual evaluation logic
        # This should:
        # 1. Load built-in prompts if judge_prompt is a known template
        # 2. Extract query and responses from conversations
        # 3. Create comparative ranking prompt with query and all responses
        # 4. Call OpenAI API to get ranking/selection
        # 5. Parse response to get top-N indices
        # 6. Return binary list (1.0 for selected, 0.0 for not selected)
        
        raise NotImplementedError("Groupwise evaluation not yet implemented")


