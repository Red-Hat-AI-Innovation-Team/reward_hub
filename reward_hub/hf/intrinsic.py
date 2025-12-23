"""LLM-based intrinsic reward model using Hugging Face models for conditional likelihood scoring."""

import logging
import math
from typing import Union, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward_hub.base import AbstractProcessRewardModel

logger = logging.getLogger(__name__)


class HuggingFaceIntrinsicRewardModel(AbstractProcessRewardModel):
    """
    Intrinsic reward model using Hugging Face transformers for conditional likelihood scoring.

    Scoring Methods:
        - likelihood: Mean token log probabilities (higher = more likely)
        - entropy: Sum of negative normalized entropy (higher = more confident)
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 scoring_method: str = "likelihood",
                 temperature: float = 1.0,
                 max_length: int = 4096,
                 **kwargs):
        """
        Initialize the HuggingFace intrinsic reward model.

        Args:
            model_name: Name or path of the HuggingFace model
            scoring_method: What to score ("likelihood" or "entropy")
            temperature: Temperature for probability scaling
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.scoring_method = scoring_method
        self.temperature = temperature
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()

    def _tokenize_prompt_response(self, prompt: str, response: str):
        """Tokenize prompt and response, returning input_ids and response start index."""
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        combined_tokens = prompt_tokens + response_tokens

        # Truncate if too long
        if len(combined_tokens) > self.max_length:
            if len(prompt_tokens) < self.max_length:
                response_tokens = response_tokens[
                    : self.max_length - len(prompt_tokens)
                ]
                combined_tokens = prompt_tokens + response_tokens
            else:
                combined_tokens = combined_tokens[-self.max_length :]
                prompt_tokens = combined_tokens[
                    : len(combined_tokens) - len(response_tokens)
                ]

        return {
            "input_ids": combined_tokens,
            "response_start": len(prompt_tokens),
            "response_length": len(response_tokens),
        }

    def _compute_token_log_probs(
        self, input_ids: List[int], response_start: int, response_length: int
    ) -> List[float]:
        """Compute log probabilities for response tokens using cross-entropy loss."""
        if response_length == 0:
            return []

        # Convert to tensor
        input_tensor = torch.tensor([input_ids], device=self.model.device)

        with torch.no_grad():
            # Get model outputs
            outputs = self.model(input_tensor)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

            # Apply temperature scaling
            if self.temperature != 1.0:
                logits = logits / self.temperature

            # Extract logits and targets for response tokens
            # Logits are shifted: logit[i] predicts token[i+1]
            response_logits = logits[
                response_start - 1 : response_start + response_length - 1
            ]  # [response_length, vocab_size]
            response_targets = torch.tensor(
                input_ids[response_start : response_start + response_length],
                device=self.model.device,
            )

            # Compute cross-entropy loss for each token (reduction='none' gives per-token losses)
            token_losses = F.cross_entropy(
                response_logits, response_targets, reduction="none"
            )

            # Convert losses to log probabilities (CE loss = -log P(target))
            token_log_probs = (-token_losses).tolist()

        return sum(token_log_probs) / len(token_log_probs)

    def _compute_tokens_entropy(
        self, input_ids: List[int], response_start: int, response_length: int
    ) -> List[float]:
        """Compute normalized conditional entropy for response tokens - measures model uncertainty."""
        if response_length == 0:
            return []

        # Convert to tensor
        input_tensor = torch.tensor([input_ids], device=self.model.device)

        with torch.no_grad():
            # Get model outputs
            outputs = self.model(input_tensor)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

            # Apply temperature scaling
            if self.temperature != 1.0:
                logits = logits / self.temperature

            # Extract logits for response token positions
            # Logits are shifted: logit[i] predicts token[i+1]
            response_logits = logits[
                response_start - 1 : response_start + response_length - 1
            ]  # [response_length, vocab_size]

            # Compute probabilities
            probs = F.softmax(response_logits, dim=-1)  # [response_length, vocab_size]

            # Compute conditional entropy: H(Y|X) = -∑ p(y|x) log p(y|x)
            log_probs = F.log_softmax(response_logits, dim=-1)
            token_entropies = -(probs * log_probs).sum(dim=-1)  # [response_length]

            # Normalize by maximum possible entropy (log of vocab size)
            vocab_size = response_logits.shape[-1]
            max_entropy = math.log(vocab_size)
            normalized_entropies = token_entropies / max_entropy  # Now in [0, 1]

            # Convert to list - higher entropy = more uncertainty
            # We return negative normalized entropy so higher scores = more confident predictions
            token_neg_entropies = (-normalized_entropies).tolist()

        return sum(token_neg_entropies)

    def _score_single(self, prompt: str, response: str) -> float:
        """Score a single prompt-response pair using the configured scoring method."""
        try:
            # Tokenize prompt and response
            tokenization_result = self._tokenize_prompt_response(prompt, response)

            # Compute token-level scores based on scoring method
            if self.scoring_method == "likelihood":
                token_scores = self._compute_token_log_probs(
                    tokenization_result["input_ids"],
                    tokenization_result["response_start"],
                    tokenization_result["response_length"],
                )
            elif self.scoring_method == "entropy":
                token_scores = self._compute_tokens_entropy(
                    tokenization_result["input_ids"],
                    tokenization_result["response_start"],
                    tokenization_result["response_length"],
                )
            else:
                raise ValueError(f"Unknown scoring method: {self.scoring_method}")

            # Aggregate into single score
            return token_scores

        except Exception as e:
            logger.warning(
                f"Failed to compute score for prompt-response pair: {e}",
                exc_info=True,
            )
            return 0.0

    def _build_prompt_from_messages(self, messages: List[dict]) -> str:
        """Build a prompt string from OpenAI-style messages."""
        parts = []
        for msg in messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def score(
        self,
        messages: Union[List[List[dict]], List[dict]],
        responses: Union[str, List[str]] = None,
    ) -> Union[float, List[float]]:
        """
        Score response(s) using conditional likelihood.

        Args:
            messages: OpenAI-style messages. Can be:
                - List[dict]: Single conversation (last message is the response to score)
                - List[List[dict]]: Multiple conversations (each last message is scored)
            responses: Optional explicit response(s) to score against the conversation context.
                If provided, the last message in each conversation is treated as part of the prompt.

        Returns:
            - For single conversation: float score
            - For multiple conversations: List[float] scores
        """
        # Normalize input to list of conversations
        if isinstance(messages[0], dict):
            messages = [messages]

        all_scores = []

        for conv_messages in messages:
            if responses is None:
                # Last message is the response to score
                prompt_messages = conv_messages[:-1]
                response = conv_messages[-1]['content']
            else:
                # All messages are prompt, use provided response
                prompt_messages = conv_messages
                response = responses if isinstance(responses, str) else responses[len(all_scores)]

            # Build prompt from all messages except the last
            prompt = self._build_prompt_from_messages(prompt_messages) if prompt_messages else ""

            # Score this conversation
            score = self._score_single(prompt, response)
            all_scores.append(score)

        # Return single score or list based on input
        return all_scores[0] if len(all_scores) == 1 and isinstance(messages, list) and len(messages) == 1 else all_scores
