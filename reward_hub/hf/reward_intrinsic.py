"""LLM-based reward model integration using Hugging Face models."""

import re
import math
import torch
import torch.nn.functional as F
from typing import Optional, Union

from its_hub.base import AbstractProcessRewardModel, AbstractLanguageModel


class IntrinsicRewardModel(AbstractProcessRewardModel):
    """
    A reward model that uses the conditional likelihood P(response|prompt) as the reward score.
    
    This leverages the language model's evaluate() method to compute token-level likelihoods
    and aggregates them into a single score.
    """
    
    def __init__(
        self,
        lm: AbstractLanguageModel,
        aggregation_method: str = "mean_log_prob",
        normalize_by_length: bool = True,
        temperature: float = 1.0,
    ):
        """
        Initialize the likelihood-based reward model.
        
        Args:
            lm: The language model to use for likelihood computation
            aggregation_method: How to aggregate token-level likelihoods
                - "mean_log_prob": Mean of log probabilities (default)
                - "sum_log_prob": Sum of log probabilities  
                - "perplexity": Negative log perplexity
                - "normalized_prob": Geometric mean of probabilities
            normalize_by_length: Whether to normalize by sequence length
            temperature: Temperature for probability scaling (higher = more uniform)
        """
        self.lm = lm
        self.aggregation_method = aggregation_method
        self.normalize_by_length = normalize_by_length
        self.temperature = temperature
    
    def _aggregate_token_likelihoods(self, token_log_probs: list[float]) -> float:
        """Aggregate token-level log probabilities into a single score."""
        if not token_log_probs:
            return 0.0
        
        # Apply temperature scaling
        scaled_log_probs = [log_prob / self.temperature for log_prob in token_log_probs]
        
        if self.aggregation_method == "mean_log_prob":
            score = sum(scaled_log_probs) / len(scaled_log_probs)
        elif self.aggregation_method == "sum_log_prob":
            score = sum(scaled_log_probs)
            if self.normalize_by_length:
                score = score / len(scaled_log_probs)
        elif self.aggregation_method == "perplexity":
            avg_log_prob = sum(scaled_log_probs) / len(scaled_log_probs)
            score = -avg_log_prob  # Negative log perplexity (higher is better)
        elif self.aggregation_method == "normalized_prob":
            # Geometric mean of probabilities
            avg_log_prob = sum(scaled_log_probs) / len(scaled_log_probs)
            score = math.exp(avg_log_prob)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        return score
    
    def _score_single(self, prompt: str, response: str) -> float:
        """Score a single prompt-response pair using conditional likelihood."""
        try:
            # Get token-level log probabilities from the language model
            token_log_probs = self.lm.evaluate(prompt, response)
            
            # Aggregate into single score
            return self._aggregate_token_likelihoods(token_log_probs)
        except Exception as e:
            # Handle cases where evaluate() might fail
            print(f"Warning: Failed to compute likelihood for prompt-response pair: {e}")
            return 0.0
    
    def score(
        self, prompt: str | list[str], response_or_responses: str | list[str]
    ) -> float | list[float]:
        """
        Score prompt-response pairs using conditional likelihood.
        
        Args:
            prompt: Single prompt string or list of prompts
            response_or_responses: Single response string or list of responses
            
        Returns:
            Single score (float) or list of scores (list[float])
        """
        # Handle prompt input - convert to list for consistent processing
        prompts = [prompt] if isinstance(prompt, str) else prompt
        
        # Handle response input and track if single response was provided
        is_single_response = isinstance(response_or_responses, str)
        responses = (
            [response_or_responses] if is_single_response else response_or_responses
        )
        
        # Ensure prompts and responses are compatible
        if len(prompts) == 1 and len(responses) > 1:
            # Broadcast single prompt to match multiple responses
            prompts = prompts * len(responses)
        elif len(prompts) != len(responses):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) and responses ({len(responses)}) must match, "
                f"or prompt must be a single string."
            )
        
        # Score each prompt-response pair
        scores = [
            self._score_single(p, r) for p, r in zip(prompts, responses)
        ]
        
        # Return single score if single response was provided, otherwise list
        return scores[0] if is_single_response else scores


class HuggingFaceIntrinsicRewardModel(AbstractProcessRewardModel):
    """
    A reward model that directly uses Hugging Face transformers to compute conditional likelihood.
    
    This implementation loads models directly using transformers library and computes
    token-level likelihoods without going through the AbstractLanguageModel interface.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        aggregation_method: str = "mean_log_prob",
        scoring_method: str = "likelihood",
        normalize_by_length: bool = True,
        temperature: float = 1.0,
        max_length: int = 4096,
        trust_remote_code: bool = False,
    ):
        """
        Initialize the HuggingFace native likelihood reward model.
        
        Args:
            model_name: Name or path of the HuggingFace model (default: Qwen/Qwen2.5-1.5B-Instruct)
            device: Device to load model on ("auto", "cuda", "cpu", etc.)
            torch_dtype: PyTorch dtype for model weights (e.g., torch.float16)
            aggregation_method: How to aggregate token-level scores
                - "mean_log_prob": Mean of log probabilities (default)
                - "sum_log_prob": Sum of log probabilities
                - "perplexity": Negative log perplexity
                - "normalized_prob": Geometric mean of probabilities
            scoring_method: What to score
                - "likelihood": Token log probabilities (default)
                - "entropy": Conditional entropy (model uncertainty)
            normalize_by_length: Whether to normalize by sequence length
            temperature: Temperature for probability scaling
            max_length: Maximum sequence length for tokenization
            trust_remote_code: Whether to trust remote code when loading model
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.device = device
        self.aggregation_method = aggregation_method
        self.scoring_method = scoring_method
        self.normalize_by_length = normalize_by_length
        self.temperature = temperature
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )
        
        # Move model to device
        if device == "auto":
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.model = self.model.to(device)
            self.device = device
        
        self.model.eval()
    
    def _tokenize_prompt_response(self, prompt: str, response: str):
        """Tokenize prompt and response, returning input_ids and response start index."""
        # Tokenize prompt and response separately to identify response tokens
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        
        # Combine prompt + response
        combined_tokens = prompt_tokens + response_tokens
        
        # Truncate if too long
        if len(combined_tokens) > self.max_length:
            # Keep the prompt and truncate response
            if len(prompt_tokens) < self.max_length:
                response_tokens = response_tokens[:self.max_length - len(prompt_tokens)]
                combined_tokens = prompt_tokens + response_tokens
            else:
                # If prompt itself is too long, truncate from beginning
                combined_tokens = combined_tokens[-self.max_length:]
                prompt_tokens = combined_tokens[:len(combined_tokens) - len(response_tokens)]
        
        return {
            "input_ids": combined_tokens,
            "prompt_length": len(prompt_tokens),
            "response_start": len(prompt_tokens),
            "response_length": len(response_tokens)
        }
    
    def _compute_token_log_probs(self, input_ids: list[int], response_start: int, response_length: int) -> list[float]:
        """Compute log probabilities for response tokens using cross-entropy loss."""
        if response_length == 0:
            return []
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(input_tensor)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
            
            # Apply temperature scaling
            if self.temperature != 1.0:
                logits = logits / self.temperature
            
            # Extract logits and targets for response tokens
            # Logits are shifted: logit[i] predicts token[i+1]
            response_logits = logits[response_start-1:response_start+response_length-1]  # [response_length, vocab_size]
            response_targets = torch.tensor(input_ids[response_start:response_start+response_length], device=self.device)
            
            # Compute cross-entropy loss for each token (reduction='none' gives per-token losses)
            token_losses = F.cross_entropy(response_logits, response_targets, reduction='none')
            
            # Convert losses to log probabilities (CE loss = -log P(target))
            token_log_probs = (-token_losses).tolist()
        
        return token_log_probs
    
    def _compute_tokens_entropy(self, input_ids: list[int], response_start: int, response_length: int) -> list[float]:
        """Compute normalized conditional entropy for response tokens - measures model uncertainty."""
        if response_length == 0:
            return []
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(input_tensor)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
            
            # Apply temperature scaling
            if self.temperature != 1.0:
                logits = logits / self.temperature
            
            # Extract logits for response token positions
            # Logits are shifted: logit[i] predicts token[i+1]
            response_logits = logits[response_start-1:response_start+response_length-1]  # [response_length, vocab_size]
            
            # Compute probabilities
            probs = F.softmax(response_logits, dim=-1)  # [response_length, vocab_size]
            
            # Compute conditional entropy: H(Y|X) = -sum p(y|x) log p(y|x)
            log_probs = F.log_softmax(response_logits, dim=-1)
            token_entropies = -(probs * log_probs).sum(dim=-1)  # [response_length]
            
            # Normalize by maximum possible entropy (log of vocab size)
            vocab_size = response_logits.shape[-1]
            max_entropy = math.log(vocab_size)
            normalized_entropies = token_entropies / max_entropy  # Now in [0, 1]
            
            # Convert to list - higher entropy = more uncertainty
            # We return negative normalized entropy so higher scores = more confident predictions
            token_neg_entropies = (-normalized_entropies).tolist()
        
        return token_neg_entropies
    
    def _aggregate_token_likelihoods(self, token_log_probs: list[float]) -> float:
        """Aggregate token-level log probabilities into a single score."""
        if not token_log_probs:
            return 0.0
        
        if self.aggregation_method == "mean_log_prob":
            score = sum(token_log_probs) / len(token_log_probs)
        elif self.aggregation_method == "sum_log_prob":
            score = sum(token_log_probs)
            if self.normalize_by_length:
                score = score / len(token_log_probs)
        elif self.aggregation_method == "perplexity":
            avg_log_prob = sum(token_log_probs) / len(token_log_probs)
            score = -avg_log_prob  # Negative log perplexity (higher is better)
        elif self.aggregation_method == "normalized_prob":
            # Geometric mean of probabilities
            avg_log_prob = sum(token_log_probs) / len(token_log_probs)
            score = math.exp(avg_log_prob)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        return score
    
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
                    tokenization_result["response_length"]
                )
            elif self.scoring_method == "entropy":
                token_scores = self._compute_tokens_entropy(
                    tokenization_result["input_ids"],
                    tokenization_result["response_start"],
                    tokenization_result["response_length"]
                )
            else:
                raise ValueError(f"Unknown scoring method: {self.scoring_method}")
            
            # Aggregate into single score
            return self._aggregate_token_likelihoods(token_scores)
            
        except Exception as e:
            print(f"Warning: Failed to compute score for prompt-response pair: {e}")
            return 0.0
    
    def score(
        self, prompt: str | list[str], response_or_responses: str | list[str]
    ) -> float | list[float]:
        """
        Score prompt-response pairs using conditional likelihood.
        
        Args:
            prompt: Single prompt string or list of prompts
            response_or_responses: Single response string or list of responses
            
        Returns:
            Single score (float) or list of scores (list[float])
        """
        # Handle prompt input - convert to list for consistent processing
        prompts = [prompt] if isinstance(prompt, str) else prompt
        
        # Handle response input and track if single response was provided
        is_single_response = isinstance(response_or_responses, str)
        responses = (
            [response_or_responses] if is_single_response else response_or_responses
        )
        
        # Ensure prompts and responses are compatible
        if len(prompts) == 1 and len(responses) > 1:
            # Broadcast single prompt to match multiple responses
            prompts = prompts * len(responses)
        elif len(prompts) != len(responses):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) and responses ({len(responses)}) must match, "
                f"or prompt must be a single string."
            )
        
        # Score each prompt-response pair
        scores = [
            self._score_single(p, r) for p, r in zip(prompts, responses)
        ]
        
        # Return single score if single response was provided, otherwise list
        return scores[0] if is_single_response else scores
    
    def __del__(self):
        """Clean up GPU memory when object is deleted."""
        if hasattr(self, 'model') and hasattr(self.model, 'cpu'):
            self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()