"""LLM Judge implementations using LiteLLM"""

from .groupwise import GroupwiseJudgeModel
from .pointwise import PointwiseJudgeModel
from .prompts import CriterionRegistry
from .utils import (
    get_structured_output_fallback_stats,
    log_structured_output_fallback_stats,
    reset_response_format_fallback_state,
)


def create_pointwise_judge(
    model: str,
    criterion: str,
    **kwargs,
) -> PointwiseJudgeModel:
    """Create a pointwise judge instance"""
    return PointwiseJudgeModel(model=model, criterion=criterion, **kwargs)


def create_groupwise_judge(
    model: str,
    criterion: str,
    **kwargs,
) -> GroupwiseJudgeModel:
    """Create a groupwise judge instance"""
    return GroupwiseJudgeModel(model=model, criterion=criterion, **kwargs)


__all__ = [
    "PointwiseJudgeModel",
    "GroupwiseJudgeModel",
    "CriterionRegistry",
    "create_pointwise_judge",
    "create_groupwise_judge",
    "get_structured_output_fallback_stats",
    "log_structured_output_fallback_stats",
    "reset_response_format_fallback_state",
]
