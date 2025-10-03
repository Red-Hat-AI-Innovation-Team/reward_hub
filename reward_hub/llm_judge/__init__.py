"""LLM Judge implementations using LiteLLM"""

from .pointwise import PointwiseJudgeModel
from .groupwise import GroupwiseJudgeModel
from .prompts import CriterionRegistry

def create_pointwise_judge(model: str, 
                          criterion: str,
                          **kwargs) -> PointwiseJudgeModel:
    """Create a pointwise judge instance"""
    return PointwiseJudgeModel(model=model, criterion=criterion, **kwargs)

def create_groupwise_judge(model: str,
                          criterion: str, 
                          **kwargs) -> GroupwiseJudgeModel:
    """Create a groupwise judge instance"""
    return GroupwiseJudgeModel(model=model, criterion=criterion, **kwargs)

__all__ = [
    "PointwiseJudgeModel",
    "GroupwiseJudgeModel", 
    "CriterionRegistry",
    "create_pointwise_judge",
    "create_groupwise_judge"
]