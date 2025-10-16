"""Prompt registry system for LLM judges"""

from typing import List, Dict, Optional
from dataclasses import dataclass

# Fixed procedural prompts - these should not be modified
POINTWISE_PROCEDURAL = """
You are an expert evaluator. Evaluate the response based on the given criteria.

Rate the response on a scale of 0-10 where:
- 0-2: Poor (fails to meet criteria)
- 3-4: Below average (partially meets criteria)
- 5-6: Average (adequately meets criteria) 
- 7-8: Good (well meets criteria)
- 9-10: Excellent (exceeds criteria)

Provide your evaluation as a JSON object with this exact format:
{
  "reasoning": "Your detailed reasoning for the evaluation, explaining why you assigned this score based on the criteria",
  "score": <numeric score between 0 and 10>
}

The score must be a number between 0 and 10. The reasoning should explain your evaluation process.
"""

GROUPWISE_PROCEDURAL = """
You are an expert evaluator. Compare all {num_responses} responses based on the given criteria.

Analyze each response and select the top {top_n} that best meet the criteria.

Provide your evaluation as a JSON object with this exact format:
{{
  "reasoning": "Your detailed reasoning explaining why you selected these specific responses, comparing their strengths and weaknesses against the criteria",
  "selected_indices": [list of exactly {top_n} indices]
}}

"""


@dataclass
class Criterion:
    """A structured evaluation criterion"""
    name: str
    content: str
    description: Optional[str] = None

    def __post_init__(self):
        """Validate criterion after creation"""
        if not self.name or not self.content:
            raise ValueError("Criterion must have name and content")
        if len(self.content.strip()) < 50:
            raise ValueError("Criterion content should be descriptive (>50 chars)")


class CriterionRegistry:
    """Registry for user-defined evaluation criteria"""
    
    _criteria: Dict[str, Criterion] = {}
    
    @classmethod
    def register(cls, criterion: Criterion) -> None:
        """Register a new evaluation criterion"""
        criterion.__post_init__()  # Validate
        cls._criteria[criterion.name] = criterion
    
    @classmethod
    def get(cls, name: str) -> str:
        """Get a criterion content by name"""
        if name not in cls._criteria:
            raise ValueError(f"Criterion '{name}' not found. Available: {list(cls._criteria.keys())}")
        return cls._criteria[name].content
    
    @classmethod
    def get_criterion(cls, name: str) -> Criterion:
        """Get a full criterion object by name"""
        if name not in cls._criteria:
            raise ValueError(f"Criterion '{name}' not found. Available: {list(cls._criteria.keys())}")
        return cls._criteria[name]
    
    @classmethod
    def list_criteria(cls) -> List[str]:
        """List all available criteria names"""
        return list(cls._criteria.keys())
    
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a criterion is registered"""
        return name in cls._criteria
    
    @classmethod
    def _register_builtin(cls) -> None:
        """Register built-in quality-focused criteria"""
        cls.register(Criterion(
            name="overall_quality",
            description="Evaluates general response quality across multiple dimensions",
            content="""
Evaluate the overall quality of the response:
- Accuracy and correctness of information
- Completeness in addressing the question
- Clarity and coherence of explanation
- Appropriate depth and detail level
- Professional tone and presentation
            """.strip()
        ))

        cls.register(Criterion(
            name="multi_step_tool_judge",
            description="Evaluates multi-step tool usage and workflow progression",
            content="""
Evaluation Criteria:

1. PROCESS AWARENESS: What stage are we at, and which candidate best advances the workflow?
   - Does the approach match the current stage (planning, data gathering, analysis, completion)?
   - After several tool calls, you should pause and reflect on the current state of progress, and decide the next best steps.

2. STRATEGIC REASONING:
   - Early stages: High-level, flexible planning without premature assumptions
   - Data stages: Validates assumptions before proceeding (test field categories, check availability)
   - Complex requests: Logical problem decomposition
   - Does the thinking process address the user's actual needs?

3. TOOL EXECUTION:
   - Are tools appropriate for current stage and user request?
   - Are arguments correctly configured for stated goals?
   - Do the steps build logically toward the objective?

Focus: Which candidate makes the best NEXT STEP toward successfully resolving the user's request?
            """.strip()
        ))


# Initialize built-in criteria
CriterionRegistry._register_builtin()