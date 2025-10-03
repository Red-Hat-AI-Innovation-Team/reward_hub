# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RewardHub is a Python library for reward model annotation and evaluation, supporting both Process Reward Models (PRMs) and Outcome Reward Models (ORMs). The library implements a unified interface across multiple serving backends (HuggingFace, VLLM, OpenAI) and includes DrSow (Density Ratio of Strong-over-weak) functionality for preference annotation.

## Development Commands

### Installation
```bash
uv pip install -e .
```

### Testing
```bash
uv run pytest tests/
```

Run specific tests:
```bash
uv run pytest tests/hf_orm_test.py  # HuggingFace ORM tests
uv run pytest tests/vllm_prm_test.py  # VLLM PRM tests  
uv run pytest tests/openai_drsow_test.py  # DrSow tests
```

### Code Quality
```bash
ruff check .  # Lint code
ruff format .  # Format code
```

### Launching Models
Launch single reward model:
```bash
bash scripts/launch_reward.sh [model_path]
```

Launch DrSow (strong/weak model pair):
```bash
bash scripts/launch_vllm_drsow.sh [strong_model] [weak_model]
```

## Architecture

### Core Components

1. **AutoRM Factory** (`reward_hub/__init__.py`): Main entry point that auto-detects model type and backend compatibility
2. **Abstract Base Classes** (`reward_hub/base.py`): Defines interfaces for ORM/PRM models and result aggregation
3. **Backend Implementations**: 
   - `reward_hub/hf/` - HuggingFace transformers backend
   - `reward_hub/vllm/` - VLLM serving backend  
   - `reward_hub/openai/` - OpenAI-compatible API backend
4. **DrSow Module** (`reward_hub/drsow.py`): Density ratio computation for preference annotation
5. **LLM Judge Module** (`reward_hub/llm_judge/`): LiteLLM-based judges for conversation evaluation and ranking

### Model Support Matrix

Models and their supported backends are defined in `reward_hub/utils.py:SUPPORTED_BACKENDS`. The AutoRM factory uses this mapping to validate model/backend combinations at load time.

### Key Design Patterns

- **Backend Abstraction**: All backends implement the same abstract interfaces (AbstractOutcomeRewardModel, AbstractProcessRewardModel)
- **Flexible Input Format**: All models accept OpenAI chat completion format for consistency
- **PRM Aggregation**: Process reward models support multiple aggregation methods (product, min, last, model) defined in AggregationMethod enum
- **Parallel Processing**: DrSow uses multiprocessing for concurrent strong/weak model evaluation

### Server Launch Configuration

VLLM servers use specific GPU allocation and configuration:
- Default ports: 8305 (strong model), 8306 (weak model)  
- GPU memory utilization: 85%
- Tensor parallel size: 2 for multi-GPU models
- Max model length: 10,000 tokens

## LLM Judge Module

The LLM Judge module (`reward_hub/llm_judge/`) provides conversation evaluation and ranking capabilities using LiteLLM-compatible models.

### Core Features

- **Pointwise Judges**: Score individual conversations on a 0-10 scale
- **Groupwise Judges**: Rank multiple conversations and return binary scores for top-N selection
- **Tool Call Support**: Handles OpenAI format messages with tool calls in both content and context validation
- **Async Support**: Full async/await support for both judge types
- **Extensible Criteria**: Built-in evaluation criteria plus custom criterion registration

### Judge Types

#### Pointwise Judge
```python
from reward_hub.llm_judge import create_pointwise_judge

judge = create_pointwise_judge(
    model="gpt-4o-mini",
    criterion="overall_quality",  # or custom criterion
    api_key="your_api_key"
)

# Single conversation
score = judge.score(conversation)  # Returns float 0-10

# Multiple conversations
scores = judge.score(conversations)  # Returns List[float]
```

#### Groupwise Judge
```python
from reward_hub.llm_judge import create_groupwise_judge

judge = create_groupwise_judge(
    model="gpt-4o-mini", 
    criterion="tool-judge",  # specialized for tool-based workflows
    api_key="your_api_key"
)

# Rank conversations, select top N
binary_scores = judge.score(conversations, top_n=2)  # Returns List[float] (0.0 or 1.0)
```

### Built-in Evaluation Criteria

- **overall_quality**: General response quality across multiple dimensions
- **writing_quality**: Communication and writing assessment  
- **technical_quality**: Technical accuracy and methodology
- **relevance_quality**: How well response addresses the specific query
- **tool-judge**: Specialized criterion for evaluating tool usage and workflow progression

#### Tool-Judge Criterion

The `tool-judge` criterion evaluates multi-step tool-based workflows across three dimensions:

1. **Process Awareness**: Matches current workflow stage (planning, data gathering, analysis, completion)
2. **Strategic Reasoning**: Appropriate planning depth and assumption validation for the current stage
3. **Tool Execution**: Tool appropriateness, argument configuration, and logical step progression

### Custom Criteria

```python
from reward_hub.llm_judge.prompts import Criterion, CriterionRegistry

# Register custom criterion
custom_criterion = Criterion(
    name="my_criterion",
    category="custom",
    description="Custom evaluation criterion",
    prompt_text="Your evaluation instructions here...",
    examples=["Example use case 1", "Example use case 2"]
)

CriterionRegistry.register(custom_criterion)

# Use in judges
judge = create_pointwise_judge(model="gpt-4o-mini", criterion="my_criterion")
```

### Context Validation

Groupwise judges validate that all conversations share identical context (all messages except the final assistant response). This ensures fair comparison of candidate responses:

- **Context**: `conversation[:-1]` (all messages before final assistant response)
- **Comparison Target**: `conversation[-1]` (final assistant response only)
- **Tool Call Handling**: Tool calls in both context and responses are properly extracted and compared

### Implementation Details

- **Backend**: Uses LiteLLM for multi-provider LLM access (OpenAI, Anthropic, etc.)
- **Input Format**: OpenAI chat completion format with tool call support
- **JSON Parsing**: Robust JSON response parsing with regex fallback
- **Error Handling**: Comprehensive API validation and error reporting
- **Message Processing**: Unified content extraction handles both text content and tool calls