# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RewardHub is a Python library for reward model annotation and evaluation, supporting both Process Reward Models (PRMs) and Outcome Reward Models (ORMs). The library implements a unified interface across multiple serving backends (HuggingFace, VLLM, OpenAI) and includes DrSow (Density Ratio of Strong-over-weak) functionality for preference annotation.

## Development Commands

### Installation
```bash
pip install -e .
```

### Testing
```bash
pytest tests/
```

Run specific tests:
```bash
pytest tests/hf_orm_test.py  # HuggingFace ORM tests
pytest tests/vllm_prm_test.py  # VLLM PRM tests  
pytest tests/openai_drsow_test.py  # DrSow tests
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