# Factory Configuration
<!-- This file configures the Remote Factory for your project. -->
<!-- The factory reads this during Init mode and generates .factory/config.json from it. -->

## Goal

RewardHub is a Python library for reward model annotation and evaluation, supporting Process Reward Models (PRMs) and Outcome Reward Models (ORMs) across HuggingFace, VLLM, and OpenAI backends, with DrSow density-ratio and LLM Judge capabilities.

## Scope

### Modifiable
<!-- Files and directories the factory is allowed to create or edit. -->

- reward_hub/**/*.py
- tests/**/*.py
- eval/**
- scripts/**
- pyproject.toml
- CLAUDE.md
- factory.md

### Read-only
<!-- Files the factory may read but must never modify. -->

- LICENSE
- README.md

## Guards
<!-- Rules the factory must never violate. Checked before every commit. -->

- Do not delete or overwrite existing tests
- Do not modify files outside the declared scope
- Do not introduce secrets or credentials into the repository
- Do not modify test fixtures that other tests depend on
- Do not break the public API (AutoRM.load, PRMResult, AggregationMethod)

## Eval

### Command
<!-- The shell command the factory runs to score a change. -->

```bash
cd $PROJECT_PATH && python eval/score.py
```

### Threshold
<!-- Minimum composite score (0.0-1.0) required to keep a change. -->

0.6

## Target Branch

main

## Smoke Test
<!-- Optional e2e smoke test command. Failure = mandatory revert. -->

```bash
cd $PROJECT_PATH && python -m pytest tests/ -x -q --tb=short -m "not e2e" 2>&1 | tail -5
```

## Test Timeout

600

## Constraints
<!-- Soft rules that guide behavior but don't block commits. -->

- Prefer small, incremental changes over large rewrites
- Each change should be accompanied by at least one test
- Follow the existing code style and conventions
- Maintain backward compatibility with the public API (AutoRM.load interface)
- Unit tests must not trigger actual HTTP requests or model loading

## Eval Spec
<!-- Eval dimensions discovered from .factory/eval_profile.json -->

- tests | python -m pytest -v | weight=0.42 | parser=exit_code | Run test suite
- lint | python -m ruff check . | weight=0.25 | parser=exit_code | Run linter
- type_check | python -m mypy reward_hub/ | weight=0.125 | parser=exit_code | Run type checker
- coverage | python -m pytest --cov=reward_hub --cov-report=term -q | weight=0.125 | parser=exit_code | Measure test coverage
- observability | (inline) | weight=0.083 | parser=json | Analyze logging coverage and structured logging
