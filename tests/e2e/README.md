# End-to-End Tests

This folder contains end-to-end tests that load actual models and require compute resources.

## Important Notes

⚠️ **These tests are NOT run in CI/CD** - they are for manual testing only.

⚠️ **These tests require**:
- GPU availability
- Model downloads (can be several GB)
- Significant compute time
- Dependencies: transformers, vllm, torch

## Running E2E Tests

### Run all E2E tests
```bash
pytest tests/e2e/
```

### Run specific E2E test file
```bash
pytest tests/e2e/hf_orm_test.py     # HuggingFace ORM tests
pytest tests/e2e/hf_prm_test.py     # HuggingFace PRM tests
pytest tests/e2e/vllm_prm_test.py   # VLLM PRM tests
pytest tests/e2e/openai_prm_test.py # OpenAI PRM tests
pytest tests/e2e/openai_drsow_test.py # DrSow tests
```

### Run specific test
```bash
pytest tests/e2e/hf_orm_test.py::test_specific_function -v
```

## Default Test Behavior

By default, `pytest` runs only unit tests (in `tests/unit/`) which use mocks and don't load actual models.

```bash
pytest              # Runs unit tests only (fast, no models)
pytest tests/unit/  # Explicitly run unit tests
pytest tests/e2e/   # Explicitly run e2e tests (slow, loads models)
```

## CI/CD Configuration

The CI/CD pipeline is configured to run:
```bash
pytest tests/unit/  # Only unit tests
```

This ensures fast, reliable builds without requiring GPU resources or model downloads.
