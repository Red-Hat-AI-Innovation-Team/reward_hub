# End-to-End Tests

This folder contains end-to-end tests that load actual models and require compute resources.

## Dual-Mode Testing System

RewardHub uses a **dual-mode testing approach**:

- **Unit tests** (default): Mocked dependencies, fast, no model loading
- **E2E tests** (explicit): Real models, requires GPU and downloads

## Running E2E Tests

### Run E2E tests only
```bash
pytest -m e2e
```

### Run specific E2E test file
```bash
pytest -m e2e tests/e2e/test_hf_orm.py
pytest -m e2e tests/e2e/test_vllm_prm.py
```

### Run all tests (unit + e2e)
```bash
pytest -m ""  # Empty marker = all tests
```

## Default Behavior (Unit Tests Only)

```bash
pytest  # Automatically skips e2e tests, runs only unit tests
```

This is the **default mode** used in CI/CD - fast, no model loading required.

## Requirements for E2E Tests

⚠️ **E2E tests require**:
- GPU availability (CUDA)
- Model downloads (several GB per model)
- Significant compute time
- Dependencies: `transformers`, `vllm`, `torch`

## How It Works

The dual-mode system uses pytest markers:

1. **E2E tests** are marked with `pytestmark = pytest.mark.e2e`
2. **Unit tests** in `tests/unit/` have no marker (run by default)
3. **conftest.py** automatically detects the mode:
   - E2E mode (`-m e2e`): Real imports, no mocking
   - Unit mode (default): Mocked transformers/vllm

## CI/CD Configuration

The CI/CD pipeline runs:
```bash
pytest  # Default = unit tests only (mocked, fast)
```

This ensures fast, reliable builds without requiring GPU resources or model downloads.

## Test Files

All E2E tests in this folder:
- `test_hf_orm.py` - HuggingFace ORM integration tests
- `test_hf_prm.py` - HuggingFace PRM integration tests
- `test_vllm_prm.py` - VLLM PRM integration tests
- `test_openai_prm.py` - OpenAI-compatible PRM tests
- `test_openai_drsow.py` - DrSow integration tests

## Example Output

### Unit mode (default)
```bash
$ pytest
✓ Unit test mode: Mocking transformers and vllm
...
30 passed in 2.5s
```

### E2E mode
```bash
$ pytest -m e2e
🔧 E2E mode: Using real model imports (no mocking)
Downloading model...
...
5 passed in 125.3s
```
