"""Shared pytest fixtures and configuration for RewardHub tests

Dual-mode operation:
- Unit tests (default): All model loading is mocked to prevent actual downloads
- E2E tests (with -m e2e): Real model loading for integration testing

The mocking is automatically disabled when running e2e tests.
"""

import sys
import pytest
from unittest.mock import MagicMock


def pytest_configure(config):
    """
    Pytest hook that runs BEFORE test collection.
    Mock heavy dependencies to prevent actual model loading in unit tests.
    Mocking is automatically skipped for e2e tests.
    """
    # Check if we're running e2e tests (which need real imports)
    markexpr = config.getoption("-m", default="")
    is_e2e = "e2e" in markexpr and "not e2e" not in markexpr

    # Skip mocking for e2e tests
    if is_e2e:
        print("\n🔧 E2E mode: Using real model imports (no mocking)")
        return

    print("\n✓ Unit test mode: Mocking transformers and vllm")
    # Mock VLLM completely
    mock_vllm = MagicMock()
    mock_vllm.LLM = MagicMock
    sys.modules['vllm'] = mock_vllm

    # Mock transformers model loaders
    mock_transformers = MagicMock()

    # Create mock that returns instances with eval() method
    def create_mock_model(*args, **kwargs):
        m = MagicMock()
        m.eval.return_value = m
        m.get_score.return_value = 0.85
        return m

    def create_mock_tokenizer(*args, **kwargs):
        t = MagicMock()

        # Mock tensor-like object with .to() method for apply_chat_template
        mock_template_tensor = MagicMock()
        mock_template_tensor.to = MagicMock(return_value=mock_template_tensor)
        mock_template_tensor.shape = [1, 10]
        t.apply_chat_template.return_value = mock_template_tensor

        # Mock encode to return tensor-like when return_tensors="pt", else list
        def mock_encode(*args, **kwargs):
            if kwargs.get('return_tensors') == 'pt':
                tensor = MagicMock()
                tensor.to = MagicMock(return_value=tensor)
                tensor.__eq__ = MagicMock(return_value=MagicMock())  # For token_masks
                return tensor
            else:
                return [1, 2, 3]

        t.encode = mock_encode
        t.batch_decode.return_value = ["text"]
        t.padding_side = "right"
        t.truncation_side = "left"
        t.pad_token = "<pad>"
        t.eos_token = "<eos>"
        return t

    mock_transformers.AutoModel.from_pretrained = create_mock_model
    mock_transformers.AutoModelForSequenceClassification.from_pretrained = create_mock_model
    mock_transformers.AutoModelForCausalLM.from_pretrained = create_mock_model
    mock_transformers.AutoTokenizer.from_pretrained = create_mock_tokenizer

    if 'transformers' not in sys.modules:
        sys.modules['transformers'] = mock_transformers
    else:
        # Patch existing module
        sys.modules['transformers'].AutoModel.from_pretrained = create_mock_model
        sys.modules['transformers'].AutoModelForSequenceClassification.from_pretrained = create_mock_model
        sys.modules['transformers'].AutoModelForCausalLM.from_pretrained = create_mock_model
        sys.modules['transformers'].AutoTokenizer.from_pretrained = create_mock_tokenizer


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def sample_single_turn_messages():
    """Single turn conversation in OpenAI format"""
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."}
    ]


@pytest.fixture
def sample_multi_turn_messages():
    """Multi-turn conversation in OpenAI format"""
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Let me think..."},
        {"role": "user", "content": "Please answer"},
        {"role": "assistant", "content": "The answer is 4."}
    ]


@pytest.fixture
def sample_prm_messages():
    """PRM messages with step-by-step reasoning"""
    return [
        {"role": "user", "content": "Solve: 3x + 5 = 14"},
        {"role": "assistant", "content": "Let me solve step by step:\n\n1) Subtract 5 from both sides: 3x = 9\n\n2) Divide by 3: x = 3\n\nTherefore, x = 3"}
    ]


@pytest.fixture
def sample_batch_messages():
    """Batch of conversations"""
    return [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ],
        [
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "6"}
        ]
    ]


@pytest.fixture
def mock_prm_scores():
    """Sample PRM step scores"""
    return [0.9, 0.85, 0.8]


@pytest.fixture
def mock_orm_score():
    """Sample ORM score"""
    return 0.87


@pytest.fixture
def basic_mock_tokenizer():
    """Basic mock tokenizer for simple tests"""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.apply_chat_template.return_value = "formatted chat"
    tokenizer.batch_decode.return_value = ["text1", "text2"]
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    tokenizer.all_special_tokens = ["<pad>", "<eos>", "<bos>"]
    return tokenizer
