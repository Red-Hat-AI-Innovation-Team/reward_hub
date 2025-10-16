"""Unit tests for DrSow module"""

import pytest
from unittest.mock import MagicMock

from reward_hub.drsow import DrSow, DrSowConfig

pytestmark = pytest.mark.unit


class TestDrSowConfig:
    """Test DrSowConfig dataclass"""

    def test_config_creation(self):
        """DrSowConfig stores parameters correctly"""
        config = DrSowConfig("strong_model", 8305, "weak_model", 8306)

        assert config.strong_model_name == "strong_model"
        assert config.strong_port == 8305
        assert config.weak_model_name == "weak_model"
        assert config.weak_port == 8306


class TestDrSowInstantiation:
    """Test DrSow initialization"""

    def test_initialization(self):
        """DrSow initializes with clients and tokenizers"""
        mock_strong_client = MagicMock()
        mock_weak_client = MagicMock()
        mock_tokenizer = MagicMock()
        mock_weak_tokenizer = MagicMock()

        drsow = DrSow(mock_strong_client, mock_weak_client, mock_tokenizer, mock_weak_tokenizer)

        assert drsow is not None
        assert drsow.strong_client == mock_strong_client
        assert drsow.weak_client == mock_weak_client
        assert drsow.tokenizer == mock_tokenizer
        assert drsow.weak_tokenizer == mock_weak_tokenizer

    def test_batch_processing_interface(self):
        """DrSow has get_batch_logprobs method with correct signature"""
        mock_strong_client = MagicMock()
        mock_weak_client = MagicMock()
        mock_tokenizer = MagicMock()
        mock_weak_tokenizer = MagicMock()

        drsow = DrSow(mock_strong_client, mock_weak_client, mock_tokenizer, mock_weak_tokenizer)

        # Verify method exists
        assert hasattr(drsow, 'get_batch_logprobs')
        assert callable(drsow.get_batch_logprobs)
