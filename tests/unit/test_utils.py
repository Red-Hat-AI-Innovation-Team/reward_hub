"""Unit tests for reward_hub/utils.py"""

import pytest
from reward_hub.utils import SUPPORTED_BACKENDS

pytestmark = pytest.mark.unit
from reward_hub.hf.reward import HuggingFaceOutcomeRewardModel, HuggingFaceProcessRewardModel
from reward_hub.vllm.reward import VllmProcessRewardModel
from reward_hub.openai.reward import OpenAIOutcomeRewardModel, OpenAIProcessRewardModel


class TestSupportedBackends:
    """Test SUPPORTED_BACKENDS configuration"""

    def test_supported_backends_structure(self):
        """SUPPORTED_BACKENDS dict has correct structure"""
        assert isinstance(SUPPORTED_BACKENDS, dict)
        assert len(SUPPORTED_BACKENDS) > 0

        # Verify all entries are valid
        for model_name, backends in SUPPORTED_BACKENDS.items():
            assert isinstance(model_name, str)
            assert isinstance(backends, list)
            assert len(backends) > 0

    def test_backend_classes_are_valid(self):
        """All backend classes in mapping are valid classes"""
        valid_classes = {
            HuggingFaceOutcomeRewardModel,
            HuggingFaceProcessRewardModel,
            VllmProcessRewardModel,
            OpenAIOutcomeRewardModel,
            OpenAIProcessRewardModel,
        }

        for model_name, backends in SUPPORTED_BACKENDS.items():
            for backend_class in backends:
                assert backend_class in valid_classes

    def test_expected_models_present(self):
        """Verify expected models are present"""
        expected_models = {
            "Qwen/Qwen2.5-Math-PRM-7B",
            "internlm/internlm2-7b-reward",
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
            "RLHFlow/ArmoRM-Llama3-8B-v0.1",
            "drsow",
        }

        for model in expected_models:
            assert model in SUPPORTED_BACKENDS
