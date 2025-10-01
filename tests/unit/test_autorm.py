"""Unit tests for reward_hub/__init__.py (AutoRM factory)"""

import pytest

pytestmark = pytest.mark.unit


class TestAutoRMLoad:
    """Test AutoRM.load() factory method"""

    def test_load_with_hf_backend(self):
        """Load model with HF backend"""
        from reward_hub import AutoRM

        result = AutoRM.load("Qwen/Qwen2.5-Math-PRM-7B", "hf")
        assert result is not None

    def test_load_with_vllm_backend(self):
        """Load model with VLLM backend"""
        from reward_hub import AutoRM

        result = AutoRM.load("Qwen/Qwen2.5-Math-PRM-7B", "vllm")
        assert result is not None

    def test_unsupported_model_raises(self):
        """Unsupported model raises ValueError"""
        from reward_hub import AutoRM

        with pytest.raises(ValueError, match="not supported"):
            AutoRM.load("unsupported_model", "hf")

    def test_unsupported_backend_raises(self):
        """Unsupported backend raises ValueError"""
        from reward_hub import AutoRM

        with pytest.raises(ValueError, match="not supported"):
            AutoRM.load("Qwen/Qwen2.5-Math-PRM-7B", "invalid_method")

    def test_incompatible_model_backend_raises(self):
        """Incompatible model-backend combination raises AssertionError"""
        from reward_hub import AutoRM

        with pytest.raises(AssertionError):
            AutoRM.load("internlm/internlm2-7b-reward", "vllm")

    def test_kwargs_passed_through(self):
        """Extra kwargs passed to model constructor"""
        from reward_hub import AutoRM

        # Should not raise, just verify it works
        result = AutoRM.load("Qwen/Qwen2.5-Math-PRM-7B", "hf", custom_arg="value")
        assert result is not None
