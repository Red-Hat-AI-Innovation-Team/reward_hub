"""Consolidated unit tests for all backend implementations"""

import pytest
from unittest.mock import MagicMock, patch

from reward_hub.base import PRMResult
from reward_hub.drsow import DrSowConfig

pytestmark = pytest.mark.unit


class TestHuggingFaceBackends:
    """Test HuggingFace ORM and PRM backends"""

    def test_hf_orm_basic_functionality(self):
        """HF ORM loads and scores conversations"""
        from reward_hub.hf.reward import HuggingFaceOutcomeRewardModel

        model = HuggingFaceOutcomeRewardModel("internlm/internlm2-7b-reward")

        # Test single conversation (auto-converted to batch)
        scores = model.score([{"role": "user", "content": "Q"}])
        assert len(scores) == 1

        # Test batch
        scores = model.score([[{"role": "user", "content": "Q1"}], [{"role": "user", "content": "Q2"}]])
        assert len(scores) == 2

    def test_hf_prm_basic_functionality(self):
        """HF PRM loads and scores with PRMResult"""
        from reward_hub.hf.reward import HuggingFaceProcessRewardModel

        with patch('reward_hub.hf.reward.make_step_rewards', return_value=[[0.9, 0.8, 0.7]]):
            model = HuggingFaceProcessRewardModel("Qwen/Qwen2.5-Math-PRM-7B")

            messages = [[{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]]

            # Test with full result
            results = model.score(messages, return_full_prm_result=True)
            assert isinstance(results[0], PRMResult)

            # Test with scores only
            scores = model.score(messages, return_full_prm_result=False)
            assert isinstance(scores[0], float)


class TestVLLMBackend:
    """Test VLLM backend"""

    def test_vllm_prm_basic_functionality(self):
        """VLLM PRM loads and scores"""
        from reward_hub.vllm.reward import VllmProcessRewardModel
        from tests.mocks.mock_models import MockVLLMOutput

        with patch('reward_hub.vllm.reward.torch.cuda.device_count', return_value=4):
            with patch('reward_hub.vllm.reward.LLM') as mock_llm_class:
                mock_llm_instance = MagicMock()
                mock_llm_instance.encode.return_value = [MockVLLMOutput([0.9, 0.8, 0.7])]
                mock_llm_class.return_value = mock_llm_instance

                model = VllmProcessRewardModel("Qwen/Qwen2.5-Math-PRM-7B")

                messages = [[{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]]

                results = model.score(messages, return_full_prm_result=True)
                assert isinstance(results[0], PRMResult)

                scores = model.score(messages, return_full_prm_result=False)
                assert isinstance(scores[0], float)


class TestOpenAIBackends:
    """Test OpenAI ORM and PRM backends"""

    def test_openai_orm_drsow(self):
        """OpenAI ORM with DrSow works"""
        from reward_hub.openai.reward import OpenAIOutcomeRewardModel

        with patch('reward_hub.openai.reward.DrSow') as mock_drsow_class:
            mock_drsow_instance = MagicMock()
            mock_drsow_instance.get_batch_logprobs.return_value = [
                {"avg_drsow_reward": 0.85, "drsow_reward": 1.2}
            ]
            mock_drsow_class.return_value = mock_drsow_instance

            config = DrSowConfig("model1", 8305, "model2", 8306)
            model = OpenAIOutcomeRewardModel(model_name="drsow", drsow_config=config)

            messages = [[{"role": "user", "content": "Q"}]]

            scores = model.score(messages)
            assert len(scores) == 1
            assert isinstance(scores[0], float)

            # Test raw scores
            raw = model.score(messages, return_raw_scores=True)
            assert isinstance(raw[0], dict)

    def test_openai_prm_basic_functionality(self):
        """OpenAI PRM loads and scores"""
        from reward_hub.openai.reward import OpenAIProcessRewardModel

        with patch('reward_hub.openai.reward.HTTPClient') as mock_client_class:
            mock_client = MagicMock()
            # Match structure: ex['data'][0]['data'] = [[0.9], [0.8], [0.7]]
            mock_client.post_reward_requests.return_value = [
                {"data": [{"data": [[0.9], [0.8], [0.7]]}]}
            ]
            mock_client_class.return_value = mock_client

            model = OpenAIProcessRewardModel("Qwen/Qwen2.5-Math-PRM-7B", port=8305)

            messages = [[{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]]

            results = model.score(messages, return_full_prm_result=True)
            assert isinstance(results[0], PRMResult)

            scores = model.score(messages, return_full_prm_result=False)
            assert isinstance(scores[0], float)


class TestBackendErrors:
    """Test error handling across backends"""

    def test_hf_unsupported_model(self):
        """HF backends raise error for unsupported models"""
        from reward_hub.hf.reward import HuggingFaceOutcomeRewardModel

        model = HuggingFaceOutcomeRewardModel("unsupported/model")

        with pytest.raises(ValueError, match="not supported"):
            model.score([[{"role": "user", "content": "Q"}]])

    def test_vllm_unsupported_model(self):
        """VLLM backend raises error for unsupported models"""
        from reward_hub.vllm.reward import VllmProcessRewardModel

        with patch('reward_hub.vllm.reward.torch.cuda.device_count', return_value=4):
            model = VllmProcessRewardModel("unsupported/model")

            with pytest.raises(ValueError, match="not supported"):
                model.score([[{"role": "user", "content": "Q"}]])

    def test_openai_unsupported_model(self):
        """OpenAI PRM raises error for unsupported models"""
        from reward_hub.openai.reward import OpenAIProcessRewardModel

        with patch('reward_hub.openai.reward.HTTPClient'):
            model = OpenAIProcessRewardModel("unsupported/model", port=8305)

            with pytest.raises(ValueError, match="not supported"):
                model.score([[{"role": "user", "content": "Q"}]])

    def test_drsow_requires_config(self):
        """DrSow requires DrSowConfig"""
        from reward_hub.openai.reward import OpenAIOutcomeRewardModel

        with pytest.raises(AssertionError):
            OpenAIOutcomeRewardModel(model_name="drsow", drsow_config=None)
