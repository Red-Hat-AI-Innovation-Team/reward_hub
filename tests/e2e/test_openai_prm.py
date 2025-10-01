import pytest
from reward_hub.openai.reward import OpenAIProcessRewardModel
from reward_hub.vllm.reward import AggregationMethod, PRMResult

pytestmark = pytest.mark.e2e  # Mark all tests in this file as e2e


class TestOpenAIProcessRM:
    def test_openai_prm_prod_aggregation(self):
        model = OpenAIProcessRewardModel(
            model_name="Qwen/Qwen2.5-Math-PRM-7B",
            port=8305  # Assuming VLLM server is running on port 8305
        )
        
        messages = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "Let me solve this step by step:\n\n1) First, I'll add 2 and 2\n\n2) 2 + 2 = 4\n\nTherefore, \\boxed{4}"}
            ],
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2 + 2 = \\boxed{8}"}
            ]
        ]

        scores_prod = model.score(
            messages=messages,
            aggregation_method=AggregationMethod.PRODUCT,
            return_full_prm_result=False,
        )
        assert len(scores_prod) == len(messages)
        assert all(isinstance(score, float) for score in scores_prod)

    def test_openai_prm_last_aggregation(self):
        model = OpenAIProcessRewardModel(
            model_name="Qwen/Qwen2.5-Math-PRM-7B",
            port=8305  # Assuming VLLM server is running on port 8305
        )
        
        messages = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "Let me solve this step by step:\n\n1) First, I'll add 2 and 2\n\n2) 2 + 2 = 4\n\nTherefore, \\boxed{4}"}
            ],
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2 + 2 = \\boxed{8}"}
            ]
        ]

        scores_last = model.score(
            messages=messages,
            aggregation_method=AggregationMethod.LAST,
            return_full_prm_result=False,
        )
        assert len(scores_last) == len(messages)
        assert all(isinstance(score, float) for score in scores_last)

    def test_openai_prm_return_full_result(self):
        model = OpenAIProcessRewardModel(
            model_name="Qwen/Qwen2.5-Math-PRM-7B",
            port=8305
        )
        
        messages = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "Let me solve this step by step:\n\n1) First, I'll add 2 and 2\n\n2) 2 + 2 = 4\n\nTherefore, \\boxed{4}"}
            ]
        ]

        prm_results = model.score(
            messages=messages,
            aggregation_method=AggregationMethod.LAST,
            return_full_prm_result=True,
        )
        assert len(prm_results) == len(messages)
        assert all(isinstance(result, PRMResult) for result in prm_results)
        assert all(hasattr(result, 'step_scores') for result in prm_results)

