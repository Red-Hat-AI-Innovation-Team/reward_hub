import pytest
from reward_hub.vllm.reward import VllmProcessRewardModel
from reward_hub.base import PRMResult, AggregationMethod

class TestVLLMProcessRM:
    def test_vllm_prm_prod_aggregation(self):
        model = VllmProcessRewardModel(
            model_name="Qwen/Qwen2.5-Math-PRM-7B",
            device=0
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

    def test_vllm_prm_last_aggregation(self):
        model = VllmProcessRewardModel(
            model_name="Qwen/Qwen2.5-Math-PRM-7B",
            device=0
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

    def test_vllm_prm_full_results(self):
        model = VllmProcessRewardModel(
            model_name="Qwen/Qwen2.5-Math-PRM-7B",
            device=0
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

        full_results = model.score(
            messages=messages,
            aggregation_method=AggregationMethod.MIN,
            return_full_prm_result=True,
        )
        assert len(full_results) == len(messages)
        assert all(isinstance(result, PRMResult) for result in full_results)

    def test_vllm_prm_model_aggregation(self):
        model = VllmProcessRewardModel(
            model_name="Qwen/Qwen2.5-Math-PRM-7B",
            device=0
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

        model_agg_scores = model.score(
            messages=messages,
            aggregation_method=AggregationMethod.MODEL,
            return_full_prm_result=False,
        )
        assert len(model_agg_scores) == len(messages)
        assert all(isinstance(score, float) for score in model_agg_scores)

    def test_vllm_prm_invalid_model(self):
        with pytest.raises(ValueError):
            model = VllmProcessRewardModel(
                model_name="invalid_model",
                device=0
            )
            messages = [
                [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "test"}
                ]
            ]
            model.score(
                messages=messages,
                aggregation_method=AggregationMethod.LAST
            )
