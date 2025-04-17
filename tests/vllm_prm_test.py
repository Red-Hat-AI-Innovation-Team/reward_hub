import pytest
from reward_hub.vllm.reward import VLLMProcessRM
from reward_hub.base import PRMResult

def test_vllm_prm_prod_aggregation():
    model = VLLMProcessRM(
        model_name="Qwen/Qwen2.5-Math-PRM-7B",
        device_map=5
    )
    
    question = "What is 2+2?"
    responses = [
        "Let me solve this step by step:\n\n1) First, I'll add 2 and 2\n\n2) 2 + 2 = 4\n\nTherefore, \\boxed{4}",
        "2 + 2 = \\boxed{8}"
    ]

    scores_prod = model.score(
        question=question,
        responses=responses,
        aggregate_method="prod",
        return_full_prm_result=False,
        batch_size=8
    )
    assert len(scores_prod) == len(responses)
    assert all(isinstance(score, float) for score in scores_prod)

def test_vllm_prm_last_aggregation():
    model = VLLMProcessRM(
        model_name="Qwen/Qwen2.5-Math-PRM-7B",
        device_map=5
    )
    
    question = "What is 2+2?"
    responses = [
        "Let me solve this step by step:\n\n1) First, I'll add 2 and 2\n\n2) 2 + 2 = 4\n\nTherefore, \\boxed{4}",
        "2 + 2 = \\boxed{8}"
    ]

    scores_last = model.score(
        question=question,
        responses=responses,
        aggregate_method="last",
        return_full_prm_result=False,
        batch_size=8
    )
    assert len(scores_last) == len(responses)
    assert all(isinstance(score, float) for score in scores_last)

def test_vllm_prm_full_results():
    model = VLLMProcessRM(
        model_name="Qwen/Qwen2.5-Math-PRM-7B",
        device_map=5
    )
    
    question = "What is 2+2?"
    responses = [
        "Let me solve this step by step:\n\n1) First, I'll add 2 and 2\n\n2) 2 + 2 = 4\n\nTherefore, \\boxed{4}",
        "2 + 2 = \\boxed{8}"
    ]

    full_results = model.score(
        question=question,
        responses=responses,
        aggregate_method="min",
        return_full_prm_result=True,
        batch_size=8
    )
    assert len(full_results) == len(responses)
    assert all(isinstance(result, PRMResult) for result in full_results)

def test_vllm_prm_model_aggregation():
    model = VLLMProcessRM(
        model_name="Qwen/Qwen2.5-Math-PRM-7B",
        device_map=5
    )
    
    question = "What is 2+2?"
    responses = [
        "Let me solve this step by step:\n\n1) First, I'll add 2 and 2\n\n2) 2 + 2 = 4\n\nTherefore, \\boxed{4}",
        "2 + 2 = \\boxed{8}"
    ]

    model_agg_scores = model.score(
        question=question,
        responses=responses,
        aggregate_method="model_aggregate",
        return_full_prm_result=False,
        batch_size=8
    )
    assert len(model_agg_scores) == len(responses)
    assert all(isinstance(score, float) for score in model_agg_scores)

def test_vllm_prm_invalid_model():
    with pytest.raises(ValueError):
        model = VLLMProcessRM(
            model_name="invalid_model",
            device_map=0
        )
        model.score(
            question="test",
            responses=["test"],
            aggregate_method="last"
        )
