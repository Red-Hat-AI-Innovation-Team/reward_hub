import pytest
from reward_hub.openai.reward import OpenAIOutcomeRM
from reward_hub.drsow import DrSowConfig
def test_drsow_reward_basic():
    drsow_config = DrSowConfig(
        strong_model_name="Qwen/Qwen2.5-32B-instruct",
        strong_port=8305,
        weak_model_name="Qwen/Qwen2.5-32B", 
        weak_port=8306
    )
    
    reward_model = OpenAIOutcomeRM(model_name="drsow", drsow_config=drsow_config)
    
    question = "Who is Michael Jordan?"
    responses = [
        "Michael Jordan is the greatest basketball player of all time",
        "Michael Jordan is a good friend of mine who is from Ohio."
    ]
    system_prompt = "You are a helpful assistant."
    
    scores = reward_model.score(
        question=question,
        responses=responses,
        system_prompt=system_prompt,
        return_raw_scores=False
    )
    
    assert len(scores) == len(responses)
    assert all(isinstance(score, float) for score in scores)
    # First response should score higher as it's more factual
    assert scores[0] > scores[1]

def test_drsow_reward_raw_scores():
    drsow_config = DrSowConfig(
        strong_model_name="Qwen/Qwen2.5-32B-instruct",
        strong_port=8305,
        weak_model_name="Qwen/Qwen2.5-32B",
        weak_port=8306
    )
    
    reward_model = OpenAIOutcomeRM(model_name="drsow", drsow_config=drsow_config)
    
    question = "What is the capital of France?"
    responses = [
        "The capital of France is Paris.",
        "The capital of France is London."
    ]
    system_prompt = "You are a helpful assistant."
    
    raw_results = reward_model.score(
        question=question,
        responses=responses,
        system_prompt=system_prompt,
        return_raw_scores=True
    )
    
    assert len(raw_results) == len(responses)
    assert all("avg_drsow_reward" in result for result in raw_results)
    # First response should have higher reward as it's correct
    assert raw_results[0]["avg_drsow_reward"] > raw_results[1]["avg_drsow_reward"]

def test_drsow_reward_batch_processing():
    drsow_config = DrSowConfig(
        strong_model_name="Qwen/Qwen2.5-32B-instruct",
        strong_port=8305,
        weak_model_name="Qwen/Qwen2.5-32B",
        weak_port=8306
    )
    
    reward_model = OpenAIOutcomeRM(model_name="drsow", drsow_config=drsow_config)
    
    question = "What is 2+2?"
    responses = [
        "2 + 2 = 4",
        "2 + 2 = 5",
        "Let me calculate: 2 plus 2 equals 4",
        "The answer is 22"
    ]
    system_prompt = "You are a helpful assistant."
    
    scores = reward_model.score(
        question=question,
        responses=responses,
        system_prompt=system_prompt,
        num_workers=40
    )
    
    assert len(scores) == len(responses)
    assert all(isinstance(score, float) for score in scores)
    # Correct answers should score higher
    assert scores[0] > scores[1]
    assert scores[2] > scores[3]
