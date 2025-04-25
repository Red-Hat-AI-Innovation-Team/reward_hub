
import pytest
from reward_hub.hf.reward import HuggingFaceOutcomeRM

def test_internlm_orm():
    model = HuggingFaceOutcomeRM(
        model_name="internlm/internlm2-7b-reward",
        device=0
    )
    
    question = "What is 2+2?"
    responses = [
        "Let me solve this step by step:\n\n1) First, I'll add 2 and 2\n\n2) 2 + 2 = 4\n\nTherefore, 4",
        "2 + 2 = 8"
    ]

    scores = model.score(
        question=question,
        responses=responses,
    )
    
    assert len(scores) == len(responses)
    assert all(isinstance(score, float) for score in scores)
    # First response should have higher score than second response
    assert scores[0] > scores[1]

def test_armo_orm():
    model = HuggingFaceOutcomeRM(
        model_name="RLHFlow/ArmoRM-Llama3-8B-v0.1", 
        device=0
    )
    
    question = "What is 2+2?"
    responses = [
        "Let me solve this step by step:\n\n1) First, I'll add 2 and 2\n\n2) 2 + 2 = 4\n\nTherefore, 4",
        "what the is this? 2 + 2 = 8 lol for sure. low key easy."
    ]

    scores = model.score(
        question=question,
        responses=responses,
    )
    
    assert len(scores) == len(responses)
    assert all(isinstance(score, float) for score in scores)
    # First response should have higher score than second response
    assert scores[0] > scores[1]
