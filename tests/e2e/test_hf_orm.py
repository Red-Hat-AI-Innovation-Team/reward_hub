import pytest
from reward_hub.hf.reward import HuggingFaceOutcomeRewardModel


pytestmark = pytest.mark.e2e  # Mark all tests in this file as e2e

class TestHuggingFaceOutcomeRM:
    def test_internlm_orm(self):
        model = HuggingFaceOutcomeRewardModel(
            model_name="internlm/internlm2-7b-reward"
        )
        
        messages = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "Let me solve this step by step:\n\n1) First, I'll add 2 and 2\n\n2) 2 + 2 = 4\n\nTherefore, 4"}
            ],
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2 + 2 = 8"}
            ]
        ]

        scores = model.score(messages)
        
        assert len(scores) == len(messages)
        assert all(isinstance(score, float) for score in scores)
        # First response should have higher score than second response
        assert scores[0] > scores[1]
