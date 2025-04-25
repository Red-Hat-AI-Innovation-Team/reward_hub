from reward_hub.openai.reward import OpenAIOutcomeRewardModel
from reward_hub.drsow import DrSowConfig

class TestDrSowReward:
    def test_drsow_reward_basic(self):
        drsow_config = DrSowConfig(
            strong_model_name="Qwen/Qwen2.5-32B-instruct",
            strong_port=8305,
            weak_model_name="Qwen/Qwen2.5-32B", 
            weak_port=8306
        )
        
        reward_model = OpenAIOutcomeRewardModel(model_name="drsow", drsow_config=drsow_config)
        
        messages = [
            [
                {"role": "user", "content": "Who is Michael Jordan?"},
                {"role": "assistant", "content": "Michael Jordan is the greatest basketball player of all time"}
            ],
            [
                {"role": "user", "content": "Who is Michael Jordan?"},
                {"role": "assistant", "content": "Michael Jordan is a good friend of mine who is from Ohio."}
            ]
        ]
        
        scores = reward_model.score(
            messages=messages,
            return_raw_scores=False
        )
        
        assert len(scores) == len(messages)
        assert all(isinstance(score, float) for score in scores)
        assert scores[0] > scores[1]

    def test_drsow_reward_raw_scores(self):
        drsow_config = DrSowConfig(
            strong_model_name="Qwen/Qwen2.5-32B-instruct",
            strong_port=8305,
            weak_model_name="Qwen/Qwen2.5-32B",
            weak_port=8306
        )
        
        reward_model = OpenAIOutcomeRewardModel(model_name="drsow", drsow_config=drsow_config)
        
        messages = [
            [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ],
            [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is London."}
            ]
        ]
        
        raw_results = reward_model.score(
            messages=messages,
            return_raw_scores=True
        )
        
        assert len(raw_results) == len(messages)
        assert all("avg_drsow_reward" in result for result in raw_results)
        assert raw_results[0]["avg_drsow_reward"] > raw_results[1]["avg_drsow_reward"]

    def test_drsow_reward_batch_processing(self):
        drsow_config = DrSowConfig(
            strong_model_name="Qwen/Qwen2.5-32B-instruct",
            strong_port=8305,
            weak_model_name="Qwen/Qwen2.5-32B",
            weak_port=8306
        )
        
        reward_model = OpenAIOutcomeRewardModel(model_name="drsow", drsow_config=drsow_config)
        
        messages = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2 + 2 = 4"}
            ],
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2 + 2 = 5"}
            ],
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "Let me calculate: 2 plus 2 equals 4"}
            ],
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 22"}
            ]
        ]
        
        scores = reward_model.score(
            messages=messages,
            num_workers=40
        )
        
        assert len(scores) == len(messages)
        assert all(isinstance(score, float) for score in scores)
        assert scores[0] > scores[1]
        assert scores[2] > scores[3]
