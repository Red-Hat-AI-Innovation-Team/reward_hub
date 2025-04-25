from reward_hub.hf.reward import HuggingFaceOutcomeRewardModel, HuggingFaceProcessRewardModel
from reward_hub.vllm.reward import VllmProcessRewardModel
from reward_hub.openai.reward import OpenAIOutcomeRewardModel


SUPPORTED_BACKENDS = {
    "Qwen/Qwen2.5-Math-PRM-7B": [VllmProcessRewardModel, HuggingFaceProcessRewardModel],
    "internlm/internlm2-7b-reward": [HuggingFaceOutcomeRewardModel],
    "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data": [HuggingFaceProcessRewardModel],
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": [HuggingFaceOutcomeRewardModel],
    "drsow": [OpenAIOutcomeRewardModel],
}
