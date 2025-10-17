from reward_hub.hf.reward import HuggingFaceOutcomeRewardModel, HuggingFaceProcessRewardModel
from reward_hub.openai.reward import OpenAIOutcomeRewardModel, OpenAIProcessRewardModel

try:
    from reward_hub.vllm.reward import VllmProcessRewardModel
except ImportError:
    print("VLLM is not installed, skipping VLLM support. To enable VLLM support for Qwen/Qwen2.5-Math-PRM-7B, install vllm with `pip install reward_hub[vllm]`")
    VllmProcessRewardModel = None



SUPPORTED_BACKENDS = {
    "Qwen/Qwen2.5-Math-PRM-7B": [cls for cls in [VllmProcessRewardModel, HuggingFaceProcessRewardModel, OpenAIProcessRewardModel] if cls is not None],
    "internlm/internlm2-7b-reward": [HuggingFaceOutcomeRewardModel],
    "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data": [HuggingFaceProcessRewardModel],
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": [HuggingFaceOutcomeRewardModel],
    "drsow": [OpenAIOutcomeRewardModel],
}
