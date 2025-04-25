# Copyright 2025 GX Xu (gxxu@redhat.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from reward_hub.hf.reward import HuggingFaceOutcomeRewardModel, HuggingFaceProcessRewardModel
from reward_hub.vllm.reward import VllmOutcomeRewardModel, VllmProcessRewardModel
from reward_hub.openai.reward import OpenAIOutcomeRewardModel, OpenAIProcessRewardModel


SUPPORTED_BACKENDS = {
    "Qwen/Qwen2.5-Math-PRM-7B": [VllmProcessRewardModel, HuggingFaceProcessRewardModel, OpenAIProcessRewardModel],
    "internlm/internlm2-7b-reward": [HuggingFaceOutcomeRewardModel, OpenAIOutcomeRewardModel],
    # "Qwen/Qwen2.5-Math-RM-72B": [VllmOutcomeRewardModel, HuggingFaceOutcomeRewardModel, OpenAIOutcomeRewardModel],
    # "PRIME-RL/EurusPRM-Stage2": [HuggingFaceProcessRewardModel],
    "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data": [HuggingFaceProcessRewardModel],
    "RLHFlow/ArmoRM-Llama3-8B-v0.1": [HuggingFaceOutcomeRewardModel],
}
