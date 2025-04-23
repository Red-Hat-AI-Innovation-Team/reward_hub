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


from reward_hub.hf.reward import HuggingFaceOutcomeRM, HuggingFaceProcessRM
from reward_hub.vllm.reward import VLLMOutcomeRM, VLLMProcessRM
from reward_hub.openai.reward import OpenAIOutcomeRM, OpenAIProcessRM


SUPPORTED_MODELS = {
    "Qwen/Qwen2.5-Math-PRM-7B": [VLLMProcessRM, HuggingFaceProcessRM, OpenAIProcessRM],
    "internlm/internlm2-7b-reward": [HuggingFaceOutcomeRM, OpenAIOutcomeRM],
    "Qwen/Qwen2.5-Math-RM-72B": [VLLMOutcomeRM, HuggingFaceOutcomeRM, OpenAIOutcomeRM],
    "PRIME-RL/EurusPRM-Stage2": [HuggingFaceProcessRM],
    "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data": [HuggingFaceProcessRM],
}
