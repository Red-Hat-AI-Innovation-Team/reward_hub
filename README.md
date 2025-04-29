# RewardHub

**RewardHub** is an end-to-end library for annotating data using state-of-the-art (SoTA) reward models, critic functions, and related processes. It is designed to facilitate the generation of preference training data or define acceptance criteria for agentic or inference scaling systems such as Best-of-N sampling or Beam-Search.


## Getting Started

### Installation
Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/reward_hub.git
cd reward_hub
pip install -e .
```

### Usage Examples

RewardHub supports multiple types of reward models and serving methods. Here are the main ways to use the library:

#### Process Reward Models (PRM)
PRMs evaluate responses by analyzing the reasoning process:

```python
from reward_hub import AutoRM

# Load a math-focused PRM using HuggingFace backend
model = AutoRM.load("Qwen/Qwen2.5-Math-PRM-7B", load_method="hf", device=0)

# Example conversation
messages = [
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Let me solve this step by step:\n1) 2 + 2 = 4\nTherefore, 4"}
    ]
]

# Get scores with full PRM results
results = model.score(messages, return_full_prm_result=True)
# Or just get the scores
scores = model.score(messages, return_full_prm_result=False)
```

#### Outcome Reward Models (ORM)
ORMs focus on evaluating the final response quality:

```python
from reward_hub import AutoRM

# Load an ORM using HuggingFace backend
model = AutoRM.load("internlm/internlm2-7b-reward", load_method="hf", device=0)

scores = model.score([
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."}
    ]
])
```

#### DrSow Reward Model
DrSow uses density ratios between strong and weak models to evaluate responses:

Launch the strong and weak models first.

```bash
bash scripts/launch_drsow.sh Qwen/Qwen2.5-32B-instruct Qwen/Qwen2.5-32B
```

Then, you can launch client reward servers to acces the DrSow reward model.

```python
from reward_hub import AutoRM
from reward_hub.drsow import DrSowConfig

drsow_config = DrSowConfig(
    strong_model_name="Qwen/Qwen2.5-32B-instruct",
    strong_port=8305,
    weak_model_name="Qwen/Qwen2.5-32B",
    weak_port=8306
)

model = AutoRM.load("drsow", load_method="openai", drsow_config=drsow_config)

# Get scores for responses
scores = model.score([
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."}
    ]
])
```

### Supported Backends

RewardHub supports multiple serving backends:

- **HuggingFace** (`load_method="hf"`): Direct local model loading
- **VLLM** (`load_method="vllm"`): Optimized local serving
- **OpenAI API** (`load_method="openai"`): Remote API access

### Supported Models

We support various reward models including:

| Model | Type | HuggingFace | VLLM | OpenAI |
|-------|------|-------------|------|---------|
| `Qwen/Qwen2.5-Math-PRM-7B` | PRM | ✓ | ✓ | ✗ |
| `internlm/internlm2-7b-reward` | ORM | ✓ | ✗ | ✗ |
| `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data` | PRM | ✓ | ✗ | ✗ |
| `RLHFlow/ArmoRM-Llama3-8B-v0.1` | ORM | ✓ | ✗ | ✗ |
| `drsow` | ORM | ✗ | ✗ | ✓ |

## Research

**RewardHub** serves as the official implementation of the paper:  
[**Dr. SoW: Density Ratio of Strong-over-weak LLMs for Reducing the Cost of Human Annotation in Preference Tuning**](https://arxiv.org/pdf/2411.02481)  

The paper introduces CDR, a novel approach to generating high-quality preference annotations using density ratios tailored to domain-specific needs.
