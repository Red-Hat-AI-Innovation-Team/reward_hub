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


### **Supported Reward Serving / APIs**

#### Local Serving
We support huggingface and local vllm serving. 

```python
from reward_hub import AutoRM

model = AutoRM.load("Qwen/Qwen2.5-Math-PRM-7B", load_method="vllm") # default to using hf loading

```

#### Remote API access
We support openai api and vllm api. 
```python
from reward_hub import AutoRM
from reward_hub.openai import DrSow

model = AutoRM.load("gpt-4o", load_method="openai", api_key="your_api_key")

model = AutoRM.load("Qwen/Qwen2.5-Math-PRM-7B", load_method="openai", port=8020)

```

### DrSow reward
We support DrSow reward model. 

```python
from reward_hub.openai import DrSow, DrSowConfig

drsow_config = DrSowConfig(
    strong_model_name="Qwen/Qwen2.5-32B-instruct",
    strong_port=8305,
    weak_model_name="Qwen/Qwen2.5-32B",
    weak_port=8306
    )

model = OpenAIOutcomeRM(model_name="drsow", drsow_config=drsow_config)
```

#### Inference:
```python
scores = model.score(
    question = "How are you doing today?",
    responses = ["I am doing well, thank you for asking.", "I am doing great, thanks for asking!"],
)

print(scores) # List[float]
```


## Research

**RewardHub** serves as the official implementation of the paper:  
[**Dr. SoW: Density Ratio of Strong-over-weak LLMs for Reducing the Cost of Human Annotation in Preference Tuning**](https://arxiv.org/pdf/2411.02481)  

The paper introduces CDR, a novel approach to generating high-quality preference annotations using density ratios tailored to domain-specific needs.
