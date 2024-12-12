# openPref

**openPref** is an end-to-end library for annotating data using state-of-the-art (SoTA) reward models, critic functions, and related processes. It is designed to facilitate the generation of preference training data or define acceptance criteria for agentic or inference scaling systems such as Best-of-N sampling or Beam-Search.

## Research

**openPref** serves as the official implementation of the paper:  
[**CDR: Customizable Density Ratios of Strong-over-Weak LLMs for Preference Annotation**](https://arxiv.org/pdf/2411.02481)  

The paper introduces CDR, a novel approach to generating high-quality preference annotations using density ratios tailored to domain-specific needs.

## Design Overview

### **Directory Structure**

#### `src/reward_functions/`
Contains modules for different types of reward functions.  
Reward functions are defined as mappings from strings to scalar scores, enabling the evaluation of outputs against defined criteria.

#### `src/pipelines/`
Hosts pipelines for data annotation.  
Pipelines can include steps like sampling, domain-specific annotation, and reward-based annotation to produce fully annotated datasets.
A pipeline uses one or a series of reward function to perform annotation by chaining reward functions or performing majority voting.

#### `src/data/`
Defines input and output data structures used in the annotation process.  
These structures ensure consistency and interoperability across the library.

#### `src/annotate.py`
The main script for executing the annotation process.  
This script serves as the entry point for generating annotations.

#### `src/bon_sampling.py`
Handles response generation when only seed prompts are provided.  
This module performs sampling to generate outputs (e.g., responses) for subsequent ranking and rating.

#### `scripts/`
Contains example scripts demonstrating how to perform annotations.  
These scripts offer ready-to-use templates to help users get started quickly.

## Getting Started

### Installation
Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/gx-ai-architect/openPref.git
cd openPref
pip install -e .
```


