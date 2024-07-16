import pathlib
import sys
from typing import Dict, Optional, Sequence, Union

import datasets
import fire
import pandas as pd
import io
import json
import os
import math
from tqdm import tqdm
import torch
from src import (
    jdump,
    zip_,
    read_jsonl,
    TGI,
    VLLM
)
import random
from transformers import AutoTokenizer
from multiprocessing import Process, Manager
from mt_best_of_n import load_best_of_n_mt_bench
from datasets import Dataset


def read_input_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            # Parse the JSON data from each line and append to the list
            ex = json.loads(line)
            if "prompt" in ex:
                ex["formatted_input"] = ex["prompt"]
            assert "targets" in ex 
            assert "dataset" in ex 
            assert "group" in ex 
            assert "formatted_input" in ex
            data.append(ex)
    return data


def truncate_prompt(prompt, tokenizer, max_prompt_length=1024, truncate_method="middle"):
    tokens = tokenizer.encode(prompt)
    if len(tokens) <= max_prompt_length:
        return tokenizer.decode(tokens)

    if truncate_method == "middle":
        keep_tokens = int(max_prompt_length//2)
        truncated_tokens = tokens[:keep_tokens] + tokens[-keep_tokens:]
    elif truncate_method == "left":
        truncated_tokens = tokens[:max_prompt_length]
    elif truncate_method == "right":
        truncated_tokens == tokens[-max_prompt_length:]

    return tokenizer.decode(truncated_tokens)


def repeat_n_sample(
    port: int,
    decoder_name_or_path: str,
    data_obj,
    temperature=0.7,
    top_k=50,
    top_p=0.85,
    max_new_tokens=512,
    num_return_sequences=4,
    vllm_batch_size=20,
    max_prompt_length=1024,
    truncate_method="middle",
):
    """Decode samples from the policy language model.

    Args:
        decoder_name_or_path: Name or path of the policy language model.
        dataset_path: Path to the dataset for datasets.load_dataset.
        dataset_name: Name of the dataset for datasets.load_dataset.
        prompt_dict_path: Path to the prompt dictionary for formatting the instruction and input into a string.
        output_path: Optional path to save the decoding results.
        split: Split of the dataset to decode.
        max_instances: Maximum number of instances to decode.
        per_device_batch_size: Batch size for reranking for each device.
        temperature: Temperature for decoding.
        max_new_tokens: Maximum number of new tokens to generate.
        seed: Random seed for decoding.
        num_return_sequences: Number of sequences to return per each prompt.
        mixed_precision: Mixed precision mode for the reward model.
        tf32: Whether to use tensorfloat32 for matrix multiplication.

    Returns:
        List of dict data with keys.
        If num_return_sequences > 1, each 'completion' is a list of strings. Otherwise, it is a string.
    """

    prompts = [ex["formatted_input"] for ex in data_obj]

    # truncate it to less than 600 words
    # truncation code needs updates
    tokenizer = AutoTokenizer.from_pretrained(decoder_name_or_path)
    prompts = [truncate_prompt(ex, tokenizer, max_prompt_length=max_prompt_length, truncate_method=truncate_method) for ex in prompts]
   
    # this class instance only provides function call abilities for now
    vllm_server = VLLM(decoder_name_or_path, port=port)
    outputs = []

    for i in tqdm(range(math.ceil(len(prompts)/float(vllm_batch_size)))):
        start_idx = vllm_batch_size*i
        end_idx = start_idx+vllm_batch_size
        batch_prompts = prompts[start_idx:end_idx]
        repeat_batch = []
        for p in batch_prompts:
            repeat_batch.extend([p] * num_return_sequences)

        batch_full_results = vllm_server.make_vllm_request(
                repeat_batch, 
                model_name=decoder_name_or_path, 
                port=port,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                stop_sequences=["\n<|assistant|>\n", "\n<|user|>\n"],
                num_workers=vllm_batch_size
                )

        batch_results = [x["generated_text"] for x in batch_full_results]

        # consolidate the repeat sequences back into the original data sequence
        clean_results = []
        for clean_idx in range(len(batch_prompts)):
            clean_idx_start = clean_idx * num_return_sequences
            clean_idx_end = clean_idx_start + num_return_sequences
            clean_results.append(batch_results[clean_idx_start:clean_idx_end])
        outputs.extend(clean_results)

    sample_mode_formatter = "temperature={temperature},max_new_tokens={max_new_tokens},top_p={top_p},top_k={top_k},max_prompt_length={max_prompt_length}"
    sample_mode = sample_mode_formatter.format(temperature=temperature, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, max_prompt_length=max_prompt_length)
    return_data_obj = [
        {
            "target_output": dict_data["targets"],
            "dataset":  dict_data["dataset"],
            "group": dict_data["group"],
            "output": output,
            "truncated_prompt": prompt,
            "prompt": dict_data["formatted_input"],
            "decoder_name_or_path": decoder_name_or_path,
            "sample_mode": sample_mode,
        }
        for dict_data, prompt, output in zip_(data_obj, prompts, outputs)
    ]

    return return_data_obj


def data_distribution_inference(
    base_port: int,
    decoder_name_or_path: str,
    dataset_path="../noised_ppo_merlinite_train.jsonl",
    output_path: str = None,
    max_instances=sys.maxsize,
    temperature=0.7,
    top_k=50,
    top_p=1.0,
    max_new_tokens=1024,
    num_return_sequences=4,
    vllm_batch_size=40,
    max_prompt_length=2048,
    truncate_method="middle",
    num_threads=None,
    shuffle=False,
    shard_nums=1,
    shard_idx=0,
    debug=False
):
    if "mt_bench" in dataset_path:
        list_dict_data = load_best_of_n_mt_bench("questions.jsonl", "granite_model_answer/merlinite-granite-7b-lab-4.jsonl")
    else:
        list_dict_data = read_input_jsonl(dataset_path)
    if debug:
        list_dict_data = list_dict_data[:100]
    
    # obtain the shard_range; ignore if they are not set. 
    if shard_nums <= 1:
        print("Use full dataset")
    else:
        shard_size = len(list_dict_data)//int(shard_nums)
        shard_start, shard_end = shard_idx * shard_size, shard_idx * shard_size + shard_size
        list_dict_data = list_dict_data[shard_start:shard_end] 
        print("Dumping dataset between shard_indices: ", shard_start, shard_end)

    if shuffle:
        random.shuffle(list_dict_data)
        print("dataset is being shuffled")
    
    list_dict_data = list_dict_data[:max_instances]
    # This is the current sharding script; 
    # modify it to be multi-processes
    
    # explore the world size:
    # number of gpus available
    if not num_threads:
        num_threads = torch.cuda.device_count()
        print(f"Found {num_threads} GPUs, will launch {num_threads} number of distributed jobs for inference")

    final_output = []
    # evenly distribute workload across gpus
    gpu_chunk_size = int(len(list_dict_data)//num_threads) + 1

    def submit_sampling(chunked_data_dict, port, gpu_idx, result_dict):
        chunk_data = chunked_data_dict[gpu_idx]
        chunk_output = repeat_n_sample(
            port = port,
            decoder_name_or_path = decoder_name_or_path,
            data_obj = chunk_data,
            temperature = temperature,
            top_k = top_k,
            top_p = top_p,
            max_new_tokens = max_new_tokens,
            num_return_sequences = num_return_sequences,
            vllm_batch_size = vllm_batch_size,
            max_prompt_length=max_prompt_length,
            truncate_method=truncate_method,
        )
        result_dict[gpu_idx] = chunk_output

    manager = Manager()
    results = manager.dict()
    processes = []
    chunked_data_dict = {}
    for gpu_idx in range(num_threads):

        start_idx = gpu_idx * gpu_chunk_size
        end_idx = start_idx + gpu_chunk_size
        chunk_data = list_dict_data[start_idx:end_idx]
        chunked_data_dict[gpu_idx] = chunk_data
        port = base_port + gpu_idx
        # currently, it's sequential, needs to make it distributed.
        # only apply on non-empty cases
        if chunk_data:
            processes.append(
                Process(target=submit_sampling, args=(chunked_data_dict, port, gpu_idx, results)),
            )
    for process in processes:
        process.start()

    for process in processes:
        process.join()

    
    for gpu_idx in range(num_threads):
        if gpu_idx in results:
            final_output.extend(results[gpu_idx])


    def list_to_dataset(data):
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        return dataset
    # save
    # if output_path is not None:
    #     jdump(final_output, output_path)
    
    if output_path:
        list_to_dataset(final_output).to_json(output_path)
    
    return final_output


def main(**kwargs):
    data_distribution_inference(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
