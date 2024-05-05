# Copyright 2023 The Alpaca Team
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

import pathlib
import sys
from typing import Dict, Optional, Sequence, Union

import datasets
import fire
import pandas as pd
import io
import json
import os
# replace all below with a vllm server call
from src import VLLM
import math
from tqdm import tqdm


sample_mode_formatter = "temperature={temperature},max_new_tokens={max_new_tokens},seed={seed}"

def read_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            # Parse the JSON data from each line and append to the list
            data.append(json.loads(line))
    return data

def truncate_prompt_middle(prompt, max_words=600):
    words = prompt.split()  # Split the string into words
    total_words = len(words)
    
    if total_words <= max_words:
        return prompt  # Return the original prompt if it's within the word limit
    
    # Calculate how many words to keep from the start and end
    keep_words = max_words // 2
    
    # Form the truncated prompt
    start = words[:keep_words]
    end = words[-keep_words:]
    
    # Combine the start and end, adding an ellipsis in the middle to indicate truncation
    truncated_prompt = ' '.join(start) + ' ... ' + ' '.join(end)
    
    return truncated_prompt




def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def jdump(obj: Union[str, dict, list], f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()



def alleq(l, f = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.

    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.

    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])


def zip_(*args):
    """Assert sequences of same length before zipping."""
    if len(args) == 0:
        return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)

# This seems to be a very nice distributed compute function
def run_decode(
    port: int,
    decoder_name_or_path: str,
    chunk_idx=1, 
    dataset_path="../noised_ppo_merlinite_train.jsonl",
    max_instances=sys.maxsize,
    per_device_batch_size=4,
    temperature=0.7,
    top_k=50,
    top_p=0.85,
    max_new_tokens=512,
    num_return_sequences=4,
    mixed_precision="bf16",
    tf32=False,
    seed: Optional[int] = None,
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
    list_dict_data = read_jsonl(dataset_path)
    
    prompts = [ex["formatted_input"] for ex in list_dict_data]
    chunk_size = max_instances
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size
    prompts, list_dict_data = prompts[start_idx:end_idx], list_dict_data[start_idx:end_idx]

    # truncate it to less than 600 words
    prompts = [truncate_prompt_middle(ex, max_words=600) for ex in prompts]
   

    vllm_server = VLLM(model_name=decoder_name_or_path)
    
    vllm_batch_size = 40

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
                stop_sequences=["\n<|assistant|>\n", "\n<|user|>\n"]
                )
        batch_results = [x["generated_text"] for x in batch_full_results]

        clean_results = []
        for clean_idx in range(len(batch_prompts)):
            clean_idx_start = clean_idx * num_return_sequences
            clean_idx_end = clean_idx_start + num_return_sequences
            clean_results.append(batch_results[clean_idx_start:clean_idx_end])
        outputs.extend(clean_results)
   # outputs = decode.decode_prompts_with_huggingface(
   #     model_name_or_path=decoder_name_or_path,
   #     prompts=prompts,
   #     decoding_args=decode.HFDecodingArguments(
   #         temperature=temperature,
   #         max_new_tokens=max_new_tokens, 
   #         num_return_sequences=num_return_sequences,
   #         top_p = 0.75,
   #         top_k = 50,
   #     ),
   #     per_device_batch_size=per_device_batch_size,
   #     mixed_precision=mixed_precision,
   #     tf32=tf32,
   #     seed=seed,
   # )

    sample_mode = sample_mode_formatter.format(temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)
    return_list_dict_data = [
        {
            "target_output": dict_data["targets"],
           "dataset":  dict_data["dataset"],
            "group": dict_data["group"],
            "output": output,
            "prompt": dict_data["formatted_input"],
            "decoder_name_or_path": decoder_name_or_path,
            "sample_mode": sample_mode,
        }
        for dict_data, prompt, output in zip_(list_dict_data, prompts, outputs)
    ]
    # if output_path is not None and distributed_utils.is_main_process():
    #     utils.jdump(return_list_dict_data, output_path)

    return return_list_dict_data


def repeat_n_inference(
    decoder_name_or_path: str,
    port: int,
    output_path: str = None,
    chunk_idx=1, 
    prompt_dict_path=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
    split="eval",
    dataset_path="../noised_ppo_merlinite_train.jsonl",
    per_device_batch_size=2,
    max_instances=20000,
    temperature=1.0,
    num_return_sequences=4,
    max_new_tokens=1024,
    mixed_precision="bf16",
    tf32=False,
    flash_attn=False,
):
    """
    - [ ]  For each perturbed prompt, compare with generation output of non-perturbed example.
        1. The non-perturbed prompt
        2. have the Mixtral-instruct generation as gold?
        3. the perturbed prompt generation
        4. sampling 3 at different temperature if applicable. (Not applicable at this point)
        5. This code shouldn’t be difficult, it basically does the top 3 steps, iteratively, and save in a format that is inaccordance with the purpose.
    """

    decode_return_list_dict_data = run_decode(
        port=port,
        decoder_name_or_path=decoder_name_or_path,
        chunk_idx=chunk_idx, 
        max_instances=max_instances,
        per_device_batch_size=per_device_batch_size,
        dataset_path=dataset_path,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        mixed_precision=mixed_precision,
        tf32=tf32,
    )

    return_list_dict_data = decode_return_list_dict_data

    if output_path is not None:
        jdump(return_list_dict_data, output_path)
    return return_list_dict_data


def main(**kwargs):
    # from alpaca_farm import distributed_utils
    # local_rank, world_size = distributed_utils.setup()
    # import torch.distributed as dist
    # dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    repeat_n_inference(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
