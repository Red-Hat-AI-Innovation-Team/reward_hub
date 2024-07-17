# Copyright 2023 AllenAI. All rights reserved.
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

import json
import logging
import os
from typing import Any, Dict, List, Union
import io
import pandas as pd
from datasets import Dataset, Value, concatenate_datasets, load_dataset
from fastchat.conversation import Conversation
from huggingface_hub import HfApi
from transformers import PreTrainedTokenizer



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


def check_tokenizer_chat_template(tokenizer):
    """
    Check if tokenizer has non none chat_template attribute.
    """
    if hasattr(tokenizer, "chat_template"):
        if tokenizer.chat_template is not None:
            return True
    return False

def read_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            # Parse the JSON data from each line and append to the list
            data.append(json.loads(line))
    return data


MERLINITE_SYSTEM= "<|system|>\nYou are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
ASSISTANT = "\n<|assistant|>\n"
USER = "\n<|user|>\n"

def convert_to_json_format(input_string, system_prompt=MERLINITE_SYSTEM):
    
    # Remove the system prompt at the beginning if it exists
    if input_string.startswith(system_prompt):
        input_string = input_string[len(system_prompt):]
    else:
        raise Exception("no system prompt found, error")

    # Split the remaining string by the user and assistant tags
    segments = []
    temp = input_string
    assert temp.startswith(USER)
    role = None
    while temp:
        if temp.startswith(ASSISTANT):
            role = "assistant"
            temp = temp[len(ASSISTANT):]
        elif temp.startswith(USER):
            role = "user"
            temp = temp[len(USER):]
        else:
            content_end = min(temp.find(ASSISTANT) if temp.find(ASSISTANT) != -1 else len(temp),
                              temp.find(USER) if temp.find(USER) != -1 else len(temp))
            content = temp[:content_end]
            if content.strip():  # Prevent empty content due to consecutive tags
                segments.append({"content": content.strip(), "role": role})
            temp = temp[content_end:]
    
    return segments


def default_chat_formatter(messages):
    """Format chat messages to a single string."""
    formatted_input = ""
    
    # Iterate over each message to construct the conversation history
    for message in messages:
        if message["role"] == "system":
            # Add each message with a newline based on the role
            system_prompt = message["content"]
            formatted_input = f'<|system|>\n{system_prompt}'
        elif message["role"] == "user":
            formatted_input += "\n<|user|>\n"+message['content']
        elif message["role"] == "assistant":
            formatted_input += "\n<|assistant|>\n"+message['content']
    formatted_input += "\n<|assistant|>\n"
    return formatted_input


def save_to_local(
    results_dict: Union[Dict, List],
    save_path: str,
):
    """
    Utility for saving results in dict to the hub in programatic organization.

    Args:
        results_dict: dictionary of results to save.
        model_name: name of the model (including organization).
        target_path: path to save the results in the hub. Usually set in script (e.g. eval-set/, eval-set-scores/).
        debug: if True, save to debug repo on HF.
        local_only: if True, do not save to HF (for most non-AI2 users).
        save_metrics_for_beaker: if True, save metrics for AI2 beaker visualization.

    Returns:
        scores_url: URL to the saved scores (optional).
    """
    scores_path = save_path
    dirname = os.path.dirname(scores_path)
    os.makedirs(dirname, exist_ok=True)

    # remove old data
    if os.path.isfile(scores_path):
        os.remove(scores_path)

    with open(scores_path, "w") as f:
        if isinstance(results_dict, Dict):
            dumped = json.dumps(results_dict, indent=4, sort_keys=True)  # nol removed , default=str
            f.write(dumped)
        # else, dump each row in list
        else:
            for record in results_dict:
                dumped = json.dumps(record, indent=4, sort_keys=True) + "\n"
                f.write(dumped)




def load_simple_dataset(
    bon_dataset: bool = True,
    conv: Conversation = None,
    tokenizer: PreTrainedTokenizer = None,
    logger: logging.Logger = None,
    keep_columns: List[str] = ["text", "id"],
    max_prompt_length=1500,
) -> Dataset:
    """
    Loads either the core eval set for HERM or the existing preference data test sets.

    Args:
        core_set: if True, load the core eval set for HERM.
        custom_dialogue_formatting: if True, format the dialogue as needed for custom models (e.g. SHP and PairRM).

    Returns:
        dataset: loaded dataset with required properties.
    """

    assert isinstance(bon_dataset, Dataset)
    raw_dataset = bon_dataset
    logger.info("*** Use pre-loaded dataset ***")

    # Apply chat template
    usable_tokenizer = check_tokenizer_chat_template(tokenizer)

    # assert either conv is passed or tokenizer has chat_template
    assert conv is not None or usable_tokenizer


    if logger is not None:
        logger.info("*** Preparing dataset with HF Transformers ***")
    # docs https://huggingface.co/docs/transformers/main/en/chat_templating
    
    # prioritize using tokenizer-built-in chat-formatter
    if usable_tokenizer:
        if logger is not None:
            logger.info("*** Preparing dataset using tokenizer chat-template ***")
        dataset = raw_dataset.map(
            prepare_dialogue_from_tokenizer,
            fn_kwargs={"tokenizer": tokenizer, "max_prompt_length":max_prompt_length},
            num_proc=8,
            load_from_cache_file=False,
        )
    elif conv is not None:
        if logger is not None:
            logger.info("*** Preparing dataset with FastChat ***")
        dataset = raw_dataset.map(
            prepare_fastchat_conv,
            fn_kwargs={"dialogue_template": conv, "tokenizer": tokenizer, "max_prompt_length":max_prompt_length},
            num_proc=8,
            load_from_cache_file=False,
        )

    # remove columns if set and not custom_dialogue_formatting
    all_cols = dataset.column_names
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])

    return dataset



def truncate_prompt(tokenizer, prompt, max_prompt_length=2048, truncate_method="keep_right"):
    tokens = tokenizer.encode(prompt)
    if len(tokens) <= max_prompt_length:
        return prompt

    if truncate_method == "middle":
        keep_tokens = int(max_prompt_length//2)
        truncated_tokens = tokens[:keep_tokens] + tokens[-keep_tokens:]
    elif truncate_method == "keep_left":
        truncated_tokens = tokens[:max_prompt_length]
    elif truncate_method == "keep_right":
        truncated_tokens = tokens[-max_prompt_length:]
    else:
        raise Exception(f"Invalid truncate method: {truncate_method}")

    return tokenizer.decode(truncated_tokens)


def prepare_dialogue_from_tokenizer(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_prompt_length=800,
) -> Dict[str, Any]:
    if all(k in example.keys() for k in ["prompt", "response"]):
        # multi turn
        if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
            # iterate through prompt messages, alternate user and assistant, end with example["chosen"]/rejected
            messages = []
            for i, (line) in enumerate(example["prompt"]):
                p = line["content"]
                _ = line["role"]
                if (i + 1) % 2 == 1:
                    messages.append({"role": "user", "content": p})
                else:
                    messages.append({"role": "assistant", "content": p})
            # assert that the last message before this is user
            assert messages[-1]["role"] == "user"

            # required for DPO code only, otherwise discarded
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            # truncate from left would be ideal, and ensures same truncation between 
            # prompt and output
            formatted_prompt = truncate_prompt(tokenizer, formatted_prompt, max_prompt_length=max_prompt_length, truncate_method="keep_right")

            # end with chosen/rejected
            messages.append({"role": "assistant", "content": example["response"]})
            formatted_output = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            formatted_output = truncate_prompt(tokenizer, formatted_output, max_prompt_length=max_prompt_length, truncate_method="keep_right")

            example["formatted_output"] = formatted_output + tokenizer.eos_token if formatted_output[-1] != tokenizer.eos_token else formatted_output
            example["prompt"] = formatted_prompt
            example["messages"] = messages
        # single turn
        else:
            # needed for DPO
            raise Exception("prompt must be a list of content")
    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[response]` keys but found {list(example.keys())}"
        )
    return example



def prepare_fastchat_conv(
    example: Dict[str, Any],
    dialogue_template: Conversation,
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int,
) -> Dict[str, Any]:
    """Format example to single- or multi-turn dialogue."""
    if all(k in example.keys() for k in (["prompt", "response"])):
        # multi turn
        if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
            # iterate through prompt messages, alternate user and assistant, end with example["chosen"]/rejected
            dialogue_template.messages = []
            for i, (line) in enumerate(example["prompt"]):
                p = line["content"]
                _ = line["role"]
                if (i + 1) % 2 == 1:
                    dialogue_template.messages.append([dialogue_template.roles[0], p])
                else:
                    dialogue_template.messages.append([dialogue_template.roles[1], p])
            # assert that the last message before this is user
            assert dialogue_template.messages[-1][0] == dialogue_template.roles[0]

            # needed for DPO
            formatted_prompt = dialogue_template.get_prompt()

            # end with chosen/rejected
            dialogue_template.messages.append([dialogue_template.roles[1], example["response"]])
            formatted_output = truncate_prompt(tokenizer, dialogue_template.get_prompt(), max_prompt_length=max_prompt_length, truncate_method="keep_right")
            example["formatted_output"] = formatted_output + tokenizer.eos_token if formatted_output[-1] != tokenizer.eos_token else formatted_output
            example["prompt"] = truncate_prompt(tokenizer, formatted_prompt, max_prompt_length=max_prompt_length, truncate_method="keep_right")
            example["messages"] = dialogue_template.messages
        else:
            # single turn
            # needed for DPO
            raise Exception("prompt must be a list of content")
    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example

