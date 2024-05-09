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

    if usable_tokenizer:
        if logger is not None:
            logger.info("*** Preparing dataset with HF Transformers ***")
        # docs https://huggingface.co/docs/transformers/main/en/chat_templating
        dataset = raw_dataset.map(
            prepare_dialogue_from_tokenizer_simple,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=8,
            load_from_cache_file=False,
        )

    # else use FastChat to get chat template
    else:
        if logger is not None:
            logger.info("*** Preparing dataset with FastChat ***")
        dataset = raw_dataset.map(
            prepare_dialogue_simple,
            fn_kwargs={"dialogue_template": conv},
            num_proc=8,  # using >1 process causes issues with re-assigning prompt in example
            load_from_cache_file=False,
        )


    # remove columns if set and not custom_dialogue_formatting
    all_cols = dataset.column_names
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])

    return dataset



def prepare_dialogue_from_tokenizer_simple(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, Any]:
    if all(k in example.keys() for k in ["response"]):
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
            temp_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            # end with chosen/rejected
            messages.append({"role": "assistant", "content": example["response"]})
            example["formatted_output"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            example["prompt"] = temp_prompt
        # single turn
        else:
            # needed for DPO
            messages = [
                {"role": "user", "content": example["prompt"]},
            ]
            temp_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ]
            example["formatted_output"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            
            example["prompt"] = temp_prompt
    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[response]` keys but found {list(example.keys())}"
        )
    return example


def prepare_dialogue_simple(
    example: Dict[str, Any],
    dialogue_template: Conversation,
) -> Dict[str, Any]:
    """Format example to single- or multi-turn dialogue."""
    if all(k in example.keys() for k in ("chosen", "rejected")):
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
            temp_prompt = dialogue_template.get_prompt()

            # end with chosen/rejected
            dialogue_template.messages.append([dialogue_template.roles[1], example["response"]])
            example["formatted_output"] = dialogue_template.get_prompt()
            example["prompt"] = temp_prompt

        # single turn
        else:
            if isinstance(example["prompt"], list):
                example["prompt"] = example["prompt"][0]
            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
            ]
            temp_prompt = dialogue_template.get_prompt()

            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
                [dialogue_template.roles[1], example["response"]],
            ]
            example["formatted_text"] = dialogue_template.get_prompt()
            example["prompt"] = temp_prompt
    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example

