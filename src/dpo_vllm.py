
from .vllm_server import VLLM
from multiprocessing import Process, Manager
import torch 
import numpy as np


class DPOInferenceVLLM:
    def __init__(self, model, ref_model, tokenizer, max_prompt_length = 512):
        self.model = model
        self.ref_model = ref_model
        self.max_prompt_length = max_prompt_length 
        self.engine = VLLM()
        self.tokenizer = tokenizer

    def truncate_prompt(self, prompt, max_prompt_length=2048, truncate_method="keep_right"):
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) <= max_prompt_length:
            return prompt

        if truncate_method == "middle":
            keep_tokens = int(max_prompt_length//2)
            truncated_tokens = tokens[:keep_tokens] + tokens[-keep_tokens:]
        elif truncate_method == "keep_left":
            truncated_tokens = tokens[:max_prompt_length]
        elif truncate_method == "keep_right":
            truncated_tokens == tokens[-max_prompt_length:]

        return self.tokenizer.decode(truncated_tokens)

    def inference_step(self, batch, average_log_prob=False):
        """_summary_

        Args:
            batch (_type_): _description_
        
        Return:
        rewards_chosen = 
            [
                3,
                4,
                77
            ]
        rewards_rejected = 
            [
                1,
                2,
                44,
            ]
        """

        chosen_batch, rejected_batch, prompt_batch = [ex["text_chosen"] for ex in batch], [ex["text_rejected"] for ex in batch], [ex["prompt"] for ex in batch]
        # chosen_batch, rejected_batch = self.truncate_prompts(chosen_batch), self.truncate_prompts(rejected_batch)


        tokenized_prompt_batch = [self.tokenizer.encode(ex) for ex in prompt_batch]

        # for each item in the tokenized batch; find the index of last non-pad token
        generation_first_token_indices = [
            len(ex) for ex in tokenized_prompt_batch
        ]


        def fetch_logprobs(batch, model_name, port, result_dict, key):
            tokens, tokens_logprobs = self.engine.vllm_request_logprobs(batch, model_name=model_name, port=port)
            result_dict[key] = {
                "tokens": tokens,
                "tokens_logprobs": tokens_logprobs
            }

        manager = Manager()
        results = manager.dict()
        

        processes = [
            Process(target=fetch_logprobs, args=(chosen_batch, self.model["model_name"], self.model["port"], results, 'chosen_model')),
            Process(target=fetch_logprobs, args=(chosen_batch, self.ref_model["model_name"], self.ref_model["port"], results, 'chosen_ref_model')),
            Process(target=fetch_logprobs, args=(rejected_batch, self.model["model_name"], self.model["port"], results, 'rejected_model')),
            Process(target=fetch_logprobs, args=(rejected_batch, self.ref_model["model_name"], self.ref_model["port"], results, 'rejected_ref_model'))
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        chosen_tokens, chosen_ref_tokens, rejected_tokens, rejected_ref_tokens = results['chosen_model']["tokens"], results['chosen_ref_model']["tokens"], results['rejected_model']["tokens"], results['rejected_ref_model']["tokens"]
        
        chosen_logprobs, chosen_ref_logprobs, rejected_logprobs, rejected_ref_logprobs = results['chosen_model']["tokens_logprobs"], results['chosen_ref_model']["tokens_logprobs"], results['rejected_model']["tokens_logprobs"], results['rejected_ref_model']["tokens_logprobs"]
        
        PAD_TOKEN = self.tokenizer.pad_token
        chosen_rewards, rejected_rewards = [], []
        for idx in range(len(chosen_logprobs)):
            chosen_logprob, chosen_ref_logprob, rejected_logprob, rejected_ref_logprob = \
                np.array(chosen_logprobs[idx]), np.array(chosen_ref_logprobs[idx]), np.array(rejected_logprobs[idx]), np.array(rejected_ref_logprobs[idx])
            response_start_idx = generation_first_token_indices[idx]

            # this thing produce a zero, in the case when response_start_idx is greater than the largest thing in the chosen_tokens
            chosen_unmask_indices = [
                i for i, token in enumerate(chosen_tokens[idx]) if i >= response_start_idx and token != PAD_TOKEN
            ]
            rej_unmask_indices = [
                i for i, token in enumerate(rejected_tokens[idx]) if i >= response_start_idx and token != PAD_TOKEN
            ]

            # sum(chosen_logprob[1:]) - sum(chosen_ref_logprob[1:])
            # sum(rejected_logprob[1:]) - sum(rejected_ref_logprob[1:])

            chosen_rw = sum(chosen_logprob[chosen_unmask_indices]) - sum(chosen_ref_logprob[chosen_unmask_indices])
            rejected_rw = sum(rejected_logprob[rej_unmask_indices]) - sum(rejected_ref_logprob[rej_unmask_indices])
            if average_log_prob:
                chosen_rw = chosen_rw/len(chosen_unmask_indices)
                rejected_rw = rejected_rw/len(rej_unmask_indices)
            
            chosen_rewards.append(chosen_rw)
            rejected_rewards.append(rejected_rw)

        return chosen_rewards, rejected_rewards


    def monolithic_inference_step(self, batch, average_log_prob=False):
        """_summary_

        Args:
            batch (_type_): [
                {
                    "text": the text to get logprobs for
                },
                {
                    "text": the text to get logprobs for
                },
                {
                    "text": the text to get logprobs for
                },
            ]
        
        Return:
        rewards_chosen = 
            [
                3,
                4,
                77
            ]
        """
        chosen_batch, prompt_batch = [ex["formatted_output"] for ex in batch], [ex["prompt"] for ex in batch]

        # set a very high truncation threshold for both prompt and output
        # TODO: the current truncation is fixed; modify in the future;
        # Reward annotation in the 2500 range should be enough
        # prompt_batch = [self.truncate_prompt(x , max_prompt_length=800) for x in prompt_batch]
        # chosen_batch = [self.truncate_prompt(x , max_prompt_length=1000) for x in chosen_batch]

        tokenized_prompt_batch = [self.tokenizer.encode(ex) for ex in prompt_batch]

        # for each item in the tokenized batch; find the index of last non-pad token
        generation_first_token_indices = [
            len(ex) for ex in tokenized_prompt_batch
        ]

        def fetch_logprobs(batch, model_name, port, result_dict, key):
            tokens, tokens_logprobs = self.engine.vllm_request_logprobs(batch, model_name=model_name, port=port)
            result_dict[key] = {
                "tokens": tokens,
                "tokens_logprobs": tokens_logprobs
            }

        manager = Manager()
        results = manager.dict()

        processes = [
            Process(target=fetch_logprobs, args=(chosen_batch, self.model["model_name"], self.model["port"], results, 'chosen_model')),
            Process(target=fetch_logprobs, args=(chosen_batch, self.ref_model["model_name"], self.ref_model["port"], results, 'chosen_ref_model')),
        ]
        
        for process in processes:
            process.start()

        for process in processes:
            process.join()

        chosen_tokens, chosen_ref_tokens = results['chosen_model']["tokens"], results['chosen_ref_model']["tokens"]

        chosen_logprobs, chosen_ref_logprobs = results['chosen_model']["tokens_logprobs"], results['chosen_ref_model']["tokens_logprobs"]
        PAD_TOKEN = self.tokenizer.pad_token
        
        
        chosen_rewards = []
        for idx in range(len(chosen_logprobs)):
            chosen_logprob, chosen_ref_logprob = \
                np.array(chosen_logprobs[idx]), np.array(chosen_ref_logprobs[idx])
        
            response_start_idx = generation_first_token_indices[idx]
            chosen_unmask_indices = [
                i for i, token in enumerate(chosen_tokens[idx]) if i >= response_start_idx and token != PAD_TOKEN
            ]

            chosen_rw = sum(chosen_logprob[chosen_unmask_indices]) - sum(chosen_ref_logprob[chosen_unmask_indices])

            if average_log_prob:
                chosen_rw = chosen_rw/len(chosen_unmask_indices)

            chosen_rewards.append(chosen_rw)
        return chosen_rewards
