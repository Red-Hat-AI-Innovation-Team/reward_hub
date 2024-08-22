
from .vllm_server import VLLM
from multiprocessing import Process, Manager
import torch 
from transformers import AutoTokenizer
import numpy as np


class DPOInferenceVLLM:
    def __init__(self, model, ref_model, tokenizer, max_prompt_length = 512):
        self.model = model
        self.ref_model = ref_model
        self.max_prompt_length = max_prompt_length 
        self.engine = VLLM()
        self.tokenizer = tokenizer
        self.ref_tokenizer = AutoTokenizer.from_pretrained(ref_model['model_name'])

    def monolithic_inference_step(self, batch):
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

        tokenized_prompt_batch = [self.tokenizer.encode(ex) for ex in prompt_batch]
        ref_tokenize_prompt_batch = [self.ref_tokenizer.encode(ex) for ex in prompt_batch]

        # for each item in the tokenized batch; find the index of last non-pad token
        generation_first_token_indices = [
            len(ex) for ex in tokenized_prompt_batch
        ]

        ref_generation_first_token_indices = [
            len(ex) for ex in ref_tokenize_prompt_batch
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

        chosen_rewards_batch = []
        final_reward_dicts = []

        for idx in range(len(chosen_logprobs)):
            chosen_logprob, chosen_ref_logprob = \
                np.array(chosen_logprobs[idx]), np.array(chosen_ref_logprobs[idx])

            response_start_idx = generation_first_token_indices[idx]
            ref_response_start_idx = ref_generation_first_token_indices[idx]

            chosen_unmask_indices = [
                i for i, token in enumerate(chosen_tokens[idx]) if i >= response_start_idx and token != PAD_TOKEN
            ]
            ref_chosen_unmask_indices = [
                i for i, token in enumerate(chosen_ref_tokens[idx]) if i >= ref_response_start_idx and token != PAD_TOKEN
            ]

            chosen_rw = sum(chosen_logprob[chosen_unmask_indices]) - sum(chosen_ref_logprob[ref_chosen_unmask_indices])

            generated_tokens_pref =  np.array(chosen_tokens[idx])[chosen_unmask_indices]
            generated_logprobs_pref =  np.array(chosen_logprobs[idx])[chosen_unmask_indices]

            generated_tokens_ref =  np.array(chosen_ref_tokens[idx])[ref_chosen_unmask_indices]
            generated_logprobs_ref =  np.array(chosen_ref_logprobs[idx])[ref_chosen_unmask_indices]
            
            # add single instances into the batch
            final_reward_dicts.append({
                "log_ratio_reward": chosen_rw,
                "generated_tokens_pref": generated_tokens_pref,
                "logprobs_pref": generated_logprobs_pref,
                "generated_tokens_ref": generated_tokens_ref,
                "logprobs_ref": generated_logprobs_ref,
            })
        # the final rewards output needs to be an object that contains
        # generated tokens and their respective log-probs
        # for both the pref_model and the ref_model

        return final_reward_dicts
