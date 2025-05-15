from multiprocessing import Process, Manager
import numpy as np


class DrSowConfig:
    def __init__(self, strong_model_name: str, strong_port: int, weak_model_name: str, weak_port: int):
        self.strong_model_name = strong_model_name
        self.strong_port = strong_port
        self.weak_model_name = weak_model_name
        self.weak_port = weak_port


class DrSow:
    def __init__(self, strong_client, weak_client, tokenizer, weak_tokenizer):
        self.strong_client = strong_client
        self.weak_client = weak_client
        self.tokenizer = tokenizer
        self.weak_tokenizer = weak_tokenizer

    def get_batch_logprobs(self, batch, num_workers=40, mask_logprob_special_tokens=False):
        """_summary_

        Args:
            batch (_type_): []
                {
                    "formatted_conv": the text to get logprobs for.
                     "prompt": the text to get logprobs for.
                }
            ]
        
        Return:
        rewards_chosen = 
            [
                3,
                4,
                77
            ]
        """
        chosen_messages, chosen_batch, prompt_batch = [ex["messages"] for ex in batch], [ex["formatted_conv"] for ex in batch], [ex["prompt"] for ex in batch]

        tokenized_prompt_batch = [self.tokenizer.encode(ex) for ex in prompt_batch]
        weak_tokenize_prompt_batch = [self.weak_tokenizer.encode(ex) for ex in prompt_batch]

        # for each item in the tokenized batch; find the index of last non-pad token
        strong_init_indices = [
            len(ex) - 1 for ex in tokenized_prompt_batch  # -1 to ensure correct matching
        ]

        weak_init_indices = [
            len(ex) - 1 for ex in weak_tokenize_prompt_batch # -1 to ensure correct matching
        ]

        def fetch_logprobs(batch, vllm_client, result_dict, key):
            tokens_logprobs, tokens, token_ids = vllm_client.request_logprobs(batch, add_generation_prompt=False, num_workers=num_workers)
            result_dict[key] = {
                "tokens": tokens,
                "tokens_logprobs": tokens_logprobs,
                "token_ids": token_ids
            }

        manager = Manager()
        results = manager.dict()

        processes = [
            Process(target=fetch_logprobs, args=(chosen_messages, self.strong_client, results, 'strong_model')),
            Process(target=fetch_logprobs, args=(chosen_messages, self.weak_client, results, 'weak_model')),
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        strong_tokens, weak_tokens = results['strong_model']["tokens"], results['weak_model']["tokens"]

        strong_logprobs, weak_logprobs = results['strong_model']["tokens_logprobs"], results['weak_model']["tokens_logprobs"]

        if mask_logprob_special_tokens:
            special_tokens = set(self.tokenizer.all_special_tokens)
        else:
            special_tokens = set([self.tokenizer.pad_token])

        final_reward_dicts = []

        for idx in range(len(strong_logprobs)):
            strong_logprob, weak_logprob = \
                np.array(strong_logprobs[idx]), np.array(weak_logprobs[idx])

            strong_init_idx = strong_init_indices[idx]
            weak_init_idx = weak_init_indices[idx]

            chosen_unmask_indices = [
                i for i, token in enumerate(strong_tokens[idx]) if i >= strong_init_idx and token not in special_tokens
            ]
            ref_chosen_unmask_indices = [
                i for i, token in enumerate(weak_tokens[idx]) if i >= weak_init_idx and token not in special_tokens
            ]


            drsow_reward = sum(strong_logprob[chosen_unmask_indices]) - sum(weak_logprob[ref_chosen_unmask_indices])

            strong_generated_tokens =  np.array(strong_tokens[idx])[chosen_unmask_indices]
            strong_token_logprobs =  np.array(strong_logprobs[idx])[chosen_unmask_indices]

            weak_generated_tokens =  np.array(weak_tokens[idx])[ref_chosen_unmask_indices]
            weak_token_logprobs =  np.array(weak_logprobs[idx])[ref_chosen_unmask_indices]
            
            # add single instances into the batch
            final_reward_dicts.append({
                "drsow_reward": drsow_reward,
                "avg_drsow_reward": drsow_reward / max(1, float(len(strong_generated_tokens))), # avoid divide by zero
                "strong_generated_tokens": strong_generated_tokens,
                "strong_logprobs": strong_token_logprobs,
                "weak_generated_tokens": weak_generated_tokens,
                "weak_logprobs": weak_token_logprobs,
            })

        return final_reward_dicts
