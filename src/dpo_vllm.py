
from .vllm_server import VLLM
from multiprocessing import Process, Manager

class DPOInferenceVLLM:
    def __init__(self, model, ref_model, max_prompt_length = 512):
        self.model = model
        self.ref_model = ref_model
        self.max_prompt_length = max_prompt_length 
        self.engine = VLLM()

    def truncate_prompts(self, batch_prompts):
        # do a proper tokenizer based truncation
        ret = []
        for prompt in batch_prompts:
            new_prompt_tokens = prompt.split()[:self.max_prompt_length]
            new_prompt = " ".join(new_prompt_tokens)
            ret.append(new_prompt)
        return ret

    def inference_step(self, batch):
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

        chosen_batch, rejected_batch = [ex["text_chosen"] for ex in batch], [ex["text_rejected"] for ex in batch]
        # chosen_batch, rejected_batch = self.truncate_prompts(chosen_batch), self.truncate_prompts(rejected_batch)

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

        
        chosen_logprobs, chosen_ref_logprobs, rejected_logprobs, rejected_ref_logprobs = results['chosen_model']["tokens_logprobs"], results['chosen_ref_model']["tokens_logprobs"], results['rejected_model']["tokens_logprobs"], results['rejected_ref_model']["tokens_logprobs"]
        
        chosen_rewards, rejected_rewards = [], []
        for idx in range(len(chosen_logprobs)):
            chosen_logprob, chosen_ref_logprob, rejected_logprob, rejected_ref_logprob = \
                chosen_logprobs[idx], chosen_ref_logprobs[idx], rejected_logprobs[idx], rejected_ref_logprobs[idx]
            
            chosen_rewards.append(sum(chosen_logprob[1:]) - sum(chosen_ref_logprob[1:]) )
            rejected_rewards.append(sum(rejected_logprob[1:]) - sum(rejected_ref_logprob[1:]))
        
        return chosen_rewards, rejected_rewards
        

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

        chosen_batch = [ex["formatted_output"] for ex in batch]
        # chosen_batch, rejected_batch = self.truncate_prompts(chosen_batch), self.truncate_prompts(rejected_batch)

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

        chosen_logprobs, chosen_ref_logprobs = results['chosen_model']["tokens_logprobs"], results['chosen_ref_model']["tokens_logprobs"]

        chosen_rewards = []
        for idx in range(len(chosen_logprobs)):
            chosen_logprob, chosen_ref_logprob = \
                chosen_logprobs[idx], chosen_ref_logprobs[idx]

            chosen_rewards.append(sum(chosen_logprob[1:]) - sum(chosen_ref_logprob[1:]) )

        return chosen_rewards
