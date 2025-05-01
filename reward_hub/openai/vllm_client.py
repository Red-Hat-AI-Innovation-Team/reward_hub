import concurrent.futures
import time
import requests
from typing import List, Dict, Any, Optional



class vllmClient:
    def __init__(self, model_name, port):
        self.model_name = model_name
        self.port = port
    
    def make_chat_completion_request(  # noqa
        self,
        batch_messages: List[List[Dict]],
        max_completion_tokens: int = 4096,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.85,
        stop: Optional[List[str]] = None,
        num_workers: int = 40,
        prompt_logprobs: int = 0,
        add_generation_prompt: bool = True,
    ):
        """Sampling parameters for text generation.
        Args:
            batch_messages: List of message lists in OpenAI chat format
            max_completion_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to consider for sampling
            top_p: Cumulative probability threshold for token sampling
            stop: List of strings that stop generation when encountered
            num_workers: Number of parallel workers for processing requests
            prompt_logprobs: Number of logprobs to return for prompt tokens (0 to disable)
            add_generation_prompt: Whether to add generation prompt to the end of messages
        
        Returns:
            List of response objects from the vLLM server
        """
        port = self.port
        model_name = self.model_name
        endpoint=f"http://localhost:{port}/v1/chat/completions"

        headers = {"Content-type": "application/json"}

        def post_request(body):
            tries = 10
            for j in range(tries):  # 10 retries
                try:
                    response = requests.post(endpoint, json=body, headers=headers)
                    jres = response.json()
                    assert "choices" in jres, f"invalid outputs, resubmit request {jres}"
                except Exception as e:
                    if j < tries - 1:
                        print("Timeout connection to vLLM, waiting 5 secs before retry")
                        print(e)
                        time.sleep(1)
                        continue
                    else:
                        raise e
                break

            response_json = response.json()  # Assuming the response is in JSON format
            response_json["messages"] = body["messages"]  # Add 'input_prompt' to response JSON
            return response_json


        def post_requests(bodies):
            return [post_request(body) for body in bodies]

        results = []

        if not num_workers:
            num_workers = len(batch_messages)
        else:
            num_workers = min(num_workers, len(batch_messages))

        def create_chat_completion_request_instance(messages, stop):
            request_body = {
                "model": model_name,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens,
                "temperature": temperature,
                "stop": stop,
                "top_p": top_p,
                "top_k": top_k,
                "prompt_logprobs": prompt_logprobs,
                "add_generation_prompt": add_generation_prompt,
            }
            return request_body

        # Chunk prompt data into evenly distributed pieces to for each worker
        def serialize_worker(batch_messages, num_workers):
            """
            Returns chunked batch_prompts, where len(chunked_prompts) == num_workers,
            and ensures that when chunked_prompts are flattened, they equal batch_prompts.
            """
            assert num_workers <= len(
                batch_messages
            ), "Error, number of workers must be less than or equal to batch_size"

            # Calculate the size of each chunk
            chunk_size = len(batch_messages) // num_workers
            extra = len(batch_messages) % num_workers

            worker_chunks = []
            start = 0
            for i in range(num_workers):
                # Adjust the end index to include an extra item for the first few chunks if needed
                end = start + chunk_size + (1 if i < extra else 0)

                # Ensures that the correct stop sequence is passed to each prompt_instance
                chunk_ls = [create_chat_completion_request_instance(batch_messages[j], stop) for j in range(start, end)]
                worker_chunks.append(chunk_ls)
                start = end

            return worker_chunks

        worker_chunks = serialize_worker(batch_messages, num_workers)
        # test that the worker chunks can be decoded into the original sequence
        assert [
            x["messages"] for x in sum(worker_chunks, [])
        ] == batch_messages, "The flattened chunked prompts do not match the original batch prompts"

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all the request batches to be processed by post_requests function
            api_results = list(executor.map(post_requests, worker_chunks))
            for chunk_idx, chunk_results in enumerate(api_results):
                for item_idx, data in enumerate(chunk_results):
                    assert (
                        data["messages"] == worker_chunks[chunk_idx][item_idx]["messages"]
                    ), "output order is incorrect compared with submitted input order"
                    
                    if "prompt_logprobs" in data:
                        prompt_tokens, prompt_token_ids, prompt_logprobs = self.get_logprobs_and_tokens(data['prompt_logprobs'])
                    else:
                        prompt_tokens, prompt_token_ids, prompt_logprobs = None, None, None
                    results.append(
                        dict(
                            messages=worker_chunks[chunk_idx][item_idx]["messages"],
                            generated_text=data["choices"][0]["message"]['content'],
                            stop_reason=data["choices"][0]["stop_reason"],
                            finish_reason=data["choices"][0]["finish_reason"],
                            stop_sequence=worker_chunks[chunk_idx][item_idx]["stop"],
                            prompt_logprobs=prompt_logprobs,
                            prompt_tokens=prompt_tokens,
                            prompt_token_ids=prompt_token_ids,
                        )
                    )

        assert [
            x["messages"] for x in results
        ] == batch_messages, "The resulting prompt is different from original batch prompt"

        return results


    def request_logprobs(  # noqa
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool,
        num_workers: int = 40,
    ):
        raw_request_outputs = self.make_chat_completion_request(  # noqa
            batch_messages=batch_messages,
            temperature = 0.0,
            max_completion_tokens=1,
            num_workers=num_workers,
            prompt_logprobs = True,
            add_generation_prompt=add_generation_prompt,
        )
        all_prompt_tokens, all_prompt_token_ids, all_prompt_logprobs = self.post_process_logprobs(raw_request_outputs)
        return all_prompt_tokens, all_prompt_token_ids, all_prompt_logprobs

    def get_logprobs_and_tokens(self, prompt_logprobs_ob):
        tokens, token_ids, token_logprobs = [], [], []
        for ex in prompt_logprobs_ob:
            if ex == None:
                continue
            prompt_token_ob = list(ex.items())[0]
            token_ids.append(prompt_token_ob[0])
            token_logprobs.append(prompt_token_ob[1]['logprob'])
            tokens.append(prompt_token_ob[1]['decoded_token'])

        return tokens, token_ids, token_logprobs

    def post_process_logprobs(self, raw_outputs):
        for output in raw_outputs:
            assert "prompt_logprobs" in output, "prompt log prob is not available in the outputs"
        
        return [x["prompt_logprobs"] for x in raw_outputs], [x["prompt_tokens"] for x in raw_outputs], [x["prompt_token_ids"] for x in raw_outputs]

