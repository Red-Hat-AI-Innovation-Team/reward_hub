import logging
from tqdm.auto import tqdm
from datasets import load_dataset
from rich.logging import RichHandler
from huggingface_hub import AsyncInferenceClient, InferenceClient
import argparse
import time 
import os
import asyncio


logger = logging.getLogger("rich")
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])


def get_dataset_shard(args, dataset):
    total_shards = args.total_shards if args.total_shards is not None else args.num_gpus
    shard_size = dataset.num_rows // total_shards
    shard_start, shard_end = args.shard * shard_size, min(args.shard * shard_size + shard_size, dataset.num_rows)
    logger.info(f"Running shard {args.shard}. Picked range {shard_start} - {shard_end}")
    dataset = dataset.select(range(shard_start, shard_end))

    return dataset
    

def main(args):

    logger.info("Setting up Client")
    # if args.do_sync:
    # client = InferenceClient(f"http://localhost:{args.port}")
    # else:
    client = AsyncInferenceClient(f"http://0.0.0.0:{args.port}")
    
    async def generate_samples(prompt, args):
        tasks = [
            client.text_generation(
                prompt,
                do_sample=args.do_sample,
                stop_sequences=args.stop_sequences,
                temperature=0.7,
                top_p=0.85,
                top_k=50,
                max_new_tokens=args.max_new_tokens
            ) for _ in range(args.nsample)
        ]
        return await asyncio.gather(*tasks)

    logger.info("Loading dataset")
    dataset = load_dataset('json', data_files=args.data_path, split='train')

    if args.subsample is not None:
        logger.info(f"Subsampling dataset to {args.subsample}")
        dataset = dataset.select(range(args.subsample))

    if args.shard is not None:
        logger.info("Sharding dataset")
        dataset = get_dataset_shard(args, dataset)

    if args.do_sample:
        logger.info(f"Generation parameters: \ndo_sample={args.do_sample}, \
                    \ntemperature={args.temperature}, \ntop_p={args.top_p}, \
                    \ntop_k={args.top_k}, repetition_penalty={args.repetition_penalty}, \
                    \nmax_new_tokens={args.max_new_tokens}")
    else:
        logger.info(f"Generation parameters: do_sample={args.do_sample}, repetition_penalty={args.repetition_penalty}, max_new_tokens={args.max_new_tokens}")
    logger.info("Running generation")

    training_bar = tqdm(total=len(dataset))
    for i in range(0, len(dataset), args.batch_size):
        logger.info(f"Generating for batch {i}")
        start = time.time()
        data_sub = dataset.select(range(i, min(i+args.batch_size, len(dataset))))
        if os.path.isfile(f"{args.output_path}/{args.exp_name}/shard{args.shard}_batch{i}.jsonl") or i < args.start_from:
            print("Skipping batch. Already exist", f"{args.output_path}/{args.exp_name}/shard{args.shard}_batch{i}.jsonl")
            training_bar.update(args.batch_size)
            continue
        # if args.do_sync:
        logger.info("Running synchronous generation")
        responses = []
        for j,sample in enumerate(data_sub):
            prompt = sample[args.prompt_col]

            try:
                if args.do_sample:
                    result = asyncio.run(generate_samples(prompt, args))
                    responses.append(result)
                else:
                    result = asyncio.run(generate_samples(prompt, args))
                    responses.append(result)
            except Exception as e:
                logger.error(f"Error in generation: {e}")
                responses.append(["-1"]*args.nsample)
            training_bar.update(1)
            
            if i < 100 and j % 10 == 0:
                logger.info(f"sample {j} \n\
                            prompt: {prompt}, \n\
                            response: {responses[-1][0]}")
        
        if args.response_col not in data_sub.column_names:
            data_sub = data_sub.add_column(args.response_col, responses)
        else:
            col_name = args.response_col + "_new"
            data_sub = data_sub.add_column(col_name, responses)
        logger.info(f"Generated for batch {i} in {time.time() - start:.2f}s")
        data_sub.to_json(f"{args.output_path}/{args.exp_name}/shard{args.shard}_batch{i}.jsonl", orient='records', lines=True)
        
        # else:
        #     logger.info("Running asynchronous generation")
        #     responses = [client.text_generation(prompt, do_sample=args.do_sample, stop_sequences=args.stop_sequences) for prompt in data_sub[args.prompt_col]]

        #     for i, r in enumerate(responses): 
        #         data_sub['forca_response'][i] = r.generated_text

        #     data_sub.to_json(f"{args.output_path}/{args.exp_name}/batch_{i}.json", orient='records', lines=True)

    logger.info("Finished generation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/forca/Documents/rlhf/data/forca.jsonl")
    parser.add_argument("--output_path", type=str, default="/home/forca/Documents/rlhf/data/forca.jsonl")
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--nsample", type=int, default=1)
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--prompt_col", type=str, default="prompt")
    parser.add_argument("--response_col", type=str, default="response")
    parser.add_argument("--do_sample", 
                        action="store_true",
                        help="Whether to use sampling or greedy decoding.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=3)
    # parser.add_argument("--do_sync", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--total_shards", type=int, default=None)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--start_from",
                        type=int,
                        default=0,
                        help="Starting index for generation. Useful for resuming generation, or running multiple instances of generation in parallel.")
    parser.add_argument("--stop_sequences", type=str, nargs="+", default=None)
    
    args = parser.parse_args()
    #pretty print args
    logger.info(args)
    main(args)