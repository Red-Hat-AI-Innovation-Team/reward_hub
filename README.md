# preference-generator


### Ray Distributed Cluster

We use the following setup due to the default hardware on a VELA node; 
You should update according to your hardware environment. 

VELA node typically has 2 sockets, with each socket 20 physical gpus. Hence num-cpus for ray cluster needs to be less than 40.
The high compute setup of the VELA allows 2 threads per cpu, so we should set total number of theads less than 80.

I have 4 scripts to run in the ray cluster, so per script threads should be less than 20. 


```python
ray disable-usage-stats
export OPENBLAS_NUM_THREADS=18
export OMP_NUM_THREADS=18
ray start --head --num-cpus=32 --num-gpus=8
```

### Launch VLLM Sampling

It will automatically shard data according to num_shards and the current index. 
This is very useful for multi-node distribtued compute. 

Server will be automatically launched for the job. 
```
bash run_inference.sh SHARD_NUMS SHARD_IDX
```


### Launch reward annotation
This assumes a full node of 8 gpus. Will automatically launch server and run annotation jobs; 
Outputs are saved at the same directory as the input_data. 
```
run_annotate.sh input_data_path
```

Single-thread testing:
```
run_annotate_test.sh input_data_path
```

