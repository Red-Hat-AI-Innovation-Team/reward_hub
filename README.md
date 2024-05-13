# preference-generator



### Install flash-attention to speedup
```bash
python -m pip install flash-attn --no-build-isolation
```

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