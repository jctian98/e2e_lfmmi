import time
import torch.distributed as dist


def step_print(ctx, flush=False):
    tmark = time.asctime(time.localtime(time.time()))
    rank = dist.get_rank()
    print(f"{tmark} | rank: {rank} | {ctx}", flush=flush)
