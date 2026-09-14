"""Distributed process group setup / teardown."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    print(f"[Rank {rank}] Starting distributed setup...")
    
    try:
        # Set CUDA device first
        torch.cuda.set_device(rank)

        # Initialize process group. Passing device_id pins the NCCL
        # communicator to this rank's GPU (avoids the "Guessing device ID"
        # warning and possible hangs with heterogeneous rank->GPU mapping)
        # and lets barrier() use the current device without warning.
        # The timeout is raised well above the default 10 min: validation
        # shards are unbalanced (held-out punches vary in tile count), so
        # fast ranks can wait >10 min at the end-of-epoch allreduce for the
        # slowest rank before the NCCL watchdog kills the run.
        dist.init_process_group(
            backend=args.dist_backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=60),
            device_id=torch.device(f'cuda:{rank}') if torch.cuda.is_available() else None,
        )
        
        # Synchronize all processes
        dist.barrier()
        
        if rank == 0:
            print(f"Distributed training initialized with {world_size} GPUs")
            
    except Exception as e:
        print(f"[Rank {rank}] Distributed setup failed: {e}")
        raise


def cleanup():
    """
    Clean up the distributed environment.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception as e:
        print(f"Warning during cleanup: {e}")


