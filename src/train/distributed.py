"""Distributed process group setup / teardown."""

import torch
import torch.distributed as dist


def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    print(f"[Rank {rank}] Starting distributed setup...")
    
    try:
        # Set CUDA device first
        torch.cuda.set_device(rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=args.dist_backend,
            rank=rank,
            world_size=world_size
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


