"""
DDP (DistributedDataParallel) utility functions for multi-GPU training
DDP 多 GPU 训练工具函数
"""
import os
import torch
import torch.distributed as dist


def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialize distributed process group
    初始化分布式进程组

    Args:
        rank: Current process rank (0 to world_size-1)
        world_size: Total number of processes/GPUs
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed process group / 清理分布式进程组"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if current process is the main process (rank 0)"""
    return rank == 0


def print_rank0(msg: str, rank: int):
    """Print message only on rank 0"""
    if is_main_process(rank):
        print(msg)


def get_device(rank: int) -> torch.device:
    """Get device for current rank / 获取当前 rank 的设备"""
    return torch.device(f"cuda:{rank}")


def barrier():
    """Synchronize all processes / 同步所有进程"""
    if dist.is_initialized():
        dist.barrier()
