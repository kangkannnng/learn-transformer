import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim

from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix} peak memory: {torch.cuda.max_memory_allocated(device) // 1e6} MB")

def example(rank, world_size, use_zero):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
    print_peak_memory(f"Max memory allocated after creating local model", rank)
    
    ddp_model = DDP(model, device_ids=[rank])
    print_peak_memory(f"Max memory allocated after creating DDP model", rank)

    loss_fn = nn.MSELoss()
    if use_zero:
        optimizer = ZeroRedundancyOptimizer(
            ddp_model.parameters(),
            optim.Adam,
            lr=0.01,
        )
    else:
        optimizer = optim.Adam(ddp_model.parameters(), lr=0.01)
    
    outputs = ddp_model(torch.randn(20, 2000).to(rank))
    labels = torch.randn(20, 2000).to(rank)

    loss_fn(outputs, labels).backward()
    print_peak_memory(f"Max memory allocated before zero step", rank)

    optimizer.step()
    print_peak_memory(f"Max memory allocated after zero step", rank)

    print(f"params sum is {sum(model.parameters()).sum()}")

def main():
    world_size = 2
    print("Using Zero")
    mp.spawn(example, args=(world_size, True), nprocs=world_size, join=True)
    print("Not using Zero")
    mp.spawn(example, args=(world_size, False), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()