import torch
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"RAM used: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
    print(f"CUDA memory cached: {torch.cuda.memory_reserved() / (1024 * 1024):.2f} MB")