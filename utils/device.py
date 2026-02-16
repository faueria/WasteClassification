"""
Device management utilities for M1 Pro MPS acceleration.

Provides unified device selection and memory management across training scripts.
"""
import os
import torch

# Enable CPU fallback for unsupported MPS operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def get_device() -> torch.device:
    """
    Determine optimal compute device with MPS priority.
    
    Returns:
        torch.device: MPS if available, CUDA if available, else CPU.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory_cache(device: torch.device) -> None:
    """
    Clear device memory cache to prevent OOM during long training runs.
    
    Args:
        device: Current compute device.
    """
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def get_memory_usage(device: torch.device) -> float:
    """
    Get current memory allocation in GB.
    
    Args:
        device: Current compute device.
        
    Returns:
        Memory usage in gigabytes, or 0.0 if unavailable.
    """
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def print_device_info() -> None:
    """Print diagnostic information about available compute devices."""
    device = get_device()
    print(f"Using device: {device}")
    
    if device.type == "mps":
        print("  Backend: Apple Metal Performance Shaders")
    elif device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  Warning: Running on CPU - training will be slow")