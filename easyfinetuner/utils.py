"""
Utility functions for device detection, memory checks, and helpers.
"""

import os
import re
import torch
import warnings
from typing import Optional, Dict, Any


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "device_name": None,
        "total_memory_gb": 0.0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info["current_device"] = device
        info["device_name"] = props.name
        info["total_memory_gb"] = props.total_memory / (1024**3)
    
    return info


def print_device_info():
    """Print formatted device information."""
    info = get_device_info()
    
    print("=" * 50)
    print("Device Information")
    print("=" * 50)
    
    if info["cuda_available"]:
        print(f"GPU: {info['device_name']}")
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"GPU Memory: {info['total_memory_gb']:.2f} GB")
        print(f"Device Count: {info['device_count']}")
    else:
        print("No CUDA GPU detected. Training will use CPU (very slow!)")
        warnings.warn("CUDA not available. Fine-tuning on CPU is not recommended.")
    
    print("=" * 50)


def check_memory_available(required_gb: float) -> bool:
    """Check if sufficient GPU memory is available."""
    if not torch.cuda.is_available():
        return False
    
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    free = total - allocated
    
    return free >= required_gb


def get_memory_stats() -> Dict[str, float]:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
        "free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3),
    }


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def estimate_training_time(
    num_examples: int,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    seq_length: int,
    model_size_b: float
) -> str:
    """Estimate training time in human-readable format."""
    # Very rough estimates (tokens per second)
    # Small model (<2B): ~1000 tok/s
    # Medium (2-8B): ~500 tok/s
    # Large (>8B): ~200 tok/s
    
    if model_size_b < 2:
        tokens_per_sec = 1000
    elif model_size_b < 8:
        tokens_per_sec = 500
    else:
        tokens_per_sec = 200
    
    total_tokens = num_examples * num_epochs * seq_length
    effective_batch = batch_size * gradient_accumulation_steps
    steps = (num_examples // effective_batch) * num_epochs
    
    estimated_seconds = (total_tokens / tokens_per_sec) + (steps * 2)  # +2s per step overhead
    
    if estimated_seconds < 60:
        return f"{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        return f"{estimated_seconds/60:.1f} minutes"
    elif estimated_seconds < 86400:
        return f"{estimated_seconds/3600:.1f} hours"
    else:
        return f"{estimated_seconds/86400:.1f} days"


def sanitize_model_name(model_name: str) -> str:
    """Create a safe directory name from model name."""
    # Remove or replace unsafe characters
    safe = re.sub(r'[^\w\-_.]', '_', model_name)
    safe = re.sub(r'_+', '_', safe)
    return safe.strip('_')


def create_output_dir(base_dir: str, model_name: str, experiment_name: Optional[str] = None) -> str:
    """Create and return a unique output directory path."""
    safe_name = sanitize_model_name(model_name)
    
    if experiment_name:
        output_dir = os.path.join(base_dir, f"{safe_name}_{experiment_name}")
    else:
        # Find next available directory number
        counter = 1
        while True:
            output_dir = os.path.join(base_dir, f"{safe_name}_run{counter}")
            if not os.path.exists(output_dir):
                break
            counter += 1
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(output_dir: str, log_level: str = "INFO"):
    """Setup logging configuration."""
    import logging
    
    log_file = os.path.join(output_dir, "training.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("easyfinetuner")


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to JSON file."""
    import json
    
    # Convert any non-serializable values
    serializable = {}
    for k, v in config.items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            serializable[k] = v
        else:
            serializable[k] = str(v)
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    import json
    
    with open(config_path, 'r') as f:
        return json.load(f)


def print_training_config(config: Dict[str, Any]):
    """Pretty print training configuration."""
    print("\n" + "=" * 50)
    print("Training Configuration")
    print("=" * 50)
    
    key_order = [
        "model_name",
        "max_seq_length",
        "batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
        "learning_rate",
        "lora_r",
        "lora_alpha",
        "warmup_steps",
        "weight_decay",
        "output_dir",
    ]
    
    for key in key_order:
        if key in config:
            value = config[key]
            if isinstance(value, float):
                print(f"{key:30s}: {value:.2e}" if value < 0.001 else f"{key:30s}: {value:.6f}")
            else:
                print(f"{key:30s}: {value}")
    
    print("=" * 50 + "\n")


class ProgressCallback:
    """Simple progress callback for training."""
    
    def __init__(self, total_steps: int, desc: str = "Training"):
        self.total_steps = total_steps
        self.current_step = 0
        self.desc = desc
        self.start_time = None
        
    def on_train_begin(self):
        import time
        self.start_time = time.time()
        print(f"\n{self.desc} started...")
        
    def on_step_end(self, step: int, loss: float, learning_rate: float):
        self.current_step = step
        progress = step / self.total_steps * 100
        
        import time
        elapsed = time.time() - self.start_time
        if step > 0:
            time_per_step = elapsed / step
            remaining = time_per_step * (self.total_steps - step)
            eta = f", ETA: {remaining/60:.1f}m"
        else:
            eta = ""
        
        if step % 10 == 0 or step == self.total_steps:
            print(f"  Step {step}/{self.total_steps} ({progress:.1f}%) - Loss: {loss:.4f}, LR: {learning_rate:.2e}{eta}")
    
    def on_train_end(self, final_loss: float):
        import time
        elapsed = time.time() - self.start_time
        print(f"\n{self.desc} complete!")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Total time: {elapsed/60:.1f} minutes")


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are installed."""
    deps = {}
    
    required = [
        "torch",
        "transformers",
        "datasets",
        "unsloth",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes",
    ]
    
    for dep in required:
        try:
            __import__(dep)
            deps[dep] = True
        except ImportError:
            deps[dep] = False
    
    return deps


def validate_dataset_format(sample: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate that a dataset sample has the required format.
    Returns (is_valid, message)
    """
    # Check for common formats
    valid_keys = [
        {"input", "output"},
        {"instruction", "input", "output"},
        {"question", "answer"},
        {"prompt", "completion"},
        {"text"},
        {"messages"},
    ]
    
    sample_keys = set(sample.keys())
    
    for valid in valid_keys:
        if valid.issubset(sample_keys):
            return True, f"Valid format detected: {valid}"
    
    # Check for common mistakes
    if "inputs" in sample_keys or "outputs" in sample_keys:
        return False, "Found 'inputs'/'outputs' - use 'input'/'output' (singular) instead"
    
    if "instruct" in sample_keys:
        return False, "Found 'instruct' - use 'instruction' instead"
    
    available = ", ".join(f"'{k}'" for k in sample_keys)
    expected = "; ".join(str(v) for v in valid_keys)
    
    return False, f"Unknown format. Keys found: {available}. Expected one of: {expected}"
