"""
Auto-configuration logic for optimal training parameters based on
GPU memory, model size, and dataset characteristics.
"""

import torch
import re
from typing import Dict, Any, Optional


def get_gpu_memory() -> float:
    """Get available GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0


def estimate_model_size(model_name: str) -> float:
    """Estimate model size in billions of parameters."""
    # Extract size from common naming patterns
    patterns = [
        r"(\d+\.?\d*)b",
        r"(\d+\.?\d*)B",
        r"-(\d+\.?\d+)-",
        r"-(\d+)-",
    ]
    
    model_lower = model_name.lower()
    for pattern in patterns:
        match = re.search(pattern, model_lower)
        if match:
            return float(match.group(1))
    
    # Default estimates based on model family
    if "qwen3-0.6" in model_lower or "qwen2.5-0.5" in model_lower:
        return 0.6
    elif "qwen3-1.7" in model_lower or "qwen2.5-1.5" in model_lower:
        return 1.7
    elif "qwen3-4" in model_lower or "qwen2.5-3" in model_lower:
        return 4.0
    elif "qwen3-8" in model_lower or "qwen2.5-7" in model_lower:
        return 8.0
    elif "qwen3-14" in model_lower or "qwen2.5-14" in model_lower:
        return 14.0
    elif "qwen3-32" in model_lower or "qwen2.5-32" in model_lower:
        return 32.0
    elif "llama-3.2-1" in model_lower or "llama-3.2-3" in model_lower:
        return 3.0
    elif "llama-3.1-8" in model_lower:
        return 8.0
    elif "llama-3.1-70" in model_lower:
        return 70.0
    elif "mistral-7" in model_lower or "mistral-small" in model_lower:
        return 7.0
    elif "gemma-2" in model_lower and "2b" in model_lower:
        return 2.0
    elif "gemma-2" in model_lower and "4b" in model_lower:
        return 4.0
    elif "gemma-2" in model_lower and "9b" in model_lower:
        return 9.0
    elif "gemma-2" in model_lower and "27b" in model_lower:
        return 27.0
    elif "phi-4" in model_lower:
        return 14.0
    elif "phi-3" in model_lower and "mini" in model_lower:
        return 3.8
    elif "phi-3" in model_lower and "small" in model_lower:
        return 7.0
    elif "phi-3" in model_lower and "medium" in model_lower:
        return 14.0
    
    return 7.0  # Default to 7B if unknown


def get_optimal_config(
    model_name: str,
    dataset_stats: Optional[Dict[str, Any]] = None,
    gpu_memory: Optional[float] = None,
    load_in_4bit: bool = True
) -> Dict[str, Any]:
    """
    Analyze model size, dataset, and available VRAM to return optimal training config.
    
    Args:
        model_name: HuggingFace model name or path
        dataset_stats: Dictionary with 'avg_length', 'max_length', 'num_examples'
        gpu_memory: Available GPU memory in GB (auto-detected if None)
        load_in_4bit: Whether using 4-bit quantization
        
    Returns:
        Dictionary with optimal: batch_size, lora_r, max_seq_length, learning_rate, num_epochs
    """
    if gpu_memory is None:
        gpu_memory = get_gpu_memory()
    
    if dataset_stats is None:
        dataset_stats = {"avg_length": 512, "max_length": 2048, "num_examples": 1000}
    
    model_size = estimate_model_size(model_name)
    
    # Determine LoRA rank based on model size
    if model_size < 2:
        lora_r = 32
        lora_alpha = 64
    elif model_size < 8:
        lora_r = 64
        lora_alpha = 128
    else:
        lora_r = 64
        lora_alpha = 128
    
    # Determine batch size based on VRAM and model size
    effective_vram = gpu_memory * (2.0 if load_in_4bit else 0.5)  # 4-bit uses ~25% memory
    
    if model_size < 2:
        base_batch = 4
    elif model_size < 8:
        base_batch = 2
    else:
        base_batch = 1
    
    # Adjust for available VRAM
    if effective_vram < 8:
        batch_size = max(1, base_batch // 2)
    elif effective_vram < 16:
        batch_size = base_batch
    elif effective_vram < 24:
        batch_size = min(4, base_batch * 2)
    else:
        batch_size = min(8, base_batch * 4)
    
    # Set max_seq_length based on dataset (95th percentile + padding)
    avg_len = dataset_stats.get("avg_length", 512)
    max_len = dataset_stats.get("max_length", 2048)
    
    # Use 95th percentile estimate or cap at reasonable limits
    estimated_95th = min(int(avg_len * 1.5), max_len)
    
    if estimated_95th <= 256:
        max_seq_length = 512
    elif estimated_95th <= 512:
        max_seq_length = 1024
    elif estimated_95th <= 1024:
        max_seq_length = 2048
    elif estimated_95th <= 2048:
        max_seq_length = 4096
    else:
        max_seq_length = min(8192, ((estimated_95th // 1024) + 1) * 1024)
    
    # Adjust sequence length based on VRAM
    if effective_vram < 8 and max_seq_length > 2048:
        max_seq_length = 2048
    elif effective_vram < 16 and max_seq_length > 4096:
        max_seq_length = 4096
    
    # Learning rate based on dataset size
    num_examples = dataset_stats.get("num_examples", 1000)
    if num_examples < 100:
        learning_rate = 5e-4
        num_epochs = 10
    elif num_examples < 1000:
        learning_rate = 2e-4
        num_epochs = 5
    elif num_examples < 10000:
        learning_rate = 1e-4
        num_epochs = 3
    else:
        learning_rate = 5e-5
        num_epochs = 2
    
    # Gradient accumulation steps to maintain effective batch size of ~32
    gradient_accumulation_steps = max(1, 32 // batch_size)
    
    # Warmup steps
    warmup_steps = min(100, max(10, num_examples // (batch_size * gradient_accumulation_steps * 10)))
    
    # Logging and saving steps
    logging_steps = max(1, num_examples // (batch_size * gradient_accumulation_steps * 20))
    save_steps = max(10, num_examples // (batch_size * gradient_accumulation_steps * 5))
    
    return {
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "max_seq_length": max_seq_length,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "warmup_steps": warmup_steps,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": 3407,
    }


def validate_config(config: Dict[str, Any], gpu_memory: float, model_size: float) -> None:
    """
    Validate that the configuration fits within available resources.
    Raises ValueError with helpful suggestions if not.
    """
    # Estimate memory usage (rough approximation)
    # Base model in 4-bit: ~0.5GB per billion params
    # LoRA adapters: ~0.01GB per billion params per rank
    # Activations: depends on batch_size and seq_length
    
    batch_size = config["batch_size"]
    max_seq = config["max_seq_length"]
    lora_r = config["lora_r"]
    
    base_model_mem = model_size * 0.5  # 4-bit quantized
    lora_mem = model_size * 0.01 * lora_r / 16  # Scaled by rank
    activation_mem = batch_size * max_seq * 0.001  # Rough estimate
    optimizer_mem = lora_mem * 3  # Adam states
    
    estimated_total = base_model_mem + lora_mem + activation_mem + optimizer_mem
    # Add 20% buffer
    required_memory = estimated_total * 1.2
    
    if required_memory > gpu_memory:
        suggested_batch = max(1, batch_size // 2)
        suggested_seq = max(256, max_seq // 2)
        
        raise ValueError(
            f"GPU memory may be insufficient! Estimated need: {required_memory:.1f}GB, "
            f"Available: {gpu_memory:.1f}GB\n\n"
            f"Try one or more of these solutions:\n"
            f"  1. Reduce batch_size: {batch_size} → {suggested_batch}\n"
            f"  2. Reduce max_seq_length: {max_seq} → {suggested_seq}\n"
            f"  3. Ensure load_in_4bit=True\n"
            f"  4. Use a smaller model (currently ~{model_size:.1f}B params)\n"
            f"  5. Use a GPU with more memory"
        )
    
    return True
