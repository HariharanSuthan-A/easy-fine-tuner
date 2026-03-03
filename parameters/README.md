# EasyFinetuner Parameters Guide

This guide explains key parameters for training models and pushing them to HuggingFace Hub using easyfinetuner.

---

## 📤 Push to HuggingFace Hub

After training, you can export your model to GGUF format and push it directly to HuggingFace Hub.

### Option 1: Export & Push in One Call (Recommended)

```python
from easyfinetuner import FineTuner

# Train your model
tuner = FineTuner("unsloth/Qwen3-1.7B")
tuner.train(dataset=data, output_dir="./model")

# Export GGUF and push to HuggingFace Hub
tuner.export_gguf(
    output_path="model.gguf",
    quantization="q4_k_m",
    upload_to_hub=True,
    hf_repo_id="yourusername/my-finetuned-model",  # Your HF repo
    hf_token="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"       # Your HF token
)
```

### Option 2: Push LoRA Adapters (Not GGUF)

```python
from easyfinetuner import FineTuner
from huggingface_hub import login

# Login first
login(token="hf_xxxxxxxxxxxxxxxxxxxxxxxxx")

# Train
tuner = FineTuner("unsloth/Qwen3-1.7B")
tuner.train(dataset=data, output_dir="./model")

# Push just the LoRA adapters (smaller, ~10-50MB)
tuner.save("./model", save_adapter_only=True)

# Upload to Hub
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="./model",
    repo_id="yourusername/my-lora-adapters",
    create_repo=True,
    repo_type="model"
)
print(f"Pushed to: https://huggingface.co/yourusername/my-lora-adapters")
```

### Option 3: Using GGUFExporter Directly

```python
from easyfinetuner import FineTuner, GGUFExporter

# Train
tuner = FineTuner("unsloth/Qwen3-1.7B")
tuner.train(dataset=data)

# Export first
output_path = tuner.export_gguf(quantization="q4_k_m")

# Push separately with custom model card
model_card = {
    "model_name": "My Fine-tuned Model",
    "base_model": "unsloth/Qwen3-1.7B",
    "quantization": "q4_k_m",
    "dataset_size": 100,
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "lora_r": 32,
    "final_loss": 0.5,
    "max_seq_length": 2048,
    "repo_id": "yourusername/my-model",
}

GGUFExporter.upload_to_hub(
    model_path=output_path,
    repo_id="yourusername/my-model",
    token="hf_xxxxxxxxxxxxxxxxxxxxxxxxx",
    private=False,  # Set True for private repo
    model_card=model_card
)
```

### 🔑 Get Your HuggingFace Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **"write"** permission
3. Copy the token (starts with `hf_`)

> **Note:** In Google Colab, you can also use interactive login:
> ```python
> from huggingface_hub import login
> login()  # Will prompt for token interactively
> ```

---

## 💾 Save Steps

`save_steps` controls how often checkpoints are saved during training.

### Auto (Default - Recommended)

By default, `save_steps` is auto-calculated based on dataset size:

```python
save_steps = max(10, num_examples // (batch_size * gradient_accumulation_steps * 5))
```

**Auto behavior examples:**

| Dataset Size | Approximate Save Steps |
|-------------|----------------------|
| 100 examples | ~10 steps |
| 1,000 examples | ~50 steps |
| 10,000 examples | ~100 steps |

**Use auto (default):**
```python
tuner.train(dataset=data, output_dir="./model")
```

### Manual Configuration

Set `save_steps` manually to control checkpoint frequency:

```python
# Save every 50 steps
tuner.train(
    dataset=data,
    output_dir="./model",
    save_steps=50
)

# Save every 500 steps (for large datasets)
tuner.train(
    dataset=data,
    output_dir="./model",
    save_steps=500
)

# Save less frequently (for small datasets)
tuner.train(
    dataset=data,
    output_dir="./model",
    save_steps=10
)
```

### Related Parameters

| Parameter | Auto/Default | Description |
|-----------|-------------|-------------|
| `save_steps` | Auto | Steps between checkpoints |
| `logging_steps` | Auto | Steps between console logs |
| `save_total_limit` | 3 | Maximum checkpoints to keep |

### Full Custom Control Example

```python
tuner.train(
    dataset=data,
    output_dir="./model",
    num_epochs=5,
    batch_size=2,
    gradient_accumulation_steps=4,
    save_steps=100,        # Save every 100 steps
    logging_steps=10,      # Log every 10 steps
    warmup_steps=50,       # Custom warmup
)
```

---

## ⚙️ All Training Parameters

Complete list of all parameters supported by `FineTuner.train()`:

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset` | str, List, Dataset, DataFrame | Training data (file path, list of dicts, DataFrame, or Dataset) |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | `"outputs"` | Directory to save model checkpoints |
| `num_epochs` | int or `"auto"` | `"auto"` | Number of training epochs |
| `max_seq_length` | int or `"auto"` | `"auto"` | Maximum sequence length for model |

### Training Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float or `"auto"` | `"auto"` | Learning rate (auto: 5e-5 to 5e-4 based on dataset size) |
| `batch_size` | int or `"auto"` | `"auto"` | Batch size per device (auto: based on VRAM) |
| `gradient_accumulation_steps` | int or `"auto"` | `"auto"` | Steps to accumulate gradients before updating weights |
| `warmup_steps` | int or `"auto"` | `"auto"` | Number of warmup steps for learning rate scheduler |
| `weight_decay` | float | `0.01` | Weight decay for regularization |
| `lr_scheduler_type` | str | `"cosine"` | Learning rate scheduler type |

### LoRA (Low-Rank Adaptation) Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_r` | int or `"auto"` | `"auto"` | LoRA rank (auto: 32 for small models, 64 for larger) |
| `lora_alpha` | int | auto-calculated | LoRA alpha (typically 2x lora_r) |
| `lora_dropout` | float | `0` | Dropout probability for LoRA layers |

### Checkpoint & Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_steps` | int or `None` | `None` (auto) | Steps between checkpoints |
| `logging_steps` | int or `None` | `None` (auto) | Steps between console logs |
| `save_total_limit` | int | `3` | Maximum number of checkpoints to keep |

### Data & Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validation_split` | float | `0.1` | Fraction of data for validation (0.0-1.0) |
| `template` | str | `"auto"` | Prompt template ("auto", "alpaca", "chatml", "plain") |

### Advanced Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | `3407` | Random seed for reproducibility |
| `resume_from_checkpoint` | str or `None` | `None` | Path to checkpoint to resume training from |
| `load_in_4bit` | bool | `True` | Use 4-bit quantization to save memory |
| `device_map` | str | `"auto"` | Device mapping strategy for model placement |
| `**kwargs` | any | - | Additional arguments passed to TrainingArguments |

### Auto-Configuration Rules

When parameters are set to `"auto"`, they are determined as follows:

| Parameter | Small Model (<2B) | Medium (2-8B) | Large (>8B) |
|-----------|----------------|---------------|-------------|
| `lora_r` | 32 | 64 | 64 |
| `batch_size` | 4 | 2 | 1 |
| `learning_rate` | 5e-4 (<100 ex) | 2e-4 (100-1k ex) | 5e-5 (>10k ex) |
| `num_epochs` | 10 (<100 ex) | 5 (100-1k ex) | 2 (>10k ex) |

### Complete Example with All Parameters

```python
from easyfinetuner import FineTuner

tuner = FineTuner(
    model_name="unsloth/Qwen3-1.7B",
    max_seq_length=2048,
    load_in_4bit=True,
    template="auto",
)

stats = tuner.train(
    dataset="data.json",
    output_dir="./my_model",
    
    # Training configuration
    num_epochs=3,
    batch_size=2,
    learning_rate=2e-4,
    
    # LoRA configuration
    lora_r=32,
    
    # Gradient handling
    gradient_accumulation_steps=4,
    warmup_steps=50,
    weight_decay=0.01,
    
    # Checkpointing
    save_steps=100,
    logging_steps=10,
    
    # Data & validation
    validation_split=0.1,
    
    # Reproducibility
    seed=42,
    
    # Resume training (optional)
    # resume_from_checkpoint="./my_model/checkpoint-500",
)
```

---

## 📚 Additional Resources

- [Main Repository](https://github.com/HariharanSuthan-A/easy-fine-tuner)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)

---

*Last updated: March 2025*
