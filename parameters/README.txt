================================================================================
                        EASYFINETUNER PARAMETERS GUIDE
================================================================================

This document explains key parameters for training and pushing models to 
HuggingFace Hub using easyfinetuner.

================================================================================
1. PUSH TO HUGGINGFACE HUB (push_to_hub)
================================================================================

After training, you can export your model to GGUF format and push it directly 
to HuggingFace Hub.

----------------------------------------
OPTION 1: Export & Push in One Call
----------------------------------------

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

----------------------------------------
OPTION 2: Push LoRA Adapters (Not GGUF)
----------------------------------------

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

----------------------------------------
OPTION 3: Using GGUFExporter Directly
----------------------------------------

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

----------------------------------------
GET YOUR HUGGINGFACE TOKEN
----------------------------------------

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "write" permission
3. Copy the token (starts with "hf_")

Note: In Google Colab, you can also use:
    from huggingface_hub import login
    login()  # Will prompt for token interactively

================================================================================
2. SAVE STEPS (save_steps)
================================================================================

save_steps controls how often checkpoints are saved during training.

----------------------------------------
AUTO (Default - Recommended)
----------------------------------------

By default, save_steps is auto-calculated based on dataset size:

    save_steps = max(10, num_examples // (batch_size * gradient_accumulation_steps * 5))

Example auto behavior:
    - 100 examples  -> save every ~10 steps
    - 1000 examples -> save every ~50 steps  
    - 10000 examples -> save every ~100 steps

To use auto (simply don't specify save_steps):
    tuner.train(dataset=data, output_dir="./model")

----------------------------------------
MANUAL
----------------------------------------

You can manually set save_steps to control checkpoint frequency:

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

----------------------------------------
RELATED PARAMETERS
----------------------------------------

Parameter           | Auto/Default | Description
--------------------|--------------|-------------------------------------------
save_steps          | Auto         | Steps between checkpoints
logging_steps       | Auto         | Steps between console logs  
save_total_limit    | 3            | Maximum checkpoints to keep

----------------------------------------
FULL CUSTOM CONTROL EXAMPLE
----------------------------------------

tuner.train(
    dataset=data,
    output_dir="./model",
    num_epochs=5,
    batch_size=2,
    gradient_accumulation_steps=4,
    save_steps=100,          # Save every 100 steps
    logging_steps=10,      # Log every 10 steps
    warmup_steps=50,       # Custom warmup
)

================================================================================
                           END OF DOCUMENT
================================================================================
