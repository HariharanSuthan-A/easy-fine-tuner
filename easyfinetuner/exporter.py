"""
GGUF export functionality for saving fine-tuned models.
Integrates with Unsloth's export capabilities.
"""

import os
from typing import Optional
from pathlib import Path


class GGUFExporter:
    """Export fine-tuned models to GGUF format for llama.cpp."""
    
    # Valid quantization methods
    QUANTIZATION_METHODS = [
        "q4_k_m",   # Balanced quality/size (recommended)
        "q5_k_m",   # Better quality
        "q5_k_s",   # Q5 with small buffer
        "q6_k",     # High quality
        "q8_0",     # Very high quality
        "f16",      # Half precision (no quantization)
        "q4_0",     # Legacy Q4
        "q4_1",     # Legacy Q4 variant
        "q5_0",     # Legacy Q5
        "q5_1",     # Legacy Q5 variant
    ]
    
    @staticmethod
    def validate_quantization(quantization: str) -> str:
        """Validate and normalize quantization method."""
        q = quantization.lower().replace("-", "_")
        
        if q not in GGUFExporter.QUANTIZATION_METHODS:
            raise ValueError(
                f"Invalid quantization: {quantization}. "
                f"Choose from: {', '.join(GGUFExporter.QUANTIZATION_METHODS)}"
            )
        
        return q
    
    @staticmethod
    def merge_and_export(
        model,
        tokenizer,
        output_path: str,
        quantization: str = "q4_k_m"
    ) -> str:
        """
        Merge LoRA adapters with base model and export to GGUF.
        
        Args:
            model: The model (with LoRA adapters)
            tokenizer: The tokenizer
            output_path: Path for output GGUF file
            quantization: Quantization method (q4_k_m, q5_k_m, q8_0, f16, etc.)
            
        Returns:
            Path to exported GGUF file
        """
        quantization = GGUFExporter.validate_quantization(quantization)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Use Unsloth's export if available
        try:
            from unsloth import FastLanguageModel
            
            # First save merged model to temp directory
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                merged_path = os.path.join(tmpdir, "merged")
                
                # Save merged model
                model.save_pretrained_merged(
                    merged_path,
                    tokenizer,
                    save_method="merged_16bit"
                )
                
                # Then export to GGUF
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import subprocess
                
                # Use llama.cpp convert script if available
                gguf_path = output_path
                if not gguf_path.endswith('.gguf'):
                    gguf_path += '.gguf'
                
                # Try using unsloth's native GGUF export
                try:
                    model.save_pretrained_gguf(
                        output_path.replace('.gguf', ''),
                        tokenizer,
                        quantization_method=quantization
                    )
                except:
                    # Fallback: manual GGUF conversion
                    GGUFExporter._manual_gguf_export(
                        merged_path,
                        gguf_path,
                        quantization
                    )
                
                return gguf_path
                
        except ImportError:
            # Fallback for non-Unsloth models
            return GGUFExporter._manual_export(model, tokenizer, output_path, quantization)
    
    @staticmethod
    def _manual_export(model, tokenizer, output_path: str, quantization: str) -> str:
        """Fallback export method using transformers and manual conversion."""
        # Save merged model
        merged_dir = output_path.replace('.gguf', '_merged')
        os.makedirs(merged_dir, exist_ok=True)
        
        print(f"Saving merged model to {merged_dir}...")
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        
        # Try to use llama.cpp for conversion if available
        gguf_path = output_path if output_path.endswith('.gguf') else output_path + '.gguf'
        
        GGUFExporter._manual_gguf_export(merged_dir, gguf_path, quantization)
        
        return gguf_path
    
    @staticmethod
    def _manual_gguf_export(merged_dir: str, output_path: str, quantization: str):
        """Attempt to use llama.cpp for GGUF conversion."""
        import subprocess
        import sys
        
        # Check for llama.cpp convert script
        convert_scripts = [
            "convert-hf-to-gguf.py",
            "convert.py",
        ]
        
        convert_script = None
        for script in convert_scripts:
            result = subprocess.run(["which", script], capture_output=True)
            if result.returncode == 0:
                convert_script = script
                break
        
        if convert_script:
            print(f"Using {convert_script} for GGUF conversion...")
            cmd = [
                sys.executable, convert_script,
                "--outfile", output_path,
                "--outtype", quantization,
                merged_dir
            ]
            subprocess.run(cmd, check=True)
        else:
            print("Warning: llama.cpp convert script not found.")
            print(f"Model saved in HuggingFace format at: {merged_dir}")
            print("To convert to GGUF, install llama.cpp and run:")
            print(f"  python convert-hf-to-gguf.py --outfile {output_path} --outtype {quantization} {merged_dir}")
            return merged_dir
    
    @staticmethod
    def upload_to_hub(
        model_path: str,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        model_card: Optional[dict] = None
    ):
        """
        Upload exported model to HuggingFace Hub.
        
        Args:
            model_path: Path to GGUF file or directory
            repo_id: HuggingFace repo ID (username/repo-name)
            token: HuggingFace API token
            private: Whether to create private repo
            model_card: Optional model card info
        """
        try:
            from huggingface_hub import HfApi, create_repo
            
            api = HfApi(token=token)
            
            # Create repo if it doesn't exist
            try:
                create_repo(repo_id, token=token, private=private, exist_ok=True)
            except Exception as e:
                print(f"Note: Repo may already exist: {e}")
            
            # Upload GGUF file
            if os.path.isfile(model_path):
                print(f"Uploading {model_path} to {repo_id}...")
                api.upload_file(
                    path_or_fileobj=model_path,
                    path_in_repo=os.path.basename(model_path),
                    repo_id=repo_id,
                    token=token
                )
            else:
                # Upload directory
                print(f"Uploading directory {model_path} to {repo_id}...")
                api.upload_folder(
                    folder_path=model_path,
                    repo_id=repo_id,
                    token=token
                )
            
            # Upload model card if provided
            if model_card:
                readme = GGUFExporter._generate_model_card(model_card)
                api.upload_file(
                    path_or_fileobj=readme.encode(),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    token=token
                )
            
            print(f"Successfully uploaded to https://huggingface.co/{repo_id}")
            
        except ImportError:
            raise ImportError("Please install huggingface_hub: pip install huggingface_hub")
        except Exception as e:
            raise RuntimeError(f"Failed to upload to HuggingFace Hub: {e}")
    
    @staticmethod
    def _generate_model_card(info: dict) -> str:
        """Generate a model card README."""
        card = f"""---
language:
- en
license: other
---

# {info.get('model_name', 'Fine-tuned Model')}

This model was fine-tuned using [easyfinetuner](https://github.com/yourusername/easyfinetuner).

## Model Details

- **Base Model:** {info.get('base_model', 'Unknown')}
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Quantization:** {info.get('quantization', 'None')}

## Training Details

- **Dataset Size:** {info.get('dataset_size', 'Unknown')} examples
- **Epochs:** {info.get('num_epochs', 'Unknown')}
- **Learning Rate:** {info.get('learning_rate', 'Unknown')}
- **LoRA Rank:** {info.get('lora_r', 'Unknown')}
- **Final Loss:** {info.get('final_loss', 'Unknown')}

## Usage

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{info.get('repo_id', 'your-model')}",
    max_seq_length={info.get('max_seq_length', 2048)},
    load_in_4bit=True,
)

inputs = tokenizer("Your prompt here", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Intended Use

This model is intended for research and educational purposes.

## Limitations

- May produce inaccurate or biased outputs
- Should not be used for critical decisions without verification
- Performance varies by use case
"""
        return card
    
    @staticmethod
    def get_quantization_info(quantization: str) -> dict:
        """Get information about a quantization method."""
        info = {
            "q4_k_m": {
                "bits": 4,
                "description": "Q4_K_M - Balanced quality and size",
                "recommended": True,
                "size_ratio": 0.25,
            },
            "q5_k_m": {
                "bits": 5,
                "description": "Q5_K_M - Better quality, slightly larger",
                "recommended": True,
                "size_ratio": 0.31,
            },
            "q6_k": {
                "bits": 6,
                "description": "Q6_K - High quality",
                "recommended": False,
                "size_ratio": 0.38,
            },
            "q8_0": {
                "bits": 8,
                "description": "Q8_0 - Very high quality",
                "recommended": False,
                "size_ratio": 0.50,
            },
            "f16": {
                "bits": 16,
                "description": "F16 - Half precision, no quantization",
                "recommended": False,
                "size_ratio": 1.0,
            },
        }
        return info.get(quantization, {"bits": 0, "description": "Unknown", "recommended": False})
