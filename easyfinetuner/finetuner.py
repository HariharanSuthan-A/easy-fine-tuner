"""
Main FineTuner class - the core of easyfinetuner.
Wraps Unsloth to provide a simple interface for LLM fine-tuning.
"""

import os

# Disable wandb by default to avoid import issues
os.environ["WANDB_DISABLED"] = "true"

import warnings
from typing import Union, List, Dict, Any, Optional, Callable
from pathlib import Path

import torch
from datasets import Dataset
from tqdm.auto import tqdm

from .config import get_optimal_config, estimate_model_size, validate_config, get_gpu_memory
from .data_processor import DataProcessor, TEMPLATES
from .evaluator import Evaluator
from .exporter import GGUFExporter
from .utils import (
    get_device_info,
    print_device_info,
    print_training_config,
    create_output_dir,
    setup_logging,
    save_config,
)


class FineTuner:
    """
    Dead simple LLM fine-tuning with Unsloth.
    
    Example:
        from easyfinetuner import FineTuner
        
        tuner = FineTuner(model_name="unsloth/Qwen3-1.7B")
        tuner.train(dataset="data.json", output_dir="my_model")
        metrics = tuner.evaluate()
        tuner.export_gguf(quantization="q4_k_m")
    """
    
    def __init__(
        self,
        model_name: str,
        max_seq_length: Union[int, str] = "auto",
        load_in_4bit: bool = True,
        template: str = "auto",
        device_map: str = "auto",
        **kwargs
    ):
        """
        Initialize the FineTuner with a base model.
        
        Args:
            model_name: HuggingFace model name or path (e.g., "unsloth/Qwen3-1.7B")
            max_seq_length: Maximum sequence length ("auto" to detect from data)
            load_in_4bit: Use 4-bit quantization to save memory
            template: Prompt template to use ("auto", "alpaca", "chatml", "plain")
            device_map: Device mapping strategy
            **kwargs: Additional arguments for model loading
        """
        self.model_name = model_name
        self.max_seq_length_setting = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.template = template
        self.device_map = device_map
        self.model_kwargs = kwargs
        
        # These will be set during train()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.val_dataset = None
        self.training_stats = {}
        self.config = {}
        self.output_dir = None
        self.logger = None
        
        # Print device info
        print_device_info()
        
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available. Fine-tuning will be extremely slow on CPU!")
    
    def train(
        self,
        dataset: Union[str, List[Dict], Dataset, "pd.DataFrame"],
        output_dir: str = "outputs",
        num_epochs: Union[int, str] = "auto",
        learning_rate: Union[float, str] = "auto",
        batch_size: Union[int, str] = "auto",
        lora_r: Union[int, str] = "auto",
        validation_split: float = 0.1,
        gradient_accumulation_steps: Union[int, str] = "auto",
        warmup_steps: Union[int, str] = "auto",
        weight_decay: float = 0.01,
        logging_steps: int = None,
        save_steps: int = None,
        seed: int = 3407,
        resume_from_checkpoint: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fine-tune the model on your dataset.
        
        Args:
            dataset: Training data (file path, list of dicts, DataFrame, or Dataset)
            output_dir: Directory to save model checkpoints
            num_epochs: Number of training epochs ("auto" based on dataset size)
            learning_rate: Learning rate ("auto" for 5e-5 to 5e-4)
            batch_size: Batch size per device ("auto" based on VRAM)
            lora_r: LoRA rank ("auto": 16/32/64 based on model size)
            validation_split: Fraction of data for validation (0.0-1.0)
            gradient_accumulation_steps: Steps to accumulate gradients ("auto")
            warmup_steps: Warmup steps ("auto")
            weight_decay: Weight decay for regularization
            logging_steps: Steps between logging ("auto")
            save_steps: Steps between checkpoints ("auto")
            seed: Random seed for reproducibility
            resume_from_checkpoint: Resume from checkpoint path
            **kwargs: Additional arguments for TrainingArguments
            
        Returns:
            Dictionary with training statistics
        """
        try:
            from unsloth import FastLanguageModel, is_bfloat16_supported
            from trl import SFTTrainer
            from transformers import TrainingArguments
            from peft import LoraConfig
            
            # Handle different TRL versions for DataCollatorForCompletionOnlyLM
            try:
                from trl import DataCollatorForCompletionOnlyLM
            except ImportError:
                try:
                    from trl.trainer import DataCollatorForCompletionOnlyLM
                except ImportError:
                    # Fallback: define a simple version
                    DataCollatorForCompletionOnlyLM = None
        except ImportError as e:
            raise ImportError(
                f"Required dependencies not installed: {e}\n"
                "Install with: pip install unsloth trl peft transformers accelerate"
            )
        
        print(f"\n{'='*50}")
        print(f"Starting Fine-Tuning: {self.model_name}")
        print(f"{'='*50}\n")
        
        # Step 1: Load and analyze dataset
        print("Step 1/5: Loading dataset...")
        full_dataset = DataProcessor.load_dataset(dataset)
        dataset_stats = DataProcessor.analyze_dataset(full_dataset)
        
        print(f"  Dataset size: {dataset_stats['num_examples']} examples")
        print(f"  Average length: {dataset_stats['avg_length']} tokens")
        print(f"  Max length: {dataset_stats['max_length']} tokens")
        print(f"  Detected format: {dataset_stats['format_type']}")
        
        # Step 2: Split dataset
        print("\nStep 2/5: Splitting dataset...")
        self.train_dataset, self.val_dataset = DataProcessor.split_dataset(
            full_dataset,
            validation_split=validation_split
        )
        
        train_size = len(self.train_dataset)
        val_size = len(self.val_dataset) if self.val_dataset else 0
        print(f"  Training: {train_size} examples")
        print(f"  Validation: {val_size} examples")
        
        # Step 3: Get optimal configuration
        print("\nStep 3/5: Computing optimal configuration...")
        
        auto_config = get_optimal_config(
            self.model_name,
            dataset_stats,
            get_gpu_memory(),
            self.load_in_4bit
        )
        
        # Override with user-provided values
        self.config = {
            "model_name": self.model_name,
            "max_seq_length": auto_config["max_seq_length"] if self.max_seq_length_setting == "auto" else self.max_seq_length_setting,
            "batch_size": auto_config["batch_size"] if batch_size == "auto" else batch_size,
            "gradient_accumulation_steps": auto_config["gradient_accumulation_steps"] if gradient_accumulation_steps == "auto" else gradient_accumulation_steps,
            "num_epochs": auto_config["num_epochs"] if num_epochs == "auto" else num_epochs,
            "learning_rate": auto_config["learning_rate"] if learning_rate == "auto" else learning_rate,
            "lora_r": auto_config["lora_r"] if lora_r == "auto" else lora_r,
            "lora_alpha": auto_config["lora_alpha"],
            "lora_dropout": 0,
            "warmup_steps": auto_config["warmup_steps"] if warmup_steps == "auto" else warmup_steps,
            "weight_decay": weight_decay,
            "logging_steps": auto_config["logging_steps"] if logging_steps is None else logging_steps,
            "save_steps": auto_config["save_steps"] if save_steps is None else save_steps,
            "lr_scheduler_type": auto_config["lr_scheduler_type"],
            "seed": seed,
            "output_dir": output_dir,
            **kwargs
        }
        
        # Validate configuration
        model_size = estimate_model_size(self.model_name)
        validate_config(self.config, get_gpu_memory(), model_size)
        
        print_training_config(self.config)
        
        # Step 4: Load model and apply LoRA
        print("\nStep 4/5: Loading model and applying LoRA...")
        
        max_seq_length = self.config["max_seq_length"]
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
            device_map=self.device_map,
            **self.model_kwargs
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_rslora=False,
        )
        
        self.model = FastLanguageModel.get_peft_model(self.model, lora_config)
        
        model_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {model_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/model_params:.2f}%)")
        
        # Step 5: Setup trainer and train
        print("\nStep 5/5: Starting training...")
        print(f"  Output directory: {output_dir}")
        
        self.output_dir = create_output_dir(output_dir, self.model_name)
        self.logger = setup_logging(self.output_dir)
        
        # Save config
        save_config(self.config, os.path.join(self.output_dir, "config.json"))
        
        # Detect template
        if self.template == "auto":
            self.template = DataProcessor._get_default_template(dataset_stats["format_type"])
            print(f"  Using template: {self.template}")
        
        # Prepare dataset for training
        formatted_dataset = DataProcessor.prepare_for_training(
            self.train_dataset,
            template=self.template,
            tokenizer=self.tokenizer
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config["num_epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            learning_rate=self.config["learning_rate"],
            warmup_steps=self.config["warmup_steps"],
            weight_decay=self.config["weight_decay"],
            logging_steps=self.config["logging_steps"],
            save_steps=self.config["save_steps"],
            save_total_limit=3,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            logging_dir=f"{self.output_dir}/logs",
            lr_scheduler_type=self.config["lr_scheduler_type"],
            seed=seed,
            report_to="none",  # Disable wandb/tensorboard by default
            remove_unused_columns=False,
            **kwargs
        )
        
        # Setup validation dataset if available
        eval_dataset = None
        if self.val_dataset:
            eval_dataset = DataProcessor.prepare_for_training(
                self.val_dataset,
                template=self.template,
                tokenizer=self.tokenizer
            )
            training_args.eval_strategy = "steps"
            training_args.eval_steps = self.config["save_steps"]
            training_args.load_best_model_at_end = True
        
        # Build trainer kwargs
        trainer_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "train_dataset": formatted_dataset,
            "eval_dataset": eval_dataset,
            "dataset_text_field": "text",
            "max_seq_length": max_seq_length,
            "dataset_num_proc": 2,
            "packing": False,
            "args": training_args,
        }
        
        # Add data collator only if available
        if DataCollatorForCompletionOnlyLM is not None:
            trainer_kwargs["data_collator"] = DataCollatorForCompletionOnlyLM(
                tokenizer=self.tokenizer,
                mlm=False,
            )
        
        # Initialize trainer
        self.trainer = SFTTrainer(**trainer_kwargs)
        
        # Train
        print("\n" + "="*50)
        print("Training in progress...")
        print("="*50)
        
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Training stats
        self.training_stats = {
            "final_loss": train_result.training_loss,
            "train_runtime_seconds": train_result.metrics.get("train_runtime", 0),
            "train_runtime_formatted": self._format_time(train_result.metrics.get("train_runtime", 0)),
            "steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
            "samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "total_flos": train_result.metrics.get("total_flos", 0),
            "num_train_samples": train_result.metrics.get("train_samples", 0),
            "output_dir": self.output_dir,
        }
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        print(f"  Final loss: {self.training_stats['final_loss']:.4f}")
        print(f"  Training time: {self.training_stats['train_runtime_formatted']}")
        print(f"  Model saved to: {self.output_dir}")
        print("="*50 + "\n")
        
        return self.training_stats
    
    def evaluate(
        self,
        test_dataset: Union[str, List[Dict], Dataset, "pd.DataFrame", None] = None,
        metrics: List[str] = ["perplexity"],
        num_samples: int = 100,
        generate_report: bool = False,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate the fine-tuned model.
        
        Args:
            test_dataset: Test data (uses validation split if None)
            metrics: List of metrics to compute ("perplexity", "bleu", "rouge")
            num_samples: Number of samples to evaluate
            generate_report: Whether to generate HTML report
            output_path: Path for report output
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be trained before evaluation. Call train() first.")
        
        print(f"\n{'='*50}")
        print("Evaluation")
        print(f"{'='*50}\n")
        
        # Get test dataset
        if test_dataset is None:
            if self.val_dataset is None:
                raise ValueError("No validation dataset available. Provide test_dataset or use validation_split during training.")
            eval_data = self.val_dataset
        else:
            eval_data = DataProcessor.load_dataset(test_dataset)
        
        # Prepare dataset
        formatted_data = DataProcessor.prepare_for_training(
            eval_data,
            template=self.template,
            tokenizer=self.tokenizer
        )
        
        # Initialize evaluator
        evaluator = Evaluator(self.model, self.tokenizer)
        
        # Run evaluation
        results = evaluator.evaluate_all(
            formatted_data,
            metrics=metrics,
            num_samples=num_samples,
            generate_report=generate_report,
            output_path=output_path or (f"{self.output_dir}/evaluation_report.html" if self.output_dir else "evaluation_report.html")
        )
        
        print("\nEvaluation Results:")
        print("-" * 50)
        for metric, value in results.items():
            if isinstance(value, dict):
                print(f"  {metric}:")
                for k, v in value.items():
                    print(f"    {k}: {v:.4f}")
            elif isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        print("=" * 50 + "\n")
        
        return results
    
    def predict(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        **generate_kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            **generate_kwargs: Additional generation arguments
            
        Returns:
            Generated text string
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        self.model.eval()
        
        # Format prompt if needed
        if self.template != "auto" and self.template in TEMPLATES:
            # Check if prompt already looks formatted
            if "<|endoftext|>" not in prompt and "###" not in prompt:
                # Apply template
                template_str = TEMPLATES[self.template]
                if "{input}" in template_str and "{output}" in template_str:
                    # Remove output part for inference
                    prompt = template_str.split("{output}")[0].format(input=prompt, instruction=prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return result.strip()
    
    def export_gguf(
        self,
        output_path: str = None,
        quantization: str = "q4_k_m",
        upload_to_hub: bool = False,
        hf_repo_id: str = None,
        hf_token: str = None,
        **kwargs
    ) -> str:
        """
        Export the fine-tuned model to GGUF format.
        
        Args:
            output_path: Path for output GGUF file (default: model.gguf in output_dir)
            quantization: Quantization method (q4_k_m, q5_k_m, q8_0, f16)
            upload_to_hub: Whether to upload to HuggingFace Hub
            hf_repo_id: HuggingFace repo ID for upload
            hf_token: HuggingFace API token
            **kwargs: Additional export arguments
            
        Returns:
            Path to exported GGUF file
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be trained before export. Call train() first.")
        
        print(f"\n{'='*50}")
        print("Exporting to GGUF")
        print(f"{'='*50}\n")
        
        # Set default output path
        if output_path is None:
            output_path = f"{self.output_dir}/model.gguf" if self.output_dir else "model.gguf"
        
        # Export
        print(f"  Quantization: {quantization}")
        print(f"  Output: {output_path}")
        
        try:
            # Try using unsloth's native export
            from unsloth import FastLanguageModel
            
            # Save to GGUF
            output_dir = os.path.dirname(output_path) or "."
            base_name = os.path.basename(output_path).replace('.gguf', '')
            
            self.model.save_pretrained_gguf(
                output_dir,
                self.tokenizer,
                quantization_method=quantization,
                **kwargs
            )
            
            # Rename if needed
            exported_path = os.path.join(output_dir, f"{base_name}.gguf")
            if os.path.exists(os.path.join(output_dir, "model.gguf")):
                os.rename(os.path.join(output_dir, "model.gguf"), exported_path)
            
            print(f"\n  Successfully exported to: {exported_path}")
            
        except Exception as e:
            print(f"  Native export failed: {e}")
            print("  Trying fallback export method...")
            
            # Use exporter fallback
            exported_path = GGUFExporter.merge_and_export(
                self.model,
                self.tokenizer,
                output_path,
                quantization
            )
            
            print(f"\n  Exported to: {exported_path}")
        
        # Upload to hub if requested
        if upload_to_hub:
            if hf_repo_id is None:
                raise ValueError("hf_repo_id required when upload_to_hub=True")
            
            print(f"\n  Uploading to HuggingFace Hub: {hf_repo_id}")
            
            model_card = {
                "model_name": os.path.basename(hf_repo_id),
                "base_model": self.model_name,
                "quantization": quantization,
                "dataset_size": self.training_stats.get("num_train_samples", "Unknown"),
                "num_epochs": self.config.get("num_epochs", "Unknown"),
                "learning_rate": self.config.get("learning_rate", "Unknown"),
                "lora_r": self.config.get("lora_r", "Unknown"),
                "final_loss": self.training_stats.get("final_loss", "Unknown"),
                "max_seq_length": self.config.get("max_seq_length", "Unknown"),
                "repo_id": hf_repo_id,
            }
            
            GGUFExporter.upload_to_hub(
                exported_path,
                hf_repo_id,
                token=hf_token,
                model_card=model_card
            )
        
        print("=" * 50 + "\n")
        
        return exported_path
    
    def save(self, path: str = None, save_adapter_only: bool = True):
        """
        Save the model.
        
        Args:
            path: Save path (default: output_dir)
            save_adapter_only: Save only LoRA adapters (smaller)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be trained before saving. Call train() first.")
        
        save_path = path or self.output_dir or "./saved_model"
        os.makedirs(save_path, exist_ok=True)
        
        if save_adapter_only:
            # Save only LoRA adapters
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"LoRA adapters saved to: {save_path}")
        else:
            # Save full merged model
            from unsloth import FastLanguageModel
            self.model.save_pretrained_merged(save_path, self.tokenizer)
            print(f"Full model saved to: {save_path}")
    
    def load_adapters(self, adapter_path: str):
        """
        Load LoRA adapters from a previous training run.
        
        Args:
            adapter_path: Path to adapter checkpoint
        """
        if self.model is None:
            raise ValueError("Base model must be loaded first. Use the same model_name as during training.")
        
        from peft import PeftModel
        
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        print(f"Loaded adapters from: {adapter_path}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def __repr__(self):
        return f"FineTuner(model_name='{self.model_name}', template='{self.template}')"
    
    @staticmethod
    def enable_wandb(project_name: str = "easyfinetuner", **kwargs):
        """
        Enable Weights & Biases logging.
        Call this BEFORE creating a FineTuner instance.
        
        Args:
            project_name: W&B project name
            **kwargs: Additional W&B init arguments
            
        Example:
            FineTuner.enable_wandb(project_name="my_experiments")
            tuner = FineTuner("unsloth/Qwen3-1.7B")
            tuner.train(dataset=data)
        """
        # Re-enable wandb
        os.environ["WANDB_DISABLED"] = "false"
        
        try:
            import wandb
            wandb.init(project=project_name, **kwargs)
            print(f"W&B enabled: https://wandb.ai/{project_name}")
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
    
    @staticmethod
    def disable_wandb():
        """Disable Weights & Biases logging (default behavior)."""
        os.environ["WANDB_DISABLED"] = "true"
        print("W&B disabled.")
