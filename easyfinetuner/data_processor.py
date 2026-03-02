"""
Data processing utilities for loading and formatting training data.
Supports JSON, JSONL, CSV, DataFrames, and HuggingFace datasets.
"""

import json
import os
from typing import Union, List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset


class DataProcessor:
    """Handle multi-format data loading and prompt formatting."""
    
    # Common field mappings
    FIELD_MAPPINGS = {
        "input": ["input", "question", "prompt", "instruction", "query", "text"],
        "output": ["output", "answer", "completion", "response", "target"],
    }
    
    @staticmethod
    def load_dataset(source: Union[str, List[Dict], pd.DataFrame, Dataset]) -> Dataset:
        """
        Load dataset from various sources.
        
        Args:
            source: File path (json/jsonl/csv), list of dicts, DataFrame, or HF dataset
            
        Returns:
            HuggingFace Dataset object
        """
        # Already a Dataset
        if isinstance(source, Dataset):
            return source
        
        # List of dictionaries
        if isinstance(source, list):
            return Dataset.from_list(source)
        
        # Pandas DataFrame
        if isinstance(source, pd.DataFrame):
            return Dataset.from_pandas(source)
        
        # File path
        if isinstance(source, (str, Path)):
            path = str(source)
            
            if not os.path.exists(path):
                # Try loading from HuggingFace Hub
                try:
                    return load_dataset(path, split="train")
                except Exception as e:
                    raise ValueError(f"Could not load dataset from '{path}': {e}")
            
            # Determine format from extension
            ext = Path(path).suffix.lower()
            
            if ext == '.json':
                return DataProcessor._load_json(path)
            elif ext == '.jsonl':
                return DataProcessor._load_jsonl(path)
            elif ext == '.csv':
                return DataProcessor._load_csv(path)
            elif ext in ['.parquet', '.pq']:
                return DataProcessor._load_parquet(path)
            else:
                # Try JSON as default
                try:
                    return DataProcessor._load_json(path)
                except:
                    raise ValueError(f"Unsupported file format: {ext}. Use .json, .jsonl, .csv, or .parquet")
        
        raise ValueError(f"Unsupported data source type: {type(source)}")
    
    @staticmethod
    def _load_json(path: str) -> Dataset:
        """Load JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            return Dataset.from_list(data)
        elif isinstance(data, dict):
            # Might be a dataset dict, try to extract
            if 'data' in data:
                return Dataset.from_list(data['data'])
            else:
                return Dataset.from_list([data])
        else:
            raise ValueError(f"Unexpected JSON structure in {path}")
    
    @staticmethod
    def _load_jsonl(path: str) -> Dataset:
        """Load JSONL file."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}")
        
        return Dataset.from_list(data)
    
    @staticmethod
    def _load_csv(path: str) -> Dataset:
        """Load CSV file."""
        df = pd.read_csv(path)
        return Dataset.from_pandas(df)
    
    @staticmethod
    def _load_parquet(path: str) -> Dataset:
        """Load Parquet file."""
        df = pd.read_parquet(path)
        return Dataset.from_pandas(df)
    
    @staticmethod
    def auto_detect_format(sample: Dict[str, Any]) -> str:
        """
        Auto-detect the format of a dataset sample.
        
        Returns:
            Format name: 'input_output', 'instruction', 'question_answer', 
                       'prompt_completion', 'text_only', 'chat', or 'unknown'
        """
        keys = set(sample.keys())
        
        # Chat format
        if "messages" in keys:
            return "chat"
        
        # Input-Output format
        if {"input", "output"}.issubset(keys):
            return "input_output"
        
        # Instruction format (Alpaca-style)
        if {"instruction", "output"}.issubset(keys):
            return "instruction"
        
        # Question-Answer format
        if {"question", "answer"}.issubset(keys):
            return "question_answer"
        
        # Prompt-Completion format
        if {"prompt", "completion"}.issubset(keys):
            return "prompt_completion"
        
        # Text-only format
        if "text" in keys:
            return "text_only"
        
        # Context-Response format
        if {"context", "response"}.issubset(keys):
            return "context_response"
        
        return "unknown"
    
    @staticmethod
    def format_prompt(example: Dict[str, Any], template: str = "auto") -> str:
        """
        Apply a prompt template to format the example.
        
        Args:
            example: Dictionary with input/output data
            template: Template name ('auto', 'alpaca', 'chatml', 'plain') or custom string
            
        Returns:
            Formatted prompt string
        """
        if template == "auto":
            detected = DataProcessor.auto_detect_format(example)
            template = DataProcessor._get_default_template(detected)
        
        # Use built-in templates
        if template in TEMPLATES:
            template_str = TEMPLATES[template]
        else:
            template_str = template  # Custom template
        
        # Format with example values
        try:
            formatted = template_str.format(**example)
        except KeyError as e:
            # Try to map similar keys
            mapped = DataProcessor._map_keys(example)
            try:
                formatted = template_str.format(**mapped)
            except KeyError:
                raise ValueError(f"Template requires key {e} but not found in example. Available keys: {list(example.keys())}")
        
        return formatted
    
    @staticmethod
    def _get_default_template(format_type: str) -> str:
        """Get default template for detected format."""
        defaults = {
            "input_output": "alpaca",
            "instruction": "alpaca",
            "question_answer": "plain",
            "prompt_completion": "plain",
            "text_only": "text",
            "chat": "chatml",
            "context_response": "alpaca",
        }
        return defaults.get(format_type, "plain")
    
    @staticmethod
    def _map_keys(example: Dict[str, Any]) -> Dict[str, Any]:
        """Map common alternative key names to standard keys."""
        mapped = dict(example)
        
        # Map to 'input'
        for key in ["question", "prompt", "query", "instruction", "context"]:
            if key in example and "input" not in example:
                mapped["input"] = example[key]
                break
        
        # Map to 'output'
        for key in ["answer", "completion", "response", "target"]:
            if key in example and "output" not in example:
                mapped["output"] = example[key]
                break
        
        return mapped
    
    @staticmethod
    def analyze_dataset(dataset: Dataset) -> Dict[str, Any]:
        """
        Analyze dataset statistics for auto-configuration.
        
        Returns:
            Dictionary with statistics: num_examples, avg_length, max_length, 
            min_length, format_type
        """
        num_examples = len(dataset)
        
        # Detect format from first example
        if num_examples > 0:
            sample = dataset[0]
            format_type = DataProcessor.auto_detect_format(sample)
        else:
            format_type = "unknown"
        
        # Analyze text lengths
        lengths = []
        
        for i in range(min(num_examples, 1000)):  # Sample up to 1000
            example = dataset[i]
            
            # Try to get text content
            text_parts = []
            
            for key in ["input", "instruction", "question", "prompt", "text", "context"]:
                if key in example:
                    text_parts.append(str(example[key]))
                    break
            
            for key in ["output", "answer", "completion", "response", "target"]:
                if key in example:
                    text_parts.append(str(example[key]))
                    break
            
            if text_parts:
                full_text = " ".join(text_parts)
                lengths.append(len(full_text.split()))  # Word count
            elif "messages" in example:
                # Chat format
                total_words = 0
                for msg in example["messages"]:
                    if isinstance(msg, dict) and "content" in msg:
                        total_words += len(str(msg["content"]).split())
                lengths.append(total_words)
        
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            max_length = max(lengths)
            min_length = min(lengths)
            p95_length = sorted(lengths)[int(len(lengths) * 0.95)] if len(lengths) > 20 else max_length
        else:
            avg_length = 512
            max_length = 2048
            min_length = 10
            p95_length = 1024
        
        return {
            "num_examples": num_examples,
            "avg_length": int(avg_length),
            "max_length": int(max_length),
            "min_length": int(min_length),
            "p95_length": int(p95_length),
            "format_type": format_type,
            "sample_keys": list(dataset[0].keys()) if num_examples > 0 else [],
        }
    
    @staticmethod
    def prepare_for_training(
        dataset: Dataset,
        template: str = "auto",
        tokenizer=None
    ) -> Dataset:
        """
        Prepare dataset for training by applying templates.
        
        Args:
            dataset: Input dataset
            template: Template to apply
            tokenizer: Tokenizer for additional processing (optional)
            
        Returns:
            Processed dataset with 'text' field
        """
        def format_example(example):
            if "text" in example and template == "auto":
                # Already has text field
                return example
            
            formatted_text = DataProcessor.format_prompt(example, template)
            return {"text": formatted_text}
        
        # Apply formatting
        formatted_dataset = dataset.map(format_example)
        
        return formatted_dataset
    
    @staticmethod
    def split_dataset(
        dataset: Dataset,
        validation_split: float = 0.1,
        seed: int = 3407
    ) -> tuple[Dataset, Optional[Dataset]]:
        """
        Split dataset into train and validation sets.
        
        Args:
            dataset: Input dataset
            validation_split: Fraction for validation (0.0 to 1.0)
            seed: Random seed
            
        Returns:
            Tuple of (train_dataset, val_dataset or None)
        """
        if validation_split <= 0 or validation_split >= 1:
            return dataset, None
        
        # Shuffle and split
        dataset = dataset.shuffle(seed=seed)
        
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        
        if val_size == 0:
            return dataset, None
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, len(dataset)))
        
        return train_dataset, val_dataset
    
    @staticmethod
    def save_dataset(dataset: Dataset, path: str, format: str = "auto"):
        """
        Save dataset to file.
        
        Args:
            dataset: Dataset to save
            path: Output file path
            format: Format to use (auto, json, jsonl, csv, parquet)
        """
        if format == "auto":
            ext = Path(path).suffix.lower()
            if ext == '.json':
                format = "json"
            elif ext == '.jsonl':
                format = "jsonl"
            elif ext == '.csv':
                format = "csv"
            elif ext in ['.parquet', '.pq']:
                format = "parquet"
            else:
                format = "jsonl"
        
        # Convert to appropriate format
        if format == "json":
            data = dataset.to_list()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        elif format == "jsonl":
            data = dataset.to_list()
            with open(path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
        elif format == "csv":
            df = dataset.to_pandas()
            df.to_csv(path, index=False)
            
        elif format == "parquet":
            df = dataset.to_pandas()
            df.to_parquet(path, index=False)
            
        else:
            raise ValueError(f"Unknown format: {format}")


# Built-in prompt templates
TEMPLATES = {
    "alpaca": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}<|endoftext|>"
    ),
    "alpaca_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n{output}<|endoftext|>"
    ),
    "chatml": (
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n{output}<|im_end|><|endoftext|>"
    ),
    "plain": "{input}\n\n{output}<|endoftext|>",
    "qa": "Question: {question}\n\nAnswer: {answer}<|endoftext|>",
    "text": "{text}<|endoftext|>",
}
