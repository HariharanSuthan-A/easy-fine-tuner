<div style="text-align: center;">
  <img src="assets/banner.png" alt="EasyFineTuner Banner" width="600">
</div>

# EasyFinetuner 🚀

**Dead simple LLM fine-tuning with Unsloth**

Reduce 50+ lines of complex code to ~5 lines. EasyFinetuner wraps Unsloth to make fine-tuning large language models accessible to beginners while remaining powerful for experts.

## 🌟 Features

- **🎯 Simple API**: Fine-tune in 4 lines of code
- **⚡ Auto-Configuration**: Automatically detects optimal settings based on your GPU and data
- **📊 Multi-Format Support**: Works with JSON, JSONL, CSV, DataFrames, or HuggingFace datasets
- **🔧 Smart Templates**: Auto-detects and applies the right prompt template (Alpaca, ChatML, etc.)
- **📈 Built-in Evaluation**: Perplexity, BLEU, and ROUGE metrics
- **💾 Easy Export**: One-line GGUF export for llama.cpp
- **💻 Memory Efficient**: 4-bit quantization with automatic VRAM optimization

## 🚀 Quick Start

```python
from easyfinetuner import FineTuner

# Load and train in 4 lines
tuner = FineTuner(model_name="unsloth/Qwen3-1.7B")
tuner.train(dataset="data.json", output_dir="my_model")
metrics = tuner.evaluate()
tuner.export_gguf(quantization="q4_k_m")
```

## 📦 Installation

```bash
pip install easyfinetuner
```

Or install from source:
```bash
git clone https://github.com/yourusername/easyfinetuner.git
cd easyfinetuner
pip install -e .
```

## 📝 Usage Guide

### Basic Training

```python
from easyfinetuner import FineTuner

# Initialize with any HuggingFace model
tuner = FineTuner(
    model_name="unsloth/Qwen3-1.7B",
    max_seq_length=2048,
    load_in_4bit=True
)

# Train on your data
tuner.train(
    dataset="path/to/train.json",
    output_dir="./output",
    num_epochs=3,
    learning_rate=2e-4
)
```

### Training Data Formats

EasyFinetuner accepts data in multiple formats:

**JSON (List of objects):**
```json
[
    {"instruction": "What is 2+2?", "input": "", "output": "4"},
    {"instruction": "Capital of France?", "input": "", "output": "Paris"}
]
```

**JSONL (One object per line):**
```jsonl
{"input": "Hello", "output": "Hi there!"}
{"input": "How are you?", "output": "I'm doing well!"}
```

**CSV:**
```csv
input,output
"What is AI?","AI stands for Artificial Intelligence..."
```

**Python List:**
```python
data = [
    {"question": "What is Python?", "answer": "A programming language"},
    {"question": "What is ML?", "answer": "Machine Learning"}
]
tuner.train(dataset=data)
```

### Auto-Configuration

Let EasyFinetuner figure out the best settings:

```python
tuner = FineTuner("unsloth/Qwen3-1.7B")

# All parameters set to "auto" - will be optimized for your GPU and data
tuner.train(
    dataset="data.json",
    num_epochs="auto",        # Based on dataset size
    learning_rate="auto",       # Based on dataset size
    batch_size="auto",          # Based on available VRAM
    lora_r="auto",              # Based on model size
)
```

### Manual Configuration

Take full control when needed:

```python
tuner.train(
    dataset="data.json",
    output_dir="./output",
    num_epochs=5,
    learning_rate=2e-4,
    batch_size=2,
    lora_r=64,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    weight_decay=0.01,
)
```

### Evaluation

```python
# Evaluate on validation set
metrics = tuner.evaluate(metrics=["perplexity", "bleu", "rouge"])

# Evaluate on custom test set
metrics = tuner.evaluate(
    test_dataset="test.json",
    metrics=["perplexity"],
    num_samples=100,
    generate_report=True
)

print(f"Perplexity: {metrics['perplexity']:.2f}")
```

### Inference

```python
# Generate text
response = tuner.predict("What is machine learning?", max_new_tokens=256)
print(response)

# With custom generation parameters
response = tuner.predict(
    "Explain quantum computing",
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.95
)
```

### Export to GGUF

```python
# Export for llama.cpp
tuner.export_gguf(quantization="q4_k_m")

# Available quantizations:
# - "q4_k_m": Balanced quality/size (recommended)
# - "q5_k_m": Better quality
# - "q8_0": Very high quality
# - "f16": Half precision, no quantization

# Export and upload to HuggingFace
tuner.export_gguf(
    quantization="q4_k_m",
    upload_to_hub=True,
    hf_repo_id="username/my-model",
    hf_token="your_hf_token"
)
```

## 🎓 Examples

### Example 1: Question Answering

```python
from easyfinetuner import FineTuner

# Prepare QA data
train_data = [
    {"question": "What is Python?", "answer": "Python is a high-level programming language."},
    {"question": "What is a neural network?", "answer": "A neural network is a computing system inspired by biological neural networks."},
    # ... more examples
]

# Train
tuner = FineTuner("unsloth/Qwen3-1.7B")
tuner.train(
    dataset=train_data,
    output_dir="qa_model",
    num_epochs=3
)

# Test
answer = tuner.predict("What is deep learning?")
print(answer)
```

### Example 2: Instruction Following (Alpaca Format)

```python
data = [
    {
        "instruction": "Summarize the following text.",
        "input": "The quick brown fox jumps over the lazy dog.",
        "output": "A fox jumps over a dog."
    },
    # ... more examples
]

tuner = FineTuner("unsloth/Qwen3-1.7B", template="alpaca")
tuner.train(dataset=data, output_dir="summarizer")
```

### Example 3: Chat Format

```python
data = [
    {
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"}
        ]
    },
    # ... more examples
]

tuner = FineTuner("unsloth/Qwen3-1.7B", template="chatml")
tuner.train(dataset=data, output_dir="chatbot")
```

## ⚙️ Advanced Features

### Custom Templates

```python
from easyfinetuner import FineTuner
from easyfinetuner.data_processor import TEMPLATES

# Use built-in templates
tuner = FineTuner("unsloth/Qwen3-1.7B", template="alpaca")  # or "chatml", "plain"

# Define custom template
TEMPLATES["my_template"] = "Instruction: {instruction}\nInput: {input}\nOutput: {output}"
tuner = FineTuner("unsloth/Qwen3-1.7B", template="my_template")
```

### Resume Training

```python
tuner.train(
    dataset="data.json",
    output_dir="./output",
    resume_from_checkpoint="./output/checkpoint-500"
)
```

### Save/Load Adapters

```python
# Save only LoRA adapters (small, efficient)
tuner.save("path/to/adapters", save_adapter_only=True)

# Save full merged model (larger)
tuner.save("path/to/full_model", save_adapter_only=False)

# Load adapters into a new tuner
new_tuner = FineTuner("unsloth/Qwen3-1.7B")
new_tuner.load_adapters("path/to/adapters")
```

## 🛠️ Supported Models

EasyFinetuner works with any model supported by Unsloth, including:

- **Qwen**: `unsloth/Qwen3-1.7B`, `unsloth/Qwen3-4B`, `unsloth/Qwen3-8B`
- **Llama**: `unsloth/Llama-3.2-3B`, `unsloth/Llama-3.1-8B`
- **Mistral**: `unsloth/Mistral-7B-v0.3`
- **Gemma**: `unsloth/gemma-2-2b`, `unsloth/gemma-2-9b`
- **Phi**: `unsloth/Phi-4`, `unsloth/Phi-3-mini-4k`

## 💾 GPU Memory Requirements

| Model Size | 4-bit Training | 4-bit Inference |
|------------|---------------|-----------------|
| 1.7B       | 4 GB          | 2 GB            |
| 3-4B       | 6 GB          | 3 GB            |
| 7-8B       | 10 GB         | 5 GB            |
| 14B        | 18 GB         | 9 GB            |

## 🔧 Troubleshooting

### Out of Memory Error

If you get OOM errors, try:

```python
tuner.train(
    dataset="data.json",
    batch_size=1,              # Reduce batch size
    max_seq_length=1024,       # Reduce sequence length
    load_in_4bit=True,         # Ensure 4-bit is enabled
    gradient_accumulation_steps=8,  # Increase to maintain effective batch
)
```

### Slow Training

```python
tuner.train(
    dataset="data.json",
    num_epochs=1,              # Reduce epochs
    batch_size=4,              # Increase if VRAM allows
    logging_steps=50,          # Log less frequently
)
```

## 📚 API Reference

### FineTuner Class

```python
FineTuner(
    model_name: str,                    # HuggingFace model name
    max_seq_length: Union[int, str],     # Max sequence length (auto)
    load_in_4bit: bool = True,           # Use 4-bit quantization
    template: str = "auto",               # Prompt template (auto)
)

.train(
    dataset: Union[str, List, Dataset],   # Training data
    output_dir: str = "outputs",         # Output directory
    num_epochs: Union[int, str] = "auto", # Number of epochs
    learning_rate: Union[float, str] = "auto",
    batch_size: Union[int, str] = "auto",
    lora_r: Union[int, str] = "auto",
    validation_split: float = 0.1,
    **kwargs
) -> Dict[str, Any]

.evaluate(
    test_dataset: Optional[Dataset] = None,
    metrics: List[str] = ["perplexity"],
    num_samples: int = 100,
    generate_report: bool = False
) -> Dict[str, Any]

.predict(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    **kwargs
) -> str

.export_gguf(
    output_path: str = "model.gguf",
    quantization: str = "q4_k_m",
    upload_to_hub: bool = False,
    hf_repo_id: str = None,
    hf_token: str = None
) -> str
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- Uses [TRL](https://github.com/huggingface/trl) for training
- Inspired by the need for simpler LLM fine-tuning tools

---

**@easy-fine-tuner🚀**
