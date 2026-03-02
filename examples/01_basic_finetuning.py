"""
Example 1: Basic Fine-tuning with EasyFinetuner

This notebook demonstrates the simplest way to fine-tune a language model.
"""

# %% [markdown]
# # Example 1: Basic Fine-tuning 🚀
# 
# This notebook shows the most basic usage of EasyFinetuner - fine-tuning a small model
# on a simple dataset.

# %%
# Install dependencies (uncomment if needed)
# !pip install easyfinetuner

# %%
from easyfinetuner import FineTuner

# %% [markdown]
# ## 1. Prepare Training Data
# 
# EasyFinetuner accepts data in multiple formats. Here's a simple list of dictionaries:

# %%
train_data = [
    {"input": "What is 2+2?", "output": "4"},
    {"input": "Capital of France?", "output": "Paris"},
    {"input": "Largest planet in our solar system?", "output": "Jupiter"},
    {"input": "Who wrote Romeo and Juliet?", "output": "William Shakespeare"},
    {"input": "What is the speed of light?", "output": "Approximately 299,792 kilometers per second"},
    {"input": "Chemical symbol for gold?", "output": "Au"},
    {"input": "How many continents are there?", "output": "Seven"},
    {"input": "What is photosynthesis?", "output": "The process by which plants convert light energy into chemical energy"},
    {"input": "Who painted the Mona Lisa?", "output": "Leonardo da Vinci"},
    {"input": "What is the capital of Japan?", "output": "Tokyo"},
    {"input": "How many bones in the human body?", "output": "206"},
    {"input": "What is the largest ocean?", "output": "Pacific Ocean"},
    {"input": "Who invented the telephone?", "output": "Alexander Graham Bell"},
    {"input": "What is the smallest prime number?", "output": "2"},
    {"input": "What language is spoken in Brazil?", "output": "Portuguese"},
    {"input": "What is the tallest mountain?", "output": "Mount Everest"},
    {"input": "How many planets in our solar system?", "output": "Eight"},
    {"input": "What is the chemical formula for water?", "output": "H2O"},
    {"input": "Who was the first person on the moon?", "output": "Neil Armstrong"},
    {"input": "What is the longest river?", "output": "Nile River"},
]

print(f"Training data: {len(train_data)} examples")

# %% [markdown]
# ## 2. Initialize the FineTuner
# 
# We'll use Qwen3-1.7B, a small but capable model perfect for learning:

# %%
tuner = FineTuner(
    model_name="unsloth/Qwen3-1.7B",
    max_seq_length=512,
    load_in_4bit=True,
)

# %% [markdown]
# ## 3. Train the Model
# 
# Just one line to train! All parameters are auto-configured.

# %%
stats = tuner.train(
    dataset=train_data,
    output_dir="outputs/qa_model",
    num_epochs=3,
)

print(f"\nTraining complete!")
print(f"Final loss: {stats['final_loss']:.4f}")
print(f"Training time: {stats['train_runtime_formatted']}")

# %% [markdown]
# ## 4. Test the Model
# 
# Let's see how well our model learned:

# %%
test_questions = [
    "What is 5+5?",
    "Capital of Germany?",
    "Who invented the light bulb?",
]

print("Testing the model:\n")
for question in test_questions:
    answer = tuner.predict(question, max_new_tokens=64)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print("-" * 50)

# %% [markdown]
# ## 5. Evaluate the Model
# 
# Get quantitative metrics:

# %%
metrics = tuner.evaluate(
    metrics=["perplexity"],
    num_samples=10
)

print(f"\nPerplexity: {metrics['perplexity']:.2f}")

# %% [markdown]
# ## 6. Export to GGUF
# 
# Save your model for use with llama.cpp or Ollama:

# %%
output_path = tuner.export_gguf(
    output_path="outputs/qa_model.gguf",
    quantization="q4_k_m"
)

print(f"\nModel exported to: {output_path}")

# %% [markdown]
# ## 🎉 Done!
# 
# You've successfully fine-tuned your first model with EasyFinetuner!
# 
# Next steps:
# - Try Example 2 for more advanced features
# - Experiment with different models
# - Fine-tune on your own dataset
